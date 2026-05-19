# 性能优化概念解析

## 1. Register Spill（寄存器溢出）

当程序中活跃变量数量超过CPU可用寄存器数量时，编译器被迫将部分变量暂存到栈内存（spill），需要时再加载回来（reload）。这引入了额外的 load/store 指令，增加延迟。

**产生原因：**
- 循环体内变量过多
- 函数内联后变量膨胀
- 编译器寄存器分配策略不够优化

### Case：矩阵乘法中 unroll 过大导致 spill

```cpp
// BAD: 过度展开，寄存器不够用，产生spill
void matmul_spill(float* C, const float* A, const float* B, int N) {
    for (int i = 0; i < N; i++) {
        // 一次性累加16个结果，需要16个累加器 + 加载寄存器
        float c0=0, c1=0, c2=0, c3=0, c4=0, c5=0, c6=0, c7=0;
        float c8=0, c9=0, c10=0, c11=0, c12=0, c13=0, c14=0, c15=0;
        // 16个累加器 + A/B的加载 → 超出x86的16个XMM寄存器
        for (int k = 0; k < N; k++) {
            float a = A[i*N + k];
            c0 += a * B[k*N+0];   c1 += a * B[k*N+1];
            c2 += a * B[k*N+2];   c3 += a * B[k*N+3];
            c4 += a * B[k*N+4];   c5 += a * B[k*N+5];
            c6 += a * B[k*N+6];   c7 += a * B[k*N+7];
            c8 += a * B[k*N+8];   c9 += a * B[k*N+9];
            c10 += a * B[k*N+10]; c11 += a * B[k*N+11];
            c12 += a * B[k*N+12]; c13 += a * B[k*N+13];
            c14 += a * B[k*N+14]; c15 += a * B[k*N+15];
        }
        C[i*N+0]=c0; C[i*N+1]=c1; /*...*/ C[i*N+15]=c15;
    }
}

// GOOD: 控制展开因子，避免spill
void matmul_no_spill(float* C, const float* A, const float* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {  // 只展开4个
            float c0=0, c1=0, c2=0, c3=0;
            for (int k = 0; k < N; k++) {
                float a = A[i*N + k];
                c0 += a * B[k*N+j];
                c1 += a * B[k*N+j+1];
                c2 += a * B[k*N+j+2];
                c3 += a * B[k*N+j+3];
            }
            C[i*N+j]=c0; C[i*N+j+1]=c1; C[i*N+j+2]=c2; C[i*N+j+3]=c3;
        }
    }
}
```

**验证方法：** 用 `gcc -S -O2` 查看汇编，搜索对 `[rsp+...]` 的 mov 指令，即为 spill。

---

## 2. Register Blocking（寄存器分块）

将计算拆分为小块（tile），使每个小块的工作集完全驻留在寄存器中，最大化数据复用，减少内存访问。

**核心思想：** 如果一个值被加载到寄存器后能被多次使用，就摊薄了 load 的开销。

Register blocking 还是一个偏方法论的概念：它强调的是“把计算结果和热点数据尽量留在寄存器里”。真正把这个思想落到代码上时，常见载体就是 `micro-kernel`。

它和单纯的 loop unroll 不同。unroll 只是把循环展开；register blocking 则会主动按 `MR x NR` 的输出 tile 来规划寄存器中的累加器布局，让一个小块 `C` 在整个 `K` 维累计过程中始终常驻寄存器。

因此，学习顺序可以按这条主线理解：

- **Register blocking**：寄存器级的数据复用思想
- **Micro-kernel**：把这种思想写成最内层的小计算内核
- **Macro-kernel**：在外层高效组织和重复调用 micro-kernel
- **Cache blocking**：继续向外控制数据驻留在哪一级 cache

---

## 3. Micro-Kernel（微内核）

在 GEMM 这类高性能实现里，`micro-kernel` 指最内层的小计算内核，一次只计算一个很小的 `C` tile，比如 `6x16`、`6x48`。它直接面对寄存器、SIMD 指令和 FMA 流水线，是 register blocking 的直接实现。

通常说一个 `6x16 micro-kernel`，本质上就是：固定一次只更新 `C[6,16]` 这个小块，并让它在整个 `K` 维累计过程中尽量常驻寄存器。

> 6x16 micro-kernel的意思是`C`的shape是6x16。也就是：A[6, k] * B[k, 16] = C[6, 16]
> 
> for循环是按k展开，每次循环正好计算一轮：A[6, 1] * B[1, 16] = C[6, 16]
>
> k轮的C[6, 16]累加，就是A[6, k] * B[k, 16]的结果

### Case：GEMM 的寄存器分块（6x16 micro-kernel）

```cpp
#include <immintrin.h>

// 6x16 register blocking micro-kernel for AVX-512
// A: 6 rows, B: 16 columns, K iterations
// 寄存器预算: 6x1(C累加器用zmm) + 1(A broadcast) + 1(B load) = 8个zmm
// AVX-512有32个zmm寄存器，完全够用
void micro_kernel_6x16(int K,
                       const float* A, int lda,
                       const float* B, int ldb,
                       float* C, int ldc) {
    // 6个累加器，每个是512bit = 16 floats
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps();
    __m512 c3 = _mm512_setzero_ps();
    __m512 c4 = _mm512_setzero_ps();
    __m512 c5 = _mm512_setzero_ps();

    for (int k = 0; k < K; k++) {
        __m512 b = _mm512_loadu_ps(&B[k * ldb]);  // load B[k, 0:16]

        // broadcast A[row, k] 并做 FMA — 每个B load被复用6次!
        c0 = _mm512_fmadd_ps(_mm512_set1_ps(A[0*lda + k]), b, c0);
        c1 = _mm512_fmadd_ps(_mm512_set1_ps(A[1*lda + k]), b, c1);
        c2 = _mm512_fmadd_ps(_mm512_set1_ps(A[2*lda + k]), b, c2);
        c3 = _mm512_fmadd_ps(_mm512_set1_ps(A[3*lda + k]), b, c3);
        c4 = _mm512_fmadd_ps(_mm512_set1_ps(A[4*lda + k]), b, c4);
        c5 = _mm512_fmadd_ps(_mm512_set1_ps(A[5*lda + k]), b, c5);
    }

    // store results
    _mm512_storeu_ps(&C[0*ldc], _mm512_add_ps(_mm512_loadu_ps(&C[0*ldc]), c0));
    _mm512_storeu_ps(&C[1*ldc], _mm512_add_ps(_mm512_loadu_ps(&C[1*ldc]), c1));
    _mm512_storeu_ps(&C[2*ldc], _mm512_add_ps(_mm512_loadu_ps(&C[2*ldc]), c2));
    _mm512_storeu_ps(&C[3*ldc], _mm512_add_ps(_mm512_loadu_ps(&C[3*ldc]), c3));
    _mm512_storeu_ps(&C[4*ldc], _mm512_add_ps(_mm512_loadu_ps(&C[4*ldc]), c4));
    _mm512_storeu_ps(&C[5*ldc], _mm512_add_ps(_mm512_loadu_ps(&C[5*ldc]), c5));
}
```

**分析：**
- 每次 K 迭代：1次 B load + 6次 A broadcast + 6次 FMA
- B 的一次 load 被 6 行 A 复用 → 数据复用率 6x
- 6个累加器 + 1个B + 临时broadcast = 8个 zmm 寄存器，远小于32个上限，无 spill

### 核心计算模式：外积累加（Rank-1 Update）

矩阵乘法 `C[6×16] = A[6×K] * B[K×16]` 可以分解为 K 次外积的累加：

```
C[6×16] = Σ(k=0 to K-1)  A[:,k] × B[k,:]
           ─────────────  ─────   ─────
           K次累加         列向量   行向量
                          [6×1]   [1×16]
```

每次迭代 k，取 A 的第 k 列（6个元素）和 B 的第 k 行（16个元素），做外积（outer product）得到一个 6×16 的矩阵，累加到 C 上：

```
k=0 时:

A[:,0]   ×   B[0,:]                       =  外积结果 [6×16]

┌────┐       ┌───────────────────────┐       ┌─────────────────────────┐
│ a0 │       │ b0  b1  b2  ...  b15  │       │a0*b0  a0*b1  ... a0*b15 │
│ a1 │   ×   └───────────────────────┘   =   │a1*b0  a1*b1  ... a1*b15 │
│ a2 │                                       │a2*b0  a2*b1  ... a2*b15 │
│ a3 │       (1×16 行向量)                    │a3*b0  a3*b1  ... a3*b15 │
│ a4 │                                       │a4*b0  a4*b1  ... a4*b15 │
│ a5 │                                       │a5*b0  a5*b1  ... a5*b15 │
└────┘                                       └─────────────────────────┘
(6×1 列向量)                                  (6×16 矩阵)

然后 C += 这个6×16矩阵，循环 K 次得到最终结果。
```

**对应到代码：**
- `_mm512_loadu_ps(&B[k * ldb])` → 取 B 的第 k 行（16个元素）
- `_mm512_set1_ps(A[i*lda + k])` → 把 A 的第 k 列第 i 个元素广播为 [ai, ai, ..., ai]
- `_mm512_fmadd_ps(broadcast_a, b, ci)` → `ci += [ai*b0, ai*b1, ..., ai*b15]`，即外积第 i 行累加

**为什么用外积而不是内积？**

传统理解是"C 的每个元素 = A 的一行点乘 B 的一列"（内积视角），但外积视角更适合高性能实现：

| | 内积视角 | 外积视角 |
|---|---|---|
| 每次计算 | C 的 1 个元素 | C 的整个 6×16 块 |
| B 的复用 | 每个 b 元素只用 1 次 | b 向量被 6 行 A 共享 |
| 适合 SIMD | 不太适合（标量结果） | 天然适合（向量 FMA） |
| 寄存器利用 | 低 | 高（累加器常驻寄存器） |

外积视角让 B 的一次 load 被 6 个 A 元素复用，最大化了数据复用率。

### Case：AVX2 的寄存器分块（6x16 micro-kernel）

AVX2 只有 16 个 ymm 寄存器（256bit = 8 floats），需要更精细的寄存器预算规划。
采用 6x16 的 tile：6 行 x 16 列，但 16 列需要 2 个 ymm 寄存器表示（2x8），所以累加器占 6x2=12 个 ymm。

```cpp
#include <immintrin.h>

// 6x16 register blocking micro-kernel for AVX2 (FMA)
// 寄存器预算: 6x2=12(C累加器) + 2(B load) + 1(A broadcast) = 15个ymm
// AVX2有16个ymm寄存器，刚好够用，紧凑但无spill
void micro_kernel_6x16_avx2(int K,
                            const float* A, int lda,
                            const float* B, int ldb,
                            float* C, int ldc) {
    // 6行 x 2列ymm = 12个累加器
    //           ymm       ymm
    // r0: [0 ................. 7]    [8 ................ 15]
    // r1: [0 ................. 7]    [8 ................ 15]
    // ...
    // r5: [0 ................. 7]    [8 ................ 15]
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();  // row 0
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();  // row 1
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();  // row 2
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();  // row 3
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();  // row 4
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();  // row 5

    for (int k = 0; k < K; k++) {
        // load B[k, 0:8] 和 B[k, 8:16]
        __m256 b0 = _mm256_loadu_ps(&B[k * ldb]);
        __m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);

        __m256 a;
        // broadcast A[row, k] 并做 FMA — 每对B load被复用6次
        a = _mm256_broadcast_ss(&A[0*lda + k]);
        c00 = _mm256_fmadd_ps(a, b0, c00);
        c01 = _mm256_fmadd_ps(a, b1, c01);

        a = _mm256_broadcast_ss(&A[1*lda + k]);
        c10 = _mm256_fmadd_ps(a, b0, c10);
        c11 = _mm256_fmadd_ps(a, b1, c11);

        a = _mm256_broadcast_ss(&A[2*lda + k]);
        c20 = _mm256_fmadd_ps(a, b0, c20);
        c21 = _mm256_fmadd_ps(a, b1, c21);

        a = _mm256_broadcast_ss(&A[3*lda + k]);
        c30 = _mm256_fmadd_ps(a, b0, c30);
        c31 = _mm256_fmadd_ps(a, b1, c31);

        a = _mm256_broadcast_ss(&A[4*lda + k]);
        c40 = _mm256_fmadd_ps(a, b0, c40);
        c41 = _mm256_fmadd_ps(a, b1, c41);

        a = _mm256_broadcast_ss(&A[5*lda + k]);
        c50 = _mm256_fmadd_ps(a, b0, c50);
        c51 = _mm256_fmadd_ps(a, b1, c51);
    }

    // store results: C += accumulated
    _mm256_storeu_ps(&C[0*ldc],   _mm256_add_ps(_mm256_loadu_ps(&C[0*ldc]),   c00));
    _mm256_storeu_ps(&C[0*ldc+8], _mm256_add_ps(_mm256_loadu_ps(&C[0*ldc+8]), c01));
    _mm256_storeu_ps(&C[1*ldc],   _mm256_add_ps(_mm256_loadu_ps(&C[1*ldc]),   c10));
    _mm256_storeu_ps(&C[1*ldc+8], _mm256_add_ps(_mm256_loadu_ps(&C[1*ldc+8]), c11));
    _mm256_storeu_ps(&C[2*ldc],   _mm256_add_ps(_mm256_loadu_ps(&C[2*ldc]),   c20));
    _mm256_storeu_ps(&C[2*ldc+8], _mm256_add_ps(_mm256_loadu_ps(&C[2*ldc+8]), c21));
    _mm256_storeu_ps(&C[3*ldc],   _mm256_add_ps(_mm256_loadu_ps(&C[3*ldc]),   c30));
    _mm256_storeu_ps(&C[3*ldc+8], _mm256_add_ps(_mm256_loadu_ps(&C[3*ldc+8]), c31));
    _mm256_storeu_ps(&C[4*ldc],   _mm256_add_ps(_mm256_loadu_ps(&C[4*ldc]),   c40));
    _mm256_storeu_ps(&C[4*ldc+8], _mm256_add_ps(_mm256_loadu_ps(&C[4*ldc+8]), c41));
    _mm256_storeu_ps(&C[5*ldc],   _mm256_add_ps(_mm256_loadu_ps(&C[5*ldc]),   c50));
    _mm256_storeu_ps(&C[5*ldc+8], _mm256_add_ps(_mm256_loadu_ps(&C[5*ldc+8]), c51));
}
```

**AVX2 vs AVX-512 对比：**

| | AVX-512 (6x16) | AVX2 (6x16) |
|---|---|---|
| 向量宽度 | 512bit (16 floats) | 256bit (8 floats) |
| 累加器数量 | 6 个 zmm | 12 个 ymm (6行 x 2列) |
| 寄存器总数 | 32 个 zmm | 16 个 ymm |
| 累加器占比 | 6/32 = 19% | 12/16 = 75% |
| B load/iter | 1 个 zmm | 2 个 ymm |
| FMA/iter | 6 | 12 |
| Spill风险 | 极低 | 紧凑但安全（15/16） |

**关键设计考量：**
- AVX2 寄存器紧张，复用 `a` 变量做 broadcast 避免额外占用寄存器
- `_mm256_broadcast_ss` 直接从内存 broadcast，不需要先 load 再 shuffle
- 如果改成 8x16 tile（16个累加器），就会超出16个ymm → 产生 spill
- 实际中也可选择 4x24（4行 x 3个ymm列 = 12累加器）等替代方案

### AVX-512 寄存器利用率优化：从 6x16 到更大 Tile

#### 问题识别

当前 AVX-512 的 6x16 micro-kernel 只用了 8 个 zmm 寄存器（6 累加器 + 1 B load + 1 A broadcast），而 AVX-512 提供了 32 个。寄存器利用率仅 25%，意味着硬件资源被大量浪费。

#### 方法论：最大化算术强度（Arithmetic Intensity）

优化的核心方法论是：**在寄存器预算允许的范围内，增大 tile 尺寸以提高每次内存访问所执行的计算量。**

具体推导过程：

1. **确定约束**：32 个 zmm 寄存器是硬性上限
2. **建立模型**：对于 MR×NR 的 tile（NR 为 16 的倍数），所需寄存器 = MR×(NR/16) 累加器 + NR/16 个 B load + 1 个 A broadcast
3. **选择目标**：最大化每次 K 迭代的 FMA 数量（= MR × NR/16），同时考虑 FMA 流水线延迟隐藏
4. **验证可行性**：确保不 spill，且 L1 带宽能喂饱计算单元

#### FMA 流水线延迟隐藏

现代 Intel AVX-512 的 FMA 指令延迟 ~4 cycles，吞吐 2/cycle（port 0 + port 5）。要充分填满流水线，需要至少 **8 条相互独立的 FMA 指令** 同时 in-flight。当前 6x16 每次迭代只有 6 条 FMA，不足以完全隐藏延迟。

**体系结构深度解析：**

**延迟 ~4 cycles 的含义：** FMA 是单条指令完成 `a = a * b + c`，但在执行单元内部需经过多个流水线阶段（乘法→对齐→加法→规格化），从 operand 就绪到结果可被后续指令使用，需要等待 4 个时钟周期。如果下一条 FMA 依赖上一条的结果（如 `c0 = fma(a, b, c0)` 后紧跟 `c0 = fma(x, y, c0)`），第二条必须等 4 cycles 才能开始。

**吞吐 2/cycle 的含义：** Skylake-X/Ice Lake 的 AVX-512 FMA 有两个独立执行端口（port 0 和 port 5），每个端口每 cycle 可接收一条新 FMA。峰值吞吐为每 cycle 发射 2 条 FMA——前提是指令间无数据依赖。

**所需独立指令数推导：**

```
所需独立指令数 = 延迟 × 吞吐 = 4 cycles × 2 条/cycle = 8 条
```

直觉：想象一个旋转调度的"指令池"——

```
cycle 0: 发射 fma(c0), fma(c1)     ← c0, c1 结果要到 cycle 4 才就绪
cycle 1: 发射 fma(c2), fma(c3)     ← c2, c3 要到 cycle 5
cycle 2: 发射 fma(c4), fma(c5)     ← c4, c5 要到 cycle 6
cycle 3: 发射 fma(c6), fma(c7)     ← c6, c7 要到 cycle 7
cycle 4: 发射 fma(c0), fma(c1)     ← c0 在 cycle 4 就绪，刚好可以再用!
cycle 5: 发射 fma(c2), fma(c3)     ← 完美衔接，零气泡
...
```

8 条独立 FMA 链（8 个不同累加器）形成恰好填满流水线的旋转调度。每个累加器被使用后，经过 4 cycles（期间其他 7 条指令执行），轮到自己时结果刚好就绪。

**6x16 为什么不够（6 条独立 FMA）：**

```
cycle 0: fma(c0), fma(c1)    ← port 0, port 5 各发射一条
cycle 1: fma(c2), fma(c3)
cycle 2: fma(c4), fma(c5)
cycle 3: ???                  ← 没有更多独立FMA! c0要到cycle 4才就绪, 产生1cycle气泡
cycle 4: fma(c0), fma(c1)    ← c0终于就绪，重新开始
```

峰值利用率 = 6/8 = **75%**，每 4 cycle 有 1 cycle 空转。

**6x48 如何解决（18 条独立 FMA）：**

```
cycle 0: fma(c00), fma(c01)
cycle 1: fma(c02), fma(c10)
cycle 2: fma(c11), fma(c12)
cycle 3: fma(c20), fma(c21)    ← 流水线持续满载，无需等待任何结果
...
cycle 8: fma(c50), fma(c51)
cycle 9: fma(c52), ...         ← 新一轮K迭代, c00早在cycle 4就绪
```

18 >> 8，流水线永远不会因数据依赖而停顿。多出的余量还能容忍 B load 和 A broadcast 指令插入时占用的发射槽位。

**性能瓶颈本质：**

```
Performance = min(计算吞吐上限, 内存带宽上限, 流水线填充率 × 吞吐上限)
                                                ↑ 6x16的瓶颈在这里

6x16: 流水线填充率 = 6/8 = 75%   → 实际吞吐 = 75% × 峰值
6x48: 流水线填充率 = min(18/8, 1) = 100% → 实际吞吐 = 100% × 峰值
```

**核心结论：无 spill 是必要条件，填满流水线才是充分条件。**

#### `_mm512_set1_ps`（broadcast）是否会打断 FMA 流水线？

答案是**不会**。虽然 micro-kernel 中 broadcast 和 FMA 交替出现，但从微架构层面看它们是并行执行的。

**原因一：使用不同的执行端口，且编译器会融合为内存操作数**

实际编译中 `_mm512_set1_ps(A[i*lda + k])` 通常被融合为内存形式的 broadcast，不占用 FMA 端口：

```asm
vbroadcastss zmm0, dword [rax]     ; 走 load port，直接从内存broadcast
vfmadd231ps  zmm6, zmm0, zmm3     ; FMA 走 Port 0/5
```

**原因二：乱序执行引擎并行调度**

现代 CPU 是乱序超标量的，调度器会将无依赖关系的指令并行发射到不同端口：

```
cycle N:   vbroadcastss zmm0, [addr_A0]    → Load port
cycle N:   vfmadd231ps  zmm10, zmm1, zmm3  → Port 0   ← 与broadcast同cycle执行!
cycle N+1: vfmadd231ps  zmm11, zmm0, zmm3  → Port 0   ← zmm0就绪后立即使用
```

**原因三：依赖链分析——broadcast 不在关键路径上**

FMA 流水线的瓶颈是**累加器的 RAW 依赖**（c00 写后读）。broadcast 结果只是 FMA 的一个输入操作数，属于独立的短依赖链：

```
关键路径（累加器）:  c00 → fma → c00 → fma → c00 ...  (间隔4 cycles)
非关键路径（broadcast）:  load A[i] → broadcast → 送入fma operand (~1-3 cycles)
```

broadcast 延迟 ~1-3 cycles（L1 命中），远小于累加器链的 4 cycle 间隔，被完全掩盖在 FMA 延迟的"阴影"中。

**原因四：端口压力验证（以 6x48 为例）**

```
每次K迭代: 6次broadcast + 3次B load + 18次FMA = 27条微操作
FMA 需要:  18 ÷ 2端口 = 9 cycles (计算 bound)
Load/broadcast: 9 ÷ 2 load ports = 4.5 cycles (内存 bound)

计算时间(9) > 内存时间(4.5) → broadcast 完全被 FMA 延迟掩盖，不是瓶颈
```

**总结：**

| 因素 | 是否打断 | 原因 |
|------|---------|------|
| 端口冲突 | 否 | broadcast 走 load port 或被融合为内存操作数 |
| 数据依赖 | 否 | broadcast 结果不在累加器关键路径上 |
| 延迟掩盖 | 否 | ~1-3 cycles 被 FMA 4 cycle 间隔完全掩盖 |
| 乱序调度 | 否 | CPU 将 broadcast 和无关 FMA 并行发射 |

#### Tile 尺寸设计空间

| Tile | 累加器 | B load | A broadcast | 总 zmm | FMA/iter | A 复用率 |
|------|--------|--------|-------------|--------|----------|----------|
| 6x16 (当前) | 6 | 1 | 1 | 8 | 6 | 6x |
| 6x32 | 12 | 2 | 1 | 15 | 12 | 6x |
| **6x48** | 18 | 3 | 1 | **22** | **18** | 6x |
| 14x16 | 14 | 1 | 1 | 16 | 14 | 14x |
| 12x32 | 24 | 2 | 1 | 27 | 24 | 12x |
| 14x32 | 28 | 2 | 1 | 31 | 28 | 14x |
| 30x16 | 30 | 1 | 1 | 32 | 30 | 30x |

#### 设计权衡

- **增大列数（NR）**：每次 A broadcast 做更多 FMA，但需要更多 B load，占用 L1 读带宽
- **增大行数（MR）**：每次 B load 被更多行复用，提高 A 复用率，但需要更多 broadcast 指令
- **行业实践**：BLIS 在 Skylake-X 上用 30x16（极致 A 复用），OneDNN 用 6x64 的变体（极致列宽）

#### 推荐方案：6x48 micro-kernel

选择 6x48 的理由：
1. 保持 MR=6 不变，outer loop 结构无需调整
2. 18 条独立 FMA 充分隐藏流水线延迟（18 > 8）
3. 22 个 zmm 使用率 69%，留有余量避免编译器临时变量导致 spill
4. 每次 B load 仍被 6 行复用，带宽效率不变

```cpp
#include <immintrin.h>

// 6x48 register blocking micro-kernel for AVX-512
// C[6x48] = A[6xK] * B[Kx48]
// 寄存器预算: 6x3=18(累加器) + 3(B load) + 1(A broadcast) = 22个zmm
// 每次K迭代: 18次FMA, 充分隐藏FMA延迟(需>=8条独立FMA)
void micro_kernel_6x48(int K,
                       const float* A, int lda,
                       const float* B, int ldb,
                       float* C, int ldc) {
    // 18个累加器: 6行 x 3段(每段16 floats = 1个zmm)
    __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps(), c02 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps(), c12 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps(), c22 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps(), c32 = _mm512_setzero_ps();
    __m512 c40 = _mm512_setzero_ps(), c41 = _mm512_setzero_ps(), c42 = _mm512_setzero_ps();
    __m512 c50 = _mm512_setzero_ps(), c51 = _mm512_setzero_ps(), c52 = _mm512_setzero_ps();

    for (int k = 0; k < K; k++) {
        // load B[k, 0:48] — 3段, 每段16 floats
        __m512 b0 = _mm512_loadu_ps(&B[k * ldb]);
        __m512 b1 = _mm512_loadu_ps(&B[k * ldb + 16]);
        __m512 b2 = _mm512_loadu_ps(&B[k * ldb + 32]);

        __m512 a;
        // 每个A元素broadcast后与3段B做FMA — 每组B load被复用6次
        a = _mm512_set1_ps(A[0*lda + k]);
        c00 = _mm512_fmadd_ps(a, b0, c00);
        c01 = _mm512_fmadd_ps(a, b1, c01);
        c02 = _mm512_fmadd_ps(a, b2, c02);

        a = _mm512_set1_ps(A[1*lda + k]);
        c10 = _mm512_fmadd_ps(a, b0, c10);
        c11 = _mm512_fmadd_ps(a, b1, c11);
        c12 = _mm512_fmadd_ps(a, b2, c12);

        a = _mm512_set1_ps(A[2*lda + k]);
        c20 = _mm512_fmadd_ps(a, b0, c20);
        c21 = _mm512_fmadd_ps(a, b1, c21);
        c22 = _mm512_fmadd_ps(a, b2, c22);

        a = _mm512_set1_ps(A[3*lda + k]);
        c30 = _mm512_fmadd_ps(a, b0, c30);
        c31 = _mm512_fmadd_ps(a, b1, c31);
        c32 = _mm512_fmadd_ps(a, b2, c32);

        a = _mm512_set1_ps(A[4*lda + k]);
        c40 = _mm512_fmadd_ps(a, b0, c40);
        c41 = _mm512_fmadd_ps(a, b1, c41);
        c42 = _mm512_fmadd_ps(a, b2, c42);

        a = _mm512_set1_ps(A[5*lda + k]);
        c50 = _mm512_fmadd_ps(a, b0, c50);
        c51 = _mm512_fmadd_ps(a, b1, c51);
        c52 = _mm512_fmadd_ps(a, b2, c52);
    }

    // store results: C += accumulated
    _mm512_storeu_ps(&C[0*ldc],    _mm512_add_ps(_mm512_loadu_ps(&C[0*ldc]),    c00));
    _mm512_storeu_ps(&C[0*ldc+16], _mm512_add_ps(_mm512_loadu_ps(&C[0*ldc+16]), c01));
    _mm512_storeu_ps(&C[0*ldc+32], _mm512_add_ps(_mm512_loadu_ps(&C[0*ldc+32]), c02));
    _mm512_storeu_ps(&C[1*ldc],    _mm512_add_ps(_mm512_loadu_ps(&C[1*ldc]),    c10));
    _mm512_storeu_ps(&C[1*ldc+16], _mm512_add_ps(_mm512_loadu_ps(&C[1*ldc+16]), c11));
    _mm512_storeu_ps(&C[1*ldc+32], _mm512_add_ps(_mm512_loadu_ps(&C[1*ldc+32]), c12));
    _mm512_storeu_ps(&C[2*ldc],    _mm512_add_ps(_mm512_loadu_ps(&C[2*ldc]),    c20));
    _mm512_storeu_ps(&C[2*ldc+16], _mm512_add_ps(_mm512_loadu_ps(&C[2*ldc+16]), c21));
    _mm512_storeu_ps(&C[2*ldc+32], _mm512_add_ps(_mm512_loadu_ps(&C[2*ldc+32]), c22));
    _mm512_storeu_ps(&C[3*ldc],    _mm512_add_ps(_mm512_loadu_ps(&C[3*ldc]),    c30));
    _mm512_storeu_ps(&C[3*ldc+16], _mm512_add_ps(_mm512_loadu_ps(&C[3*ldc+16]), c31));
    _mm512_storeu_ps(&C[3*ldc+32], _mm512_add_ps(_mm512_loadu_ps(&C[3*ldc+32]), c32));
    _mm512_storeu_ps(&C[4*ldc],    _mm512_add_ps(_mm512_loadu_ps(&C[4*ldc]),    c40));
    _mm512_storeu_ps(&C[4*ldc+16], _mm512_add_ps(_mm512_loadu_ps(&C[4*ldc+16]), c41));
    _mm512_storeu_ps(&C[4*ldc+32], _mm512_add_ps(_mm512_loadu_ps(&C[4*ldc+32]), c42));
    _mm512_storeu_ps(&C[5*ldc],    _mm512_add_ps(_mm512_loadu_ps(&C[5*ldc]),    c50));
    _mm512_storeu_ps(&C[5*ldc+16], _mm512_add_ps(_mm512_loadu_ps(&C[5*ldc+16]), c51));
    _mm512_storeu_ps(&C[5*ldc+32], _mm512_add_ps(_mm512_loadu_ps(&C[5*ldc+32]), c52));
}
```

**6x16 vs 6x48 对比：**

| | 6x16 (原始) | 6x48 (优化) |
|---|---|---|
| zmm 使用量 | 8/32 (25%) | 22/32 (69%) |
| FMA/iter | 6 | 18 |
| B load/iter | 1 | 3 |
| A broadcast/iter | 6 | 6 |
| 流水线利用 | 不充分（6 < 8） | 充分（18 > 8） |
| 理论加速比 | 1x | **3x** |
| C tile 大小 | 6×16 = 96 floats | 6×48 = 288 floats |

#### 方法论总结：Micro-Kernel Tile 尺寸选择四步法

```
1. 盘点硬件资源
   ├── 可用向量寄存器数（AVX-512: 32 zmm, AVX2: 16 ymm）
   ├── FMA 延迟和吞吐（确定最小独立指令数）
   └── L1 读端口带宽（确定每cycle能发射几个load）

2. 建立寄存器预算方程
   总寄存器 ≥ MR × (NR/向量宽度) + (NR/向量宽度) + 1 + 编译器余量

3. 在约束内最大化 FMA 密度
   ├── 目标: FMA/iter = MR × (NR/向量宽度) ≥ FMA延迟 × FMA吞吐
   ├── 行数MR ↑ → A复用率高，但broadcast指令多
   └── 列数NR ↑ → 每个broadcast做更多FMA，但B load带宽压力大

4. 验证与调优
   ├── 编译汇编确认无 spill（搜索 [rsp] 访问）
   ├── 用 perf stat 测 IPC 和 cache miss rate
   └── 实测不同 tile 组合，选吞吐最优者
```

---

## 4. Macro-Kernel（宏内核）

如果说 `micro-kernel` 负责把一个小块算快，那么 `macro-kernel` 负责把这个小块高效地重复很多次。

> **micro-kernel 解决怎么快算一个小块，macro-kernel 解决怎么高效地反复调用这个小块去算完整个大块。**

在 GEMM 里，`macro-kernel` 通常是包在 `micro-kernel` 外面的一层或几层块级调度逻辑。它负责：

- 组织 `A panel / B panel / C block` 的分块关系
- 安排 packing，把访问模式变成连续内存
- 选择 `jr/ir` 等小块遍历顺序
- 让同一批 panel 被反复送给 `micro-kernel` 复用

可以先看一个极简框架：

```cpp
for (jc ...)          // outer blocking, often L3-related
    for (pc ...)        // panel blocking, often L2-related
        pack_B(...)
        for (ic ...)      // panel blocking, often L2/L1-related
            pack_A(...)
            for (jr ...)    // macro-kernel repeatedly walks small column tiles
                for (ir ...)  // macro-kernel repeatedly walks small row tiles
                    micro_kernel(MR, NR, KC, ...);
```

这里最里面的 `micro_kernel(...)` 是寄存器级内核；围绕它反复遍历 `jr/ir` 小块并复用 packed panel 的那部分，就是常说的 `macro-kernel`。

把它写成函数后，通常长这样：

```cpp
constexpr int MR = 6;
constexpr int NR = 16;

// macro-kernel: 负责在一个 C block 上反复调用 micro-kernel
// packed_A: [mc x kc]
// packed_B: [kc x nc]
// C block : [mc x nc]
void macro_kernel_6x16(int mc, int nc, int kc,
                       const float* packed_A,
                       const float* packed_B,
                       float* C, int ldc) {
    for (int jr = 0; jr < nc; jr += NR) {
        for (int ir = 0; ir < mc; ir += MR) {
            micro_kernel_6x16(kc,
                              &packed_A[ir * kc], kc,
                              &packed_B[jr * kc], kc,
                              &C[ir * ldc + jr], ldc);
        }
    }
}
```

这个函数本身不直接做寄存器级 FMA；它的职责是遍历一个更大的 `C block`，把其中每个 `MR x NR` 小块交给 `micro_kernel_6x16(...)` 去算。

如果把职责拆开，可以更清楚地看到分工：

- `micro-kernel` 只关心一个 `MR x NR` 小块怎么算得快，例如 `6x16` 的外积累加。
- `macro-kernel` 关心的是 `packed_A` 和 `packed_B` 怎么准备、`jr/ir` 的遍历顺序怎么安排、以及同一批 panel 如何喂给很多次 `micro-kernel`。
- 更外层的 cache blocking 决定数据主要驻留在哪一级 cache，并为 `macro-kernel` 提供稳定的数据来源。

因此，`macro-kernel` 可以理解为连接 cache blocking 和 register blocking 的桥梁：

- 往里看，它服务 `micro-kernel`
- 往外看，它承接 L1/L2/L3 的分块和 packing

## 5. Cache Blocking（缓存分块 / Loop Tiling）

将大数据集的操作分块，使每个块的工作集能放进某一级 cache（L1/L2/L3），避免反复从更高层级甚至内存加载数据。

**核心问题：** 如果数据量 >> cache 大小，遍历一遍后数据就被逐出了，下次访问是 cache miss。

### Case：矩阵乘法的多级 cache blocking

```cpp
// 假设: L1=32KB, L2=256KB, L3=8MB
// float大小=4B
// L2能放 256KB/4B = 64K floats ≈ 256x256 的矩阵块

constexpr int MC = 256;  // A的行分块 (fits in L2 with B panel)
constexpr int KC = 256;  // K维分块 (A panel + B panel fit in L2)
constexpr int NC = 1024; // B的列分块 (fits in L3)

// 这段外层 cache blocking 负责准备 panel 并调用 macro-kernel。
// macro-kernel 再继续把一个 C block 切成很多个 MR x NR 小块，
// 每个小块最终由 micro-kernel 完成寄存器级计算。
void gemm_cache_blocked(int M, int N, int K,
                        const float* A, const float* B, float* C) {
    // L3 blocking: B的列方向
    for (int jc = 0; jc < N; jc += NC) {
        int nc = std::min(NC, N - jc);

        // L2 blocking: K方向 — 使 A_panel[MC x KC] 和 B_panel[KC x NC] 驻留L2
        for (int pc = 0; pc < K; pc += KC) {
            int kc = std::min(KC, K - pc);

            // Pack B[pc:pc+kc, jc:jc+nc] into contiguous buffer (改善空间局部性)
            // packed_B: KC x NC, 连续存储

            // L1 blocking: A的行方向
            for (int ic = 0; ic < M; ic += MC) {
                int mc = std::min(MC, M - ic);

                // Pack A[ic:ic+mc, pc:pc+kc] into contiguous buffer
                // packed_A: MC x KC, 连续存储

                // 此时 packed_A (MC*KC = 256*256 = 256KB) 在 L2
                //      packed_B 的当前列 (KC*NR) 在 L1
                // 这里把当前 panel 交给 macro-kernel 处理。
                macro_kernel_6x16(mc, nc, kc,
                                  packed_A,
                                  packed_B,
                                  &C[ic * N + jc], N);
            }
        }
    }
}
```

### 矩阵分块图示

```
C[M x N] = A[M x K] * B[K x N]

         K                         N
    ┌──────────┐            ┌──────────────────────────────────┐
    │          │            │            NC=1024               │
    │          │            │◄────────────────────────────────►│
    │  A panel │ KC=256     ├──────────────────────────────────┤
MC  │  [MC×KC] │ ◄─────►    │                                  │ KC=256
=256│          │            │         B panel [KC×NC]          │
    │          │            │                                  │
    ├──────────┤            ├──────────────────────────────────┤
    │          │            │                                  │
    │          │            │                                  │
  M │          │          K │                                  │
    │          │            │                                  │
    │          │            │                                  │
    │          │            │                                  │
    └──────────┘            └──────────────────────────────────┘

                            ┌──────────────────────────────────┐
                            │            NC=1024               │
                            │                                  │
                  C panel   │         C [MC×NC]                │ MC=256
                            │                                  │
                            └──────────────────────────────────┘
```

**Micro-kernel 在分块中的位置（6x16 tile）：**

```
        A panel [MC×KC]                B panel [KC×NC]
        ┌────────────┐       ┌───────────────────────────────────┐
        │            │       │  NR=16  NR=16  NR=16       NR=16  │
        │            │       │ ┌─────┐┌─────┐┌─────┐ ... ┌─────┐ │
     MR │ A[6×KC]    │   KC  │ │     ││     ││     │     │     │ │
     =6 │ ░░░░░░░░░░ │       │ │B col││     ││     │     │     │ │
        │            │       │ │[KC× ││     ││     │     │     │ │
        ├────────────┤       │ │ 16] ││     ││     │     │     │ │
        │            │       │ │     ││     ││     │     │     │ │
        │            │       │ └─────┘└─────┘└─────┘ ... └─────┘ │
     MC │            │       └───────────────────────────────────┘
        │            │
        │            │          C panel [MC×NC]
        │            │       ┌───────────────────────────────────┐
        │            │       │  NR=16                            │
        │            │       │ ┌─────┐                           │
        │            │    MR │ │█████│← micro-kernel 计算的      │
        │            │    =6 │ │█████│  C[6×16] 驻留寄存器        │
        └────────────┘       │ ├─────┤                           │
                             │ │     │                           │
                             │ │     │                           │
                             │ └─────┘                           │
                             └───────────────────────────────────┘

计算过程 (micro_kernel_6x16):

  A[6×1]      B[1×16]            C[6×16] (寄存器中累加)
  ┌───┐     ┌─────────────────┐    ┌─────────────────┐
  │ a0│     │b0 b1 b2 ... b15 │    │ c00 c01 ... c0F │
  │ a1│     └─────────────────┘    │ c10 c11 ... c1F │
  │ a2│            ×         ──►   │ c20 c21 ... c2F │  += outer product
  │ a3│                            │ c30 c31 ... c3F │
  │ a4│     (每次K迭代取1列A,       │ c40 c41 ... c4F │
  │ a5│      1行B, 做外积累加)      │ c50 c51 ... c5F │
  └───┘                            └─────────────────┘

  对K维循环: C[6×16] += A[6×1] * B[1×16]  (rank-1 update)
```

### 数据流层次

```
┌─────────────────────────────────────────┐
│  L3 blocking (NC=1024)                  │  B panel fits in L3
│  ┌───────────────────────────────────┐  │
│  │  L2 blocking (KC=256)             │  │  A panel + B slice in L2
│  │  ┌─────────────────────────────┐  │  │
│  │  │  L1 blocking (MC=256)       │  │  │  micro-tile in L1
│  │  │  ┌───────────────────────┐  │  │  │
│  │  │  │  Register blocking    │  │  │  │  6x16 in registers
│  │  │  │  (MR=6, NR=16)        │  │  │  │
│  │  │  └───────────────────────┘  │  │  │
│  │  └─────────────────────────────┘  │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 如何理解这些层级

可以把它们理解为一条从外到内的数据供给链：Memory -> L3 -> L2 -> L1 -> Registers。

如果只想抓住工程上的主线，可以记成：

- **寄存器 + L1** 属于最内层的 kernel 供给链。Register blocking 决定 micro-kernel 一次在寄存器里累计多大的 `C` tile；L1 blocking 负责把马上要消费的 `A/B` 小片段留在 L1，持续给寄存器喂数。
- **L2 blocking** 是主分块层。它决定 `A panel` 和 `B panel` 的主体工作集大小，使同一批数据能被很多次 micro-kernel 调用反复复用，而不必频繁回退到 L3 或内存。
- **L3 blocking** 是更外层的容量分块。它主要减少对内存的访问压力，为 L2 blocking 提供更大的上层缓冲。

因此，`L1 blocking` 和 `register blocking` 不是同一个层级，但它们通常协同工作，属于最内层优化链条；而 `L2 blocking` 往往才是调 `MC/KC` 这类主分块参数时最核心的一层。

---

## 层次关系总结

| 概念 | 目标层级 | 核心思想 |
|------|----------|----------|
| Register Spill | 寄存器 ↔ 栈 | **避免**超出寄存器容量，减少不必要的 load/store |
| Register Blocking | 寄存器 | **最大化**寄存器中数据的复用次数 |
| Micro-Kernel | 最内层计算内核 | 把寄存器分块写成具体的 `MR x NR` 小核 |
| Macro-Kernel | 小块调度层 | 高效组织并重复调用 micro-kernel |
| Cache Blocking | Cache各级 | 控制工作集大小，使数据**驻留**在目标cache层 |

它们是互补的：cache blocking 决定大块数据如何驻留在 L3/L2/L1，macro-kernel 负责在这些 panel 上组织小块遍历，micro-kernel 把寄存器分块真正执行出来，而控制好 blocking 的尺寸可以避免 register spill。高性能 GEMM 库（如 OpenBLAS、MKL）通常同时运用这几层。

### 软件层次 vs 硬件层次

从软件-硬件映射的角度看，这几层存在很强的对应关系，但通常不是严格的一一映射，而是“主要作用在哪一级硬件层次”。

| 软件层次 | 主要目标 | 主要硬件落点 |
|----------|----------|--------------|
| Register Blocking | 让小块结果和热点临时值常驻寄存器 | Register |
| Micro-Kernel | 用 SIMD/FMA 高效计算一个 `MR x NR` 小块 | Register + L1 |
| Macro-Kernel | 反复调度 micro-kernel，复用 packed panel | L1 + L2 |
| Cache Blocking | 控制更大工作集驻留在哪级 cache | L2 + L3 + Memory |

如果只抓最核心的映射关系，可以记成：

- `Register blocking` 主要对应 `Register`
- `Micro-kernel` 主要对应 `Register + L1`
- `Macro-kernel` 主要对应 `L1 + L2`
- `Cache blocking` 主要对应 `L2 + L3 + Memory`

之所以不是严格的一一对应，是因为软件层次回答的是“怎样组织计算与数据复用”，硬件层次回答的是“数据实际驻留在哪里”。两者高度相关，但一个软件概念通常会跨越不止一层硬件。

#### register数量 => MR, NR的选择

这部分决定 micro-kernel 的形状（`MR x NR`）。核心是先做寄存器预算，再看流水线是否能被填满。

设：

- 向量宽度（float 个数）为 `VW`：AVX2 是 8，AVX-512 是 16
- 可用于内核的向量寄存器数为 `R_eff`（不是总数，需预留少量临时）
- `nvec = NR / VW`（一行 C 需要多少个向量寄存器）

寄存器预算近似：

```
R_need = MR * nvec        // C 累加器
         + nvec             // B 向量寄存器
         + 1                // A broadcast 寄存器
         + R_tmp            // 地址/调度临时，常取 0~2

约束: R_need <= R_eff
```

等价写成：

```
MR <= floor((R_eff - nvec - 1 - R_tmp) / nvec)
```

再加一条吞吐约束（隐藏 FMA 延迟）：

```
独立累加链数量 = MR * nvec
要求: MR * nvec >= FMA_latency * FMA_throughput
```

对常见 x86 核心可粗略记成右侧约等于 8~10。

示例（AVX2，16 个 ymm）：

- 若取 `NR=16`，则 `nvec=2`
- 取 `R_eff=15, R_tmp=0`，有 `MR <= floor((15-2-1)/2)=6`
- 得到 `6x16`，累加链 `12` 条，通常足够覆盖 FMA 延迟

若取 `NR=24`，则 `nvec=3`：

- `MR <= floor((15-3-1)/3)=3`
- 若硬上 `MR=4`（即 `4x24`），寄存器占用会非常紧（常接近满寄存器）

结论：

1. 先用寄存器不等式圈定候选 `MR/NR`
2. 再用“独立累加链数量”过滤掉吞吐不足的候选
3. 最后用汇编确认无 spill（看是否出现明显的栈回写/回读）

#### L2 Cache大小 => MC, NC, KC的选择

这部分决定 macro/cache blocking 的面板大小。可先用容量模型给出一组可行初值，再实测微调。

设：

- `S_L2`：每核 L2 容量（bytes）
- `alpha`：L2 可用于 GEMM 工作集的比例（经验值 0.6~0.8）
- 元素字节数 `s`：FP32 时 `s=4`

对于一轮 `(ic, jc, pc)` 计算，保守工作集可写为：

```
W = s * (MC*KC + KC*NC + beta*MC*NC)
```

- `MC*KC`：A panel
- `KC*NC`：B panel
- `beta*MC*NC`：C block 在缓存中的占比（`beta` 常取 0~1，保守可取 1）

容量约束：

```
W <= alpha * S_L2
```

即：

```
s * (MC*KC + KC*NC + beta*MC*NC) <= alpha * S_L2
```

若先忽略 C（`beta=0`），可直接解出 `KC`：

```
KC <= floor((alpha * S_L2 / s) / (MC + NC))
```

这是最常用的一阶估算式。

数值例子 1（每核 L2 = 1 MiB，FP32，`alpha=0.8`）：

- `S_L2 = 1,048,576`
- 取 `MC=240, NC=240`

则：

```
KC <= floor((0.8 * 1,048,576 / 4) / (240 + 240))
    <= floor(209,715 / 480)
    <= 436
```

说明：若目标是让 A/B 面板主要驻留在 L2，`KC=1008` 明显偏大。

数值例子 2（每核 L2 = 256 KiB，FP32，`alpha=0.8`，同样 `MC=NC=240`）：

```
KC <= floor((0.8 * 262,144 / 4) / 480)
    <= floor(52,428 / 480)
    <= 109
```

可见小 L2 机器上需要更小的 `KC`。

工程化选参步骤：

1. 先由寄存器约束确定 `MR/NR`
2. 给定目标 `MC/NC`，用上式先算 `KC_max`
3. 从 `KC` 的低到高做 sweep（如 96/128/160/192/256/...）
4. 再在邻域微调 `MC/NC`（例如步长按 `MR/NR` 的倍数）
5. 用 GFLOPS + cache miss 指标共同选型，而不是只看单次时间

一个常见经验是：

- `KC` 往往先由 L2 容量卡住
- `NC` 更多受 LLC/L3 与并行切分影响
- `MC` 在满足 L2 约束后，通常选成 `MR` 的整数倍并尽量大

推荐初值表（可直接作为第一轮 sweep 起点）：

统一假设：

- FP32（`s=4`）
- `alpha=0.8`
- 先按 `beta=0`（只约束 A/B panel）估算 `KC_max`
- `MC` 取 `MR` 的整数倍（例如 6 的倍数）
- `NC` 取 `NR` 的整数倍（例如 16 或 24 的倍数）

##### 方案A：按 256KiB L2（每核）

| 推荐档位 | MC | NC | 估算 KC_max | 建议 KC 初值 | 说明 |
|---|---:|---:|---:|---:|---|
| 保守起步 | 120 | 120 | 218 | 160 | 容量余量较大，适合先验证稳定性 |
| 平衡默认 | 120 | 240 | 145 | 128 | 对 AVX2 6x16 常见且稳妥 |
| 偏吞吐 | 180 | 180 | 145 | 128 | A/B 面板更大，吞吐潜力更高 |
| 大NC尝试 | 96 | 288 | 136 | 128 | 增大列块，利于 B 复用 |
| 你当前风格对照 | 240 | 240 | 109 | 96 | 若坚持 240x240，KC 需明显下调 |

推荐 sweep 顺序：先固定 `MC=120, NC=240`，测试 `KC={96,128,160}`，再微调 `MC/NC`。

##### 方案B：按 1MiB L2（每核）

| 推荐档位 | MC | NC | 估算 KC_max | 建议 KC 初值 | 说明 |
|---|---:|---:|---:|---:|---|
| 保守起步 | 192 | 192 | 273 | 192 | 适合作为稳态起点 |
| 平衡默认 | 240 | 240 | 436 | 256 | 与你现有分块形态接近 |
| 偏吞吐 | 288 | 288 | 364 | 320 | 更大的 panel，适合强核大L2 |
| 大NC尝试 | 192 | 384 | 291 | 256 | 提高列方向复用 |
| 激进尝试 | 240 | 384 | 269 | 256 | 更高 NC，关注 LLC miss |

推荐 sweep 顺序：先固定 `MC=240, NC=240`，测试 `KC={192,256,320,384}`，再调 `NC`（240 -> 384）。

补充说明：

1. 上表是“第一轮初值”，不是最终最优值。
2. 若把 C block 也严格计入 L2（`beta>0`），则 `KC_max` 还要下调。
3. 若观察到 L2 miss 上升或带宽打满，优先减小 `KC`，其次减小 `NC`。

## TODO（下一步优化清单）

- [x] 提前分配并复用 `packed_A/packed_B` 缓冲区（已验证有收益）
- [ ] 将 `pack_panel` 从双层标量拷贝改为按行 `memcpy`（目标：降低 packing 开销）
- [ ] 增加参数化 benchmark（命令行或配置）并自动输出 GFLOPS
- [ ] 做三维 sweep：`MC x NC x KC`（先固定 kernel，再扫参数）
- [ ] 做并行化版本（优先并行外层 `jc`，避免写同一 `C` 子块）
- [ ] 增加 tail 处理路径（支持 `M/N` 非 `MR/NR` 整数倍）
- [ ] 用 `perf stat` 记录 IPC / cache-miss / 带宽指标，和 GFLOPS 一起评估

建议执行顺序：

1. `pack_panel` 改造 + benchmark 参数化
2. `MC/NC/KC` sweep 找单线程最优点
3. 外层并行化并重新 sweep
4. 最后补齐通用尺寸 tail 处理
