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

## 3. Cache Blocking（缓存分块 / Loop Tiling）

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
                // 调用 register-blocked micro-kernel
                for (int jr = 0; jr < nc; jr += 16) {
                    for (int ir = 0; ir < mc; ir += 6) {
                        micro_kernel_6x16(kc,
                                          &packed_A[ir * kc], kc,
                                          &packed_B[jr * kc], kc,
                                          &C[(ic+ir)*N + (jc+jr)], N);
                    }
                }
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

---

## 三者关系总结

| 概念 | 目标层级 | 核心思想 |
|------|----------|----------|
| Register Spill | 寄存器 ↔ 栈 | **避免**超出寄存器容量，减少不必要的 load/store |
| Register Blocking | 寄存器 | **最大化**寄存器中数据的复用次数 |
| Cache Blocking | Cache各级 | 控制工作集大小，使数据**驻留**在目标cache层 |

它们是互补的：cache blocking 把数据送到 L1，register blocking 把 L1 数据送到寄存器并最大化复用，而控制好 blocking 的尺寸可以避免 register spill。高性能 GEMM 库（如 OpenBLAS、MKL）同时运用三者。
