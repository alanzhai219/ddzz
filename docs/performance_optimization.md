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
