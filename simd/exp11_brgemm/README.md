# BRGEMM Intrinsic Implementation (AVX2 + AVX512)

This folder implements BRGEMM using C++ intrinsics (not JIT).

Reference paper:
- [High-Performance Deep Learning via a Single Building Block (arXiv:1906.06440)](https://arxiv.org/pdf/1906.06440)

## 1) What BRGEMM means in the paper

The paper proposes batch-reduce GEMM as the common compute primitive for multiple deep learning operators.

Mathematically:

$$
C = \sum_{b=0}^{B-1} A_b B_b
$$

where:
- $A_b \in \mathbb{R}^{M \times K}$
- $B_b \in \mathbb{R}^{K \times N}$
- $C \in \mathbb{R}^{M \times N}$

Interpretation:
- A standard GEMM performs one matrix multiply.
- BRGEMM performs many small GEMMs and reduces them into one output matrix.
- This is especially suitable for blocked/reordered DL workloads where each block pair contributes to the same output tile.

Why this is useful (paper insight):
- Instead of implementing many specialized kernels, many operators can be rewritten as loop nests around one optimized BRGEMM kernel.
- Optimization effort is concentrated in this single inner kernel.

## 2) This implementation

Files:
- `brgemm_kernel.hpp`: API and ISA dispatch declarations.
- `brgemm_kernel.cpp`: scalar reference, AVX2 intrinsic kernel, AVX512 intrinsic kernel, runtime dispatch.
- `test_brgemm.cpp`: correctness + latency test.

Data layout assumptions:
- A batch: pointer array of row-major matrices `A_batch[b]`, shape `[M x K]`.
- B batch: pointer array of row-major matrices `B_batch[b]`, shape `[K x N]`.
- Output `C`, row-major, shape `[M x ldc]` with active columns `[0, N)`.

Core loop structure:

```text
zero C
for b in [0, batch_size):
	A = A_batch[b]
	B = B_batch[b]
	for m in [0, M):
		for k in [0, K):
			a = A[m, k]
			for n in [0, N):
				C[m, n] += a * B[k, n]
```

SIMD mapping:
- AVX2 processes 8 FP32 values per vector (`__m256`).
- AVX512 processes 16 FP32 values per vector (`__m512`).
- The `n` loop is vectorized; tail columns are handled by scalar remainder.

This maps naturally to BRGEMM because each scalar `a = A[m,k]` is broadcast and fused-multiply-add with a contiguous vector slice of `B[k, :]` into `C[m, :]`.

## 3) Notes on correctness and portability

- `brgemm_f32_ref(...)` is the baseline reference.
- `brgemm_f32_avx2(...)` compiles to AVX2/FMA path only when built with AVX2 support; otherwise it falls back to scalar reference.
- `brgemm_f32_avx512(...)` similarly requires AVX512F at compile time; otherwise falls back.
- `brgemm_f32_auto(...)` does runtime ISA detection and dispatch preference.

## 4) Build and run

From this folder:

```bash
g++ -O3 -std=c++17 -mavx2 -mfma test_brgemm.cpp brgemm_kernel.cpp -o test_brgemm_avx2
./test_brgemm_avx2
```

To enable AVX512 code generation (if machine supports it):

```bash
g++ -O3 -std=c++17 -mavx2 -mfma -mavx512f test_brgemm.cpp brgemm_kernel.cpp -o test_brgemm_avx512
./test_brgemm_avx512
```

## 5) Test behavior

`test_brgemm.cpp` does:
- random input generation
- reference computation
- AVX2 correctness check
- AVX512 correctness check (if supported)
- auto-dispatch correctness check
- simple average latency comparison

Output format:
- `PASS/FAIL` for each backend
- average time in ms and relative speedup vs reference

## 6) Relation to oneDNN-style BRGEMM blocking

This demo focuses on the core compute primitive and keeps packing/blocking simple.

Production BRGEMM implementations typically add:
- packed `B` (and sometimes `A`) for cache locality
- register and cache blocking on `(M, N, K)` tiles
- thread-level parallelism
- post-ops fusion (bias/activation/etc.)

Even without these layers, this intrinsic kernel demonstrates the central paper idea: many high-level operators can share one optimized BRGEMM inner compute block.

## 7) 原理解释：为什么 BRGEMM 更快

这一节只讲原理，不做逐行对应。

### 7.1 算法层面的优势

BRGEMM 的核心是：

$$
C = \sum_{b=0}^{B-1} A_b B_b
$$

相比“每个 b 做一次 GEMM 再单独归约”，BRGEMM 更快的常见原因是：

- 归约融合：直接累加到同一个 C，减少中间结果写回与再次读取。
- 调用开销更低：把 batch 维放进一个内核循环，减少大量小 kernel 的固定开销。
- 局部性更好：同一个 C tile 在短时间内被反复更新，更容易留在 cache/寄存器中。

### 7.2 指令层面的优势（SIMD + FMA）

- AVX2 一次处理 8 个 FP32，AVX512 一次处理 16 个 FP32。
- FMA 把乘法和加法融合为一条指令，减少指令数和寄存器压力。
- 通过广播 A[m, k] 与向量化遍历 B[k, :]，单次加载的标量 a 可复用于一整段列。

### 7.3 访存层面的优势

- B 与 C 在列方向是连续访问，cache 命中和硬件预取更稳定。
- 主体循环走纯向量路径，尾部单独处理，降低分支干扰。
- 减少了“中间结果矩阵”读写，整体带宽压力更小。

### 7.4 为什么有时 AVX512 不一定更快

在部分 CPU 上，AVX512 可能触发更低频率；当问题规模较小或尾部比例较高时，AVX2 可能更占优。这不影响 BRGEMM 原理本身，而是具体硬件与规模的平衡结果。

这正是论文强调的思想：把不同算子尽量改写为围绕同一个高性能 BRGEMM 内核的外层循环，再根据硬件选择最佳实现路径。
