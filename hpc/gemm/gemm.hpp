#include <immintrin.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

// ============================================================
// Configuration
// ============================================================
//
// GEMM:
//
//     C[M,N] = A[M,K] * B[K,N]
//
// We use a classic hierarchical blocking structure:
//
//     NC : outer N blocking
//     KC : K blocking
//     MC : M blocking
//
// Then the macro-kernel uses:
//
//     MR = 6
//     NR = 16
//
// as the register-level micro-kernel size.
//
//
//             ┌───────────────────────────────┐
//             │          GEMM M × N           │
//             └───────────────┬───────────────┘
//                             │
//                       Cache Blocking
//                             │
//                 ┌───────────┼───────────┐
//                 │           │           │
//                MC          KC          NC
//                 │           │           │
//                 ▼           ▼           ▼
//              A block      K block     B block
//                 │           │           │
//                 └───────────┼───────────┘
//                             │
//                       Macro-kernel
//                             │
//                       MR × NR = 6×16
//                             │
//                       Micro-kernel
//                             │
//                       Outer Product
//                             │
//                           FMA
//
// ============================================================

constexpr int MC = 96;
constexpr int NC = 128;
constexpr int KC = 256;

constexpr int MR = 6;
constexpr int NR = 16;

// ============================================================
// Utility
// ============================================================

static inline int min_int(int a, int b) {
    return std::min(a, b);
}


// ============================================================
// Packing A
// ============================================================
//
// Original A:
//
//     A[mc][kc]
//
// row-major:
//
//     A[i * lda + k]
//
// The micro-kernel wants:
//
//     A[:, k]
//
// to be contiguous.
//
// Therefore we pack A as:
//
//     Ap[k][i]
//
// Memory:
//
//     k=0: A[0,0] A[1,0] A[2,0] ... A[mc-1,0]
//     k=1: A[0,1] A[1,1] A[2,1] ... A[mc-1,1]
//     ...
//
// Thus:
//
//     Ap[k * mc + i]
//
// gives A[i,k].
//
// This changes the access pattern from:
//
//     original:
//         A[i * lda + k]
//
// to:
//
//     packed:
//         Ap[k * mc + i]
//
// which makes the MR elements required by the micro-kernel
// contiguous.
//
// ============================================================

void pack_A(const float* A, int lda, float* Ap, int mc, int kc)
{
    for (int k = 0; k < kc; ++k) {
        for (int i = 0; i < mc; ++i) {
            Ap[k * mc + i] = A[i * lda + k];
        }
    }
}


// ============================================================
// Packing B
// ============================================================
//
// Original B:
//
//     B[k][j]
//
// row-major:
//
//     B[k * ldb + j]
//
// The micro-kernel processes NR=16 columns at once.
//
// Therefore we want:
//
//     B[k][j ... j+15]
//
// to be contiguous.
//
// Packed B:
//
//     Bp[k][j]
//
// Memory:
//
//     k=0:
//         b00 b01 b02 ... b15
//
//     k=1:
//         b10 b11 b12 ... b15
//
// etc.
//
// Therefore:
//
//     Bp[k * nc + j]
//
// ============================================================

void pack_B(const float* B, int ldb, float* Bp, int kc, int nc)
{
    for (int k = 0; k < kc; ++k) {
        for (int j = 0; j < nc; ++j) {
            Bp[k * nc + j] = B[k * ldb + j];
        }
    }
}


// ============================================================
// 6 × 16 AVX2/FMA micro-kernel
// ============================================================
//
// This is the most important part.
//
// We calculate:
//
//     C[6×16] += A[6×K] * B[K×16]
//
// Instead of thinking about 96 independent dot products,
// we use an OUTER-PRODUCT formulation:
//
//     C += A[:,k] * B[k,:]
//
// For each k:
//
//     A[:,k] = 6 × 1
//
//             a0
//             a1
//             a2
//             a3
//             a4
//             a5
//
//     B[k,:] = 1 × 16
//
//             b0 b1 ... b15
//
// Their outer product:
//
//             b0 b1 ... b15
//        a0   x  x  ... x
//        a1   x  x  ... x
//        a2   x  x  ... x
//        a3   x  x  ... x
//        a4   x  x  ... x
//        a5   x  x  ... x
//
// gives a 6×16 matrix.
//
//
// AVX2:
//
//     one __m256 = 8 floats
//
// Therefore 16 columns require:
//
//     2 × __m256
//
// per row.
//
// Total accumulators:
//
//     6 rows × 2 vectors = 12 YMM registers.
//
//
//     c00 c01
//     c10 c11
//     c20 c21
//     c30 c31
//     c40 c41
//     c50 c51
//
// Each accumulator contains 8 floats.
//
// ============================================================

static inline void micro_kernel_6x16(int K, const float* Ap, const float* Bp,
                                     float* C, int ldc, int mc, int nc)
{
    // --------------------------------------------------------
    // 12 accumulators
    //
    // c00/c01 -> C row 0, columns 0..7 / 8..15
    // c10/c11 -> C row 1
    // ...
    // c50/c51 -> C row 5
    //
    // These accumulators stay in registers during the entire
    // K loop.
    // --------------------------------------------------------

    __m256 c00 = _mm256_loadu_ps(C + 0 * ldc + 0);
    __m256 c01 = _mm256_loadu_ps(C + 0 * ldc + 8);
    __m256 c10 = _mm256_loadu_ps(C + 1 * ldc + 0);
    __m256 c11 = _mm256_loadu_ps(C + 1 * ldc + 8);
    __m256 c20 = _mm256_loadu_ps(C + 2 * ldc + 0);
    __m256 c21 = _mm256_loadu_ps(C + 2 * ldc + 8);
    __m256 c30 = _mm256_loadu_ps(C + 3 * ldc + 0);
    __m256 c31 = _mm256_loadu_ps(C + 3 * ldc + 8);
    __m256 c40 = _mm256_loadu_ps(C + 4 * ldc + 0);
    __m256 c41 = _mm256_loadu_ps(C + 4 * ldc + 8);
    __m256 c50 = _mm256_loadu_ps(C + 5 * ldc + 0);
    __m256 c51 = _mm256_loadu_ps(C + 5 * ldc + 8);

    // --------------------------------------------------------
    // K loop
    //
    // Every iteration performs:
    //
    //     C += A[:,k] * B[k,:]
    //
    // which is exactly an outer product.
    // --------------------------------------------------------

    for (int k = 0; k < K; ++k) {
        // ----------------------------------------------------
        // Packed A
        //
        // Ap layout:
        //
        //     [k][0..mc-1]
        //
        // Therefore the six values:
        //
        //     A[0,k]
        //     A[1,k]
        //     ...
        //     A[5,k]
        //
        // are contiguous.
        // ----------------------------------------------------

        const float* a = Ap + k * mc;

        // ----------------------------------------------------
        // Packed B
        //
        // Bp layout:
        //
        //     [k][0..nc-1]
        //
        // The first 16 values are the 16 columns needed by
        // this micro-kernel.
        // ----------------------------------------------------

        const float* b = Bp + k * nc;

        // ----------------------------------------------------
        // Load B[k,0:16]
        //
        // Two YMM registers:
        //
        //     b0 = b[0:8]
        //     b1 = b[8:16]
        //
        // These are reused for all six rows.
        // ----------------------------------------------------

        __m256 b0 = _mm256_loadu_ps(b + 0);
        __m256 b1 = _mm256_loadu_ps(b + 8);

        // ----------------------------------------------------
        // Broadcast A scalars
        //
        // Each A[i,k] is broadcast to 8 lanes.
        //
        // Example:
        //
        //     a0 = [a0 a0 a0 a0 a0 a0 a0 a0]
        //
        // Then:
        //
        //     a0 * b0
        //
        // computes:
        //
        //     [a0*b0, a0*b1, ..., a0*b7]
        //
        // ----------------------------------------------------

        __m256 a0 = _mm256_set1_ps(a[0]);
        __m256 a1 = _mm256_set1_ps(a[1]);
        __m256 a2 = _mm256_set1_ps(a[2]);
        __m256 a3 = _mm256_set1_ps(a[3]);
        __m256 a4 = _mm256_set1_ps(a[4]);
        __m256 a5 = _mm256_set1_ps(a[5]);

        // ----------------------------------------------------
        // Outer product
        //
        // C[i,:] += A[i,k] * B[k,:]
        //
        // FMA:
        //
        //     accumulator += a * b
        //
        // Each instruction performs 8 FMAs in parallel.
        // ----------------------------------------------------

        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
        c40 = _mm256_fmadd_ps(a4, b0, c40);
        c41 = _mm256_fmadd_ps(a4, b1, c41);
        c50 = _mm256_fmadd_ps(a5, b0, c50);
        c51 = _mm256_fmadd_ps(a5, b1, c51);
    }

    // --------------------------------------------------------
    // Store the 6×16 result.
    // --------------------------------------------------------

    _mm256_storeu_ps(C + 0 * ldc + 0, c00);
    _mm256_storeu_ps(C + 0 * ldc + 8, c01);
    _mm256_storeu_ps(C + 1 * ldc + 0, c10);
    _mm256_storeu_ps(C + 1 * ldc + 8, c11);
    _mm256_storeu_ps(C + 2 * ldc + 0, c20);
    _mm256_storeu_ps(C + 2 * ldc + 8, c21);
    _mm256_storeu_ps(C + 3 * ldc + 0, c30);
    _mm256_storeu_ps(C + 3 * ldc + 8, c31);
    _mm256_storeu_ps(C + 4 * ldc + 0, c40);
    _mm256_storeu_ps(C + 4 * ldc + 8, c41);
    _mm256_storeu_ps(C + 5 * ldc + 0, c50);
    _mm256_storeu_ps(C + 5 * ldc + 8, c51);
}


// ============================================================
// Generic edge micro-kernel
// ============================================================
//
// Matrix dimensions are not always multiples of:
//
//     MR = 6
//     NR = 16
//
// For example:
//
//     M = 100
//
// then:
//
//     96 rows -> full 6×16 kernels
//     remaining 4 rows -> edge kernel
//
// Similarly:
//
//     N = 130
//
// gives:
//
//     128 columns -> full 6×16 kernels
//     remaining 2 columns -> edge kernel
//
// The edge kernel is scalar for simplicity.
//
// Production GEMM libraries usually have specialized fringe
// kernels.
//
// ============================================================

static inline void micro_kernel_edge(int mr, int nr, int K, const float* Ap,
                                     const float* Bp, float* C, int ldc,
                                     int mc, int nc)
{
    std::vector<float> acc(mr * nr);

    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            acc[i * nr + j] = C[i * ldc + j];
        }
    }

    for (int k = 0; k < K; ++k) {
        const float* a = Ap + k * mc;
        const float* b = Bp + k * nc;
        for (int i = 0; i < mr; ++i) {
            for (int j = 0; j < nr; ++j) {
                acc[i * nr + j] += a[i] * b[j];
            }
        }
    }

    for (int i = 0; i < mr; ++i) {
        for (int j = 0; j < nr; ++j) {
            C[i * ldc + j] = acc[i * nr + j];
        }
    }
}


// ============================================================
// Macro-kernel
// ============================================================
//
// Input:
//
//     Ap = packed A, MC × KC
//     Bp = packed B, KC × NC
//
// Output:
//
//     C = MC × NC
//
// Macro-kernel divides the large block:
//
//     MC × NC
//
// into:
//
//     MR × NR
//
// micro-blocks.
//
//
//
//     MC=96
//
//     ┌──────┬──────┬──────┬──────┐
//     │6×16  │6×16  │6×16  │ ...  │
//     ├──────┼──────┼──────┼──────┤
//     │6×16  │6×16  │6×16  │ ...  │
//     ├──────┼──────┼──────┼──────┤
//     │ ...  │ ...  │ ...  │ ...  │
//     └──────┴──────┴──────┴──────┘
//                         NC=128
//
// ============================================================

void macro_kernel(int mc, int nc, int kc, const float* Ap, const float* Bp,
                  float* C, int ldc)
{
    for (int j = 0; j < nc; j += NR) {
        for (int i = 0; i < mc; i += MR) {
            int mr = min_int(MR, mc - i);
            int nr = min_int(NR, nc - j);

            // ------------------------------------------------
            // Full 6×16 block
            // ------------------------------------------------

            if (mr == MR && nr == NR) {
                micro_kernel_6x16(kc, Ap + i, Bp + j, C + i * ldc + j,
                                  ldc, mc, nc);
            } else {
                // Fringe / edge block
                micro_kernel_edge(mr, nr, kc, Ap + i, Bp + j,
                                  C + i * ldc + j, ldc, mc, nc);
            }
        }
    }
}


// ============================================================
// Top-level GEMM
// ============================================================
//
// This implements the complete hierarchy:
//
//     for jc
//         for pc
//             pack B
//
//             for ic
//                 pack A
//                 macro_kernel
//
// This is the classic:
//
//     NC / KC / MC
//
// loop ordering.
//
// ============================================================

void gemm(int M, int N, int K, const float* A, int lda,
          const float* B, int ldb, float* C, int ldc)
{
    if (M <= 0 || N <= 0) return;

    // C = A * B: initialize only the active columns, preserving row padding.
    // Every kernel can then load C for both the first and subsequent K blocks.
    for (int i = 0; i < M; ++i) {
        std::fill_n(C + i * ldc, N, 0.0f);
    }
    if (K <= 0) return;

    // --------------------------------------------------------
    // Allocate packing buffers.
    //
    // Ap:
    //
    //     MC × KC
    //
    // Bp:
    //
    //     KC × NC
    //
    // --------------------------------------------------------

    std::vector<float> Ap(static_cast<size_t>(MC) * KC);
    std::vector<float> Bp(static_cast<size_t>(KC) * NC);

    // --------------------------------------------------------
    // jc loop
    //
    // Partition N into NC-sized blocks.
    //
    // This determines which B/C columns we are working on.
    // --------------------------------------------------------

    for (int jc = 0; jc < N; jc += NC) {
        const int nc = min_int(NC, N - jc);

        // ----------------------------------------------------
        // pc loop
        //
        // Partition K into KC-sized blocks.
        //
        // One KC block is packed and reused.
        // ----------------------------------------------------

        for (int pc = 0; pc < K; pc += KC) {
            const int kc = min_int(KC, K - pc);

            // ------------------------------------------------
            // Pack B
            //
            // B source:
            //
            //     B[pc : pc+kc,
            //       jc : jc+nc]
            //
            // ------------------------------------------------

            pack_B(B + pc * ldb + jc, ldb, Bp.data(), kc, nc);

            // ------------------------------------------------
            // ic loop
            //
            // Partition M into MC-sized blocks.
            // ------------------------------------------------

            for (int ic = 0; ic < M; ic += MC) {
                const int mc = min_int(MC, M - ic);

                // --------------------------------------------
                // Pack A
                //
                // A source:
                //
                //     A[ic : ic+mc,
                //       pc : pc+kc]
                // --------------------------------------------

                pack_A(A + ic * lda + pc, lda, Ap.data(), mc, kc);

                // --------------------------------------------
                // Macro kernel
                //
                // Compute:
                //
                //     C[ic:ic+mc,
                //       jc:jc+nc]
                //
                // +=
                //
                //     Ap × Bp
                // --------------------------------------------

                macro_kernel(mc, nc, kc, Ap.data(), Bp.data(),
                             C + ic * ldc + jc, ldc);
            }
        }
    }
}


// ============================================================
// Reference GEMM
//
// Used only for correctness verification.
//
// C = A × B
// ============================================================

void gemm_reference(int M, int N, int K, const float* A, int lda,
                    const float* B, int ldb, float* C, int ldc)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * lda + k] * B[k * ldb + j];
            }
            C[i * ldc + j] = sum;
        }
    }
}
