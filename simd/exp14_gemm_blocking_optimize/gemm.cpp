#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

#include <immintrin.h>

#include "gemm_kernel_4x24_avx2.h"
#include "gemm_kernel_6x16_avx2.h"

struct gemm_config {
    // matrix dimensions
    size_t M;
    size_t N;
    size_t K;
    // for cache blocking
    size_t MC;
    size_t NC;
    size_t KC;
    // for reg blocking
    size_t MR;
    size_t NR;
};

void gemm_ref(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

inline void pack_panel(const float* src, size_t ld, size_t rows, size_t cols, float* dst) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            dst[r * cols + c] = src[r * ld + c];
        }
    }
}

// Pack the current A panel A[ic:ic+mc, pc:pc+kc] into a dense row-major buffer:
//
//   packed_A[i * kc + k] = A[(ic + i), (pc + k)]
//
// So the logical layout becomes:
//
//   packed_A =
//     row 0: A[0,0] A[0,1] ... A[0,kc-1]
//     row 1: A[1,0] A[1,1] ... A[1,kc-1]
//     ...
//
// This is exactly what the current micro-kernels expect: when the macro-kernel
// passes `&packed_A[i * kc]`, the micro-kernel can treat each of its MR rows as
// a contiguous slice with leading dimension `kc`, then broadcast A[row, k] from
// `A[row * lda + k]` without touching the original strided matrix.
void pack_a_panel(const float* A, size_t lda, size_t mc, size_t kc, float* packed_A) {
    pack_panel(A, lda, mc, kc, packed_A);
}

// Extract the current B panel B[pc:pc+kc, jc:jc+nc] and pack it into a dense
// row-major buffer so each logical B row is contiguous in memory:
//
//   packed_B[k * nc + j] = B[(pc + k), (jc + j)]
//
// So the logical layout becomes:
//
//   packed_B =
//     row 0   : B[0,0]    B[0,1]    ... B[0,nc-1]
//     row 1   : B[1,0]    B[1,1]    ... B[1,nc-1]
//     ...
//     row kc-1: B[kc-1,0] B[kc-1,1] ... B[kc-1,nc-1]
//
// This matches the outer-product micro-kernel access pattern. For a fixed k,
// the kernel wants one contiguous vector from the k-th row of B, e.g.
// B[k, j:j+15] for 6x16 or B[k, j:j+23] for 4x24. After packing, those loads
// become direct sequential reads from `&packed_B[k * nc + j]`, which avoids the
// original matrix stride and makes vector loads straightforward.
void pack_b_panel(const float* B, size_t ldb, size_t kc, size_t nc, float* packed_B) {
    pack_panel(B, ldb, kc, nc, packed_B);
}


// Each micro-kernel computes a small block of C (MR x NR) += A (MR x K) * B (K x NR).
// And inner-loop iterates over K dimension, where each iteration does:
//   1. Load one row of A (MR elements) and broadcast each element to a vector
//   2. Load one row of B (NR elements) as vectors
//   3. Do FMA to update the MR x NR block of C in registers.
// The macro-kernel calls the micro-kernel in a loop to cover the entire M x N output
// block, and handles packing of A and B panels to match the micro-kernel's expected input layout.
// macro-kernel is used to dispatch the micro-kernel across the entire output block,
// and it handles packing of A and B panels to match the micro-kernel's expected input layout.
void macro_kernel_6x16(size_t mc, size_t nc, size_t kc,
                       const float* packed_A,
                       const float* packed_B,
                       float* C, size_t ldc) {
    constexpr size_t MR = 6;
    constexpr size_t NR = 16;
    for (size_t i = 0; i < mc; i += MR) {
        for (size_t j = 0; j < nc; j += NR) {
            micro_kernel_6x16_avx2(kc,
                                   &packed_A[i * kc], kc,
                                   &packed_B[j], nc,
                                   &C[i * ldc + j], ldc);
        }
    }
}

void macro_kernel_4x24(size_t mc, size_t nc, size_t kc,
                       const float* packed_A,
                       const float* packed_B,
                       float* C, size_t ldc) {
    constexpr size_t MR = 4;
    constexpr size_t NR = 24;
    for (size_t i = 0; i < mc; i += MR) {
        for (size_t j = 0; j < nc; j += NR) {
            micro_kernel_4x24_avx2(kc,
                                   &packed_A[i * kc], kc,
                                   &packed_B[j], nc,
                                   &C[i * ldc + j], ldc);
        }
    }
}

// constexpr size_t MC = 240;
// constexpr size_t NC = 240;
// constexpr size_t KC = 1008;

// constexpr size_t M = 1008;
// constexpr size_t N = 1008;
// constexpr size_t K = 1008;

void gemm_opt_6x16(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t MC, size_t NC, size_t KC) {
    std::fill(C, C + M * N, 0.0f);

    for (size_t jc = 0; jc < N; jc += NC) {
        const size_t nc = std::min(NC, N - jc);

        for (size_t pc = 0; pc < K; pc += KC) {
            const size_t kc = std::min(KC, K - pc);

            std::vector<float> packed_B(kc * nc);
            pack_b_panel(&B[pc * N + jc], N, kc, nc, packed_B.data());

            for (size_t ic = 0; ic < M; ic += MC) {
                const size_t mc = std::min(MC, M - ic);

                std::vector<float> packed_A(mc * kc);
                pack_a_panel(&A[ic * K + pc], K, mc, kc, packed_A.data());

                macro_kernel_6x16(mc, nc, kc,
                                  packed_A.data(),
                                  packed_B.data(),
                                  &C[ic * N + jc], N);
            }
        }
    }
}

void gemm_opt_4x24(const float* A, const float* B, float* C, size_t M, size_t N, size_t K, size_t MC, size_t NC, size_t KC) {
    std::fill(C, C + M * N, 0.0f);

    for (size_t jc = 0; jc < N; jc += NC) {
        const size_t nc = std::min(NC, N - jc);

        for (size_t pc = 0; pc < K; pc += KC) {
            const size_t kc = std::min(KC, K - pc);

            std::vector<float> packed_B(kc * nc);
            pack_b_panel(&B[pc * N + jc], N, kc, nc, packed_B.data());

            for (size_t ic = 0; ic < M; ic += MC) {
                const size_t mc = std::min(MC, M - ic);

                std::vector<float> packed_A(mc * kc);
                pack_a_panel(&A[ic * K + pc], K, mc, kc, packed_A.data());

                macro_kernel_4x24(mc, nc, kc,
                                  packed_A.data(),
                                  packed_B.data(),
                                  &C[ic * N + jc], N);
            }
        }
    }
}

bool verify(const float* C_ref, const float* C_opt, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(C_ref[i] - C_opt[i]) > 1e-5f) {
            return false;
        }
    }
    return true;
}

int main() {
    // 1008 is divisible by both 6/4 on M and 16/24 on N, so both kernels run without tail handling.
    const size_t M = 1008;
    const size_t N = 1008;
    const size_t K = 1008;

    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_ref(M * N);
    std::vector<float> C_opt_6x16(M * N);
    std::vector<float> C_opt_4x24(M * N);

    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            A[i * K + k] = static_cast<float>((i * 7 + k * 3) % 17 - 8) * 0.125f;
        }
    }

    for (size_t k = 0; k < K; ++k) {
        for (size_t j = 0; j < N; ++j) {
            B[k * N + j] = static_cast<float>((k * 5 + j * 11) % 19 - 9) * 0.1f;
        }
    }

    // Run the reference implementation
    const auto ref_begin = std::chrono::steady_clock::now();
    gemm_ref(A.data(), B.data(), C_ref.data(), static_cast<int>(M), static_cast<int>(N), static_cast<int>(K));
    const auto ref_end = std::chrono::steady_clock::now();

    // Run the 6x16 kernel
    const auto opt_6x16_kc1008_begin = std::chrono::steady_clock::now();
    gemm_opt_6x16(A.data(), B.data(), C_opt_6x16.data(), M, N, K, 240, 240, 1008);
    const auto opt_6x16_kc1008_end = std::chrono::steady_clock::now();

    const auto opt_6x16_kc96_begin = std::chrono::steady_clock::now();
    gemm_opt_6x16(A.data(), B.data(), C_opt_6x16.data(), M, N, K, 240, 240, 96);
    const auto opt_6x16_kc96_end = std::chrono::steady_clock::now();

    // Run the 4x24 kernel
    const auto opt_4x24_kc1008_begin = std::chrono::steady_clock::now();
    gemm_opt_4x24(A.data(), B.data(), C_opt_4x24.data(), M, N, K, 240, 240, 1008);
    const auto opt_4x24_kc1008_end = std::chrono::steady_clock::now();

    const auto opt_4x24_kc96_begin = std::chrono::steady_clock::now();
    gemm_opt_4x24(A.data(), B.data(), C_opt_4x24.data(), M, N, K, 240, 240, 96);
    const auto opt_4x24_kc96_end = std::chrono::steady_clock::now();

    const double ref_ms = std::chrono::duration<double, std::milli>(ref_end - ref_begin).count();
    const double opt_6x16_kc1008_ms = std::chrono::duration<double, std::milli>(opt_6x16_kc1008_end - opt_6x16_kc1008_begin).count();
    const double opt_6x16_kc96_ms = std::chrono::duration<double, std::milli>(opt_6x16_kc96_end - opt_6x16_kc96_begin).count();
    const double opt_4x24_kc1008_ms = std::chrono::duration<double, std::milli>(opt_4x24_kc1008_end - opt_4x24_kc1008_begin).count();
    const double opt_4x24_kc96_ms = std::chrono::duration<double, std::milli>(opt_4x24_kc96_end - opt_4x24_kc96_begin).count();

    std::cout << "gemm_ref time: " << ref_ms << " ms" << std::endl;
    std::cout << "gemm_opt_6x16 (KC=1008) time: " << opt_6x16_kc1008_ms << " ms" << std::endl;
    std::cout << "gemm_opt_6x16 (KC=96) time: " << opt_6x16_kc96_ms << " ms" << std::endl;
    std::cout << "gemm_opt_4x24 (KC=1008) time: " << opt_4x24_kc1008_ms << " ms" << std::endl;
    std::cout << "gemm_opt_4x24 (KC=96) time: " << opt_4x24_kc96_ms << " ms" << std::endl;

    const bool pass_6x16 = verify(C_ref.data(), C_opt_6x16.data(), static_cast<int>(M), static_cast<int>(N));
    const bool pass_4x24 = verify(C_ref.data(), C_opt_4x24.data(), static_cast<int>(M), static_cast<int>(N));

    if (pass_6x16 && pass_4x24) {
        std::cout << "GEMM verification passed" << std::endl;
        std::cout << "GEMM 6x16 (KC=1008) speedup: " << ref_ms / opt_6x16_kc1008_ms << "x" << std::endl;
        std::cout << "GEMM 6x16 (KC=96) speedup: " << ref_ms / opt_6x16_kc96_ms << "x" << std::endl;
        std::cout << "GEMM 4x24 (KC=1008) speedup: " << ref_ms / opt_4x24_kc1008_ms << "x" << std::endl;
        std::cout << "GEMM 4x24 (KC=96) speedup: " << ref_ms / opt_4x24_kc96_ms << "x" << std::endl;
        return 0;
    }

    std::cout << "GEMM verification failed" << std::endl;
    return 1;
}
