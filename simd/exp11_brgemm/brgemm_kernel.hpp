// BRGEMM (Batch-Reduce GEMM) intrinsic kernels.
// Computes: C = sum_{b=0}^{batch_size-1} A_b * B_b
// A_b: [M x K], B_b: [K x N], C: [M x N]

#pragma once

#include <immintrin.h>

enum class brgemm_isa_t {
    scalar,
    avx2,
    avx512,
};

// Detects ISA support using compiler builtins.
bool brgemm_cpu_supports_avx2();
bool brgemm_cpu_supports_avx512();

// Scalar reference implementation (always available).
void brgemm_f32_ref(
        const float **A_batch,
        const float **B_batch,
        float *C,
        int batch_size,
        int M,
        int N,
        int K,
        int ldc);

// AVX2 implementation. Falls back to scalar if not compiled with AVX2 support.
void brgemm_f32_avx2(
        const float **A_batch,
        const float **B_batch,
        float *C,
        int batch_size,
        int M,
        int N,
        int K,
        int ldc);

// AVX512 implementation. Falls back to scalar if not compiled with AVX512F support.
void brgemm_f32_avx512(
        const float **A_batch,
        const float **B_batch,
        float *C,
        int batch_size,
        int M,
        int N,
        int K,
        int ldc);

// Runtime dispatch helper.
void brgemm_f32_auto(
        const float **A_batch,
        const float **B_batch,
        float *C,
        int batch_size,
        int M,
        int N,
        int K,
        int ldc,
        brgemm_isa_t prefer = brgemm_isa_t::avx512);
