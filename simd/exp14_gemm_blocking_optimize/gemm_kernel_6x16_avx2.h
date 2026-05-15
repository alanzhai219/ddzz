#include <cstddef>
#include <immintrin.h>

void micro_kernel_6x16_avx2(const size_t K,
                            const float* A, size_t lda,
                            const float* B, size_t ldb,
                            float* C, size_t ldc) {
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

    for (size_t k = 0; k < K; k++) {
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
