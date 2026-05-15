#include <cstddef>
#include <immintrin.h>

void micro_kernel_4x24_avx2(const size_t K,
							const float* A, size_t lda,
							const float* B, size_t ldb,
							float* C, size_t ldc) {
	// 4行 x 3列ymm = 12个累加器
	//           ymm       ymm       ymm
	// r0: [0 ................. 7]    [8 ................ 15]   [16 ............... 23]
	// r1: [0 ................. 7]    [8 ................ 15]   [16 ............... 23]
	// r2: [0 ................. 7]    [8 ................ 15]   [16 ............... 23]
	// r3: [0 ................. 7]    [8 ................ 15]   [16 ............... 23]
	__m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps(), c02 = _mm256_setzero_ps();
	__m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps(), c12 = _mm256_setzero_ps();
	__m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps(), c22 = _mm256_setzero_ps();
	__m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps(), c32 = _mm256_setzero_ps();

	for (size_t k = 0; k < K; ++k) {
		// load B[k, 0:8], B[k, 8:16] 和 B[k, 16:24]
		__m256 b0 = _mm256_loadu_ps(&B[k * ldb]);
		__m256 b1 = _mm256_loadu_ps(&B[k * ldb + 8]);
		__m256 b2 = _mm256_loadu_ps(&B[k * ldb + 16]);

		__m256 a;
		// broadcast A[row, k] 并做 FMA — 每组三段B load被复用4次
		a = _mm256_broadcast_ss(&A[0 * lda + k]);
		c00 = _mm256_fmadd_ps(a, b0, c00);
		c01 = _mm256_fmadd_ps(a, b1, c01);
		c02 = _mm256_fmadd_ps(a, b2, c02);

		a = _mm256_broadcast_ss(&A[1 * lda + k]);
		c10 = _mm256_fmadd_ps(a, b0, c10);
		c11 = _mm256_fmadd_ps(a, b1, c11);
		c12 = _mm256_fmadd_ps(a, b2, c12);

		a = _mm256_broadcast_ss(&A[2 * lda + k]);
		c20 = _mm256_fmadd_ps(a, b0, c20);
		c21 = _mm256_fmadd_ps(a, b1, c21);
		c22 = _mm256_fmadd_ps(a, b2, c22);

		a = _mm256_broadcast_ss(&A[3 * lda + k]);
		c30 = _mm256_fmadd_ps(a, b0, c30);
		c31 = _mm256_fmadd_ps(a, b1, c31);
		c32 = _mm256_fmadd_ps(a, b2, c32);
	}

	// store results: C += accumulated
	_mm256_storeu_ps(&C[0 * ldc],      _mm256_add_ps(_mm256_loadu_ps(&C[0 * ldc]),      c00));
	_mm256_storeu_ps(&C[0 * ldc + 8],  _mm256_add_ps(_mm256_loadu_ps(&C[0 * ldc + 8]),  c01));
	_mm256_storeu_ps(&C[0 * ldc + 16], _mm256_add_ps(_mm256_loadu_ps(&C[0 * ldc + 16]), c02));
	_mm256_storeu_ps(&C[1 * ldc],      _mm256_add_ps(_mm256_loadu_ps(&C[1 * ldc]),      c10));
	_mm256_storeu_ps(&C[1 * ldc + 8],  _mm256_add_ps(_mm256_loadu_ps(&C[1 * ldc + 8]),  c11));
	_mm256_storeu_ps(&C[1 * ldc + 16], _mm256_add_ps(_mm256_loadu_ps(&C[1 * ldc + 16]), c12));
	_mm256_storeu_ps(&C[2 * ldc],      _mm256_add_ps(_mm256_loadu_ps(&C[2 * ldc]),      c20));
	_mm256_storeu_ps(&C[2 * ldc + 8],  _mm256_add_ps(_mm256_loadu_ps(&C[2 * ldc + 8]),  c21));
	_mm256_storeu_ps(&C[2 * ldc + 16], _mm256_add_ps(_mm256_loadu_ps(&C[2 * ldc + 16]), c22));
	_mm256_storeu_ps(&C[3 * ldc],      _mm256_add_ps(_mm256_loadu_ps(&C[3 * ldc]),      c30));
	_mm256_storeu_ps(&C[3 * ldc + 8],  _mm256_add_ps(_mm256_loadu_ps(&C[3 * ldc + 8]),  c31));
	_mm256_storeu_ps(&C[3 * ldc + 16], _mm256_add_ps(_mm256_loadu_ps(&C[3 * ldc + 16]), c32));
}
