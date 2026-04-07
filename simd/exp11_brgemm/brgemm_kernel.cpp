#include "brgemm_kernel.hpp"

#include <algorithm>

bool brgemm_cpu_supports_avx2() {
#if defined(__x86_64__) || defined(_M_X64)
	// AVX2 kernel also relies on FMA for fused multiply-add throughput.
	return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
	return false;
#endif
}

bool brgemm_cpu_supports_avx512() {
#if defined(__x86_64__) || defined(_M_X64)
	// AVX512 path in this demo only requires AVX512F.
	return __builtin_cpu_supports("avx512f");
#else
	return false;
#endif
}

void brgemm_f32_ref(
		const float **A_batch,
		const float **B_batch,
		float *C,
		int batch_size,
		int M,
		int N,
		int K,
		int ldc) {
	std::fill(C, C + static_cast<size_t>(M) * static_cast<size_t>(ldc), 0.0f);
	// BRGEMM reduction: accumulate every (A_b * B_b) product into one shared C.
	for (int b = 0; b < batch_size; ++b) {
		const float *A = A_batch[b];
		const float *B = B_batch[b];
		for (int m = 0; m < M; ++m) {
			float *C_row = C + static_cast<size_t>(m) * ldc;
			for (int k = 0; k < K; ++k) {
				const float a = A[static_cast<size_t>(m) * K + k];
				const float *B_row = B + static_cast<size_t>(k) * N;
				// Scalar baseline, useful for correctness and fallback on unsupported ISA.
				for (int n = 0; n < N; ++n) {
					C_row[n] += a * B_row[n];
				}
			}
		}
	}
}

void brgemm_f32_avx2(
		const float **A_batch,
		const float **B_batch,
		float *C,
		int batch_size,
		int M,
		int N,
		int K,
		int ldc) {
#if defined(__AVX2__) && defined(__FMA__)
	// Clear output once, then reduce all batch GEMMs into the same C buffer.
	std::fill(C, C + static_cast<size_t>(M) * static_cast<size_t>(ldc), 0.0f);

	constexpr int VLEN = 8;
	const int n_vec_end = (N / VLEN) * VLEN;

	for (int b = 0; b < batch_size; ++b) {
		const float *A = A_batch[b];
		const float *B = B_batch[b];

		for (int m = 0; m < M; ++m) {
			float *C_row = C + static_cast<size_t>(m) * ldc;

			for (int k = 0; k < K; ++k) {
				const float a = A[static_cast<size_t>(m) * K + k];
				// Broadcast A[m, k] so one scalar multiplies an 8-float slice of B[k, :].
				const __m256 a_bcast = _mm256_set1_ps(a);
				const float *B_row = B + static_cast<size_t>(k) * N;

				int n = 0;
				// Main vectorized body for high-throughput contiguous columns.
				for (; n < n_vec_end; n += VLEN) {
					__m256 c = _mm256_loadu_ps(C_row + n);
					const __m256 bv = _mm256_loadu_ps(B_row + n);
					// FMA performs C += a * B for 8 columns at once.
					c = _mm256_fmadd_ps(a_bcast, bv, c);
					_mm256_storeu_ps(C_row + n, c);
				}
				// Handle N-tail when N is not a multiple of 8.
				for (; n < N; ++n) {
					C_row[n] += a * B_row[n];
				}
			}
		}
	}
#else
	// Compiled without AVX2/FMA flags: keep behavior correct via scalar fallback.
	brgemm_f32_ref(A_batch, B_batch, C, batch_size, M, N, K, ldc);
#endif
}

void brgemm_f32_avx512(
		const float **A_batch,
		const float **B_batch,
		float *C,
		int batch_size,
		int M,
		int N,
		int K,
		int ldc) {
#if defined(__AVX512F__)
	// Same reduction strategy as AVX2, with 16-float vector width.
	std::fill(C, C + static_cast<size_t>(M) * static_cast<size_t>(ldc), 0.0f);

	constexpr int VLEN = 16;
	const int n_vec_end = (N / VLEN) * VLEN;

	for (int b = 0; b < batch_size; ++b) {
		const float *A = A_batch[b];
		const float *B = B_batch[b];

		for (int m = 0; m < M; ++m) {
			float *C_row = C + static_cast<size_t>(m) * ldc;

			for (int k = 0; k < K; ++k) {
				const float a = A[static_cast<size_t>(m) * K + k];
				// AVX512 version: same math as AVX2 but 16 columns per vector step.
				const __m512 a_bcast = _mm512_set1_ps(a);
				const float *B_row = B + static_cast<size_t>(k) * N;

				int n = 0;
				// Vector core loop, followed by scalar cleanup for tail columns.
				for (; n < n_vec_end; n += VLEN) {
					__m512 c = _mm512_loadu_ps(C_row + n);
					const __m512 bv = _mm512_loadu_ps(B_row + n);
					c = _mm512_fmadd_ps(a_bcast, bv, c);
					_mm512_storeu_ps(C_row + n, c);
				}
				// Scalar remainder for non-multiple-of-16 N.
				for (; n < N; ++n) {
					C_row[n] += a * B_row[n];
				}
			}
		}
	}
#else
	// Compiled without AVX512F flags: preserve correctness via scalar fallback.
	brgemm_f32_ref(A_batch, B_batch, C, batch_size, M, N, K, ldc);
#endif
}

void brgemm_f32_auto(
		const float **A_batch,
		const float **B_batch,
		float *C,
		int batch_size,
		int M,
		int N,
		int K,
		int ldc,
		brgemm_isa_t prefer) {
	// Prefer wider ISA when available, then fall back gracefully.
	if (prefer == brgemm_isa_t::avx512 && brgemm_cpu_supports_avx512()) {
		brgemm_f32_avx512(A_batch, B_batch, C, batch_size, M, N, K, ldc);
		return;
	}
	if ((prefer == brgemm_isa_t::avx512 || prefer == brgemm_isa_t::avx2)
			&& brgemm_cpu_supports_avx2()) {
		brgemm_f32_avx2(A_batch, B_batch, C, batch_size, M, N, K, ldc);
		return;
	}
	brgemm_f32_ref(A_batch, B_batch, C, batch_size, M, N, K, ldc);
}
