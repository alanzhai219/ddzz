#include "brgemm_kernel.hpp"

#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

bool allclose(const std::vector<float> &a, const std::vector<float> &b, float tol) {
	if (a.size() != b.size()) return false;
	for (size_t i = 0; i < a.size(); ++i) {
		if (std::fabs(a[i] - b[i]) > tol) return false;
	}
	return true;
}

double benchmark_ms(const std::function<void()> &fn, int iters) {
	const auto t0 = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < iters; ++i) fn();
	const auto t1 = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

} // namespace

int main() {
	constexpr int M = 16;
	constexpr int N = 29; // intentional tail to test remainder path
	constexpr int K = 32;
	constexpr int batch_size = 7;
	constexpr int ldc = N;

	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

	std::vector<std::vector<float>> A_data(batch_size, std::vector<float>(M * K));
	std::vector<std::vector<float>> B_data(batch_size, std::vector<float>(K * N));
	std::vector<const float*> A_ptrs(batch_size), B_ptrs(batch_size);
	for (int b = 0; b < batch_size; ++b) {
		for (float &v : A_data[b]) v = dist(rng);
		for (float &v : B_data[b]) v = dist(rng);
		A_ptrs[b] = A_data[b].data();
		B_ptrs[b] = B_data[b].data();
	}

	std::vector<float> C_ref(M * ldc, 0.0f);
	std::vector<float> C_avx2(M * ldc, 0.0f);
	std::vector<float> C_avx512(M * ldc, 0.0f);
	std::vector<float> C_auto(M * ldc, 0.0f);

	brgemm_f32_ref(A_ptrs.data(), B_ptrs.data(), C_ref.data(), batch_size, M, N, K, ldc);
	brgemm_f32_avx2(A_ptrs.data(), B_ptrs.data(), C_avx2.data(), batch_size, M, N, K, ldc);
	const bool ok_avx2 = allclose(C_ref, C_avx2, 1e-4f);

	bool ok_avx512 = true;
	if (brgemm_cpu_supports_avx512()) {
		brgemm_f32_avx512(A_ptrs.data(), B_ptrs.data(), C_avx512.data(), batch_size, M, N, K, ldc);
		ok_avx512 = allclose(C_ref, C_avx512, 1e-4f);
	}

	brgemm_f32_auto(A_ptrs.data(), B_ptrs.data(), C_auto.data(), batch_size, M, N, K, ldc);
	const bool ok_auto = allclose(C_ref, C_auto, 1e-4f);

	std::cout << "Correctness:" << std::endl;
	std::cout << "  AVX2   : " << (ok_avx2 ? "PASS" : "FAIL") << std::endl;
	std::cout << "  AVX512 : "
			  << (brgemm_cpu_supports_avx512() ? (ok_avx512 ? "PASS" : "FAIL") : "SKIP (unsupported)")
			  << std::endl;
	std::cout << "  AUTO   : " << (ok_auto ? "PASS" : "FAIL") << std::endl;

	constexpr int iters = 200;
	const double t_ref = benchmark_ms([&]() {
		brgemm_f32_ref(A_ptrs.data(), B_ptrs.data(), C_ref.data(), batch_size, M, N, K, ldc);
	}, iters);
	const double t_avx2 = benchmark_ms([&]() {
		brgemm_f32_avx2(A_ptrs.data(), B_ptrs.data(), C_avx2.data(), batch_size, M, N, K, ldc);
	}, iters);

	std::cout << "\nAvg latency (ms) over " << iters << " iterations:" << std::endl;
	std::cout << "  REF    : " << t_ref << std::endl;
	std::cout << "  AVX2   : " << t_avx2 << " (speedup x" << (t_ref / t_avx2) << ")" << std::endl;

	if (brgemm_cpu_supports_avx512()) {
		const double t_avx512 = benchmark_ms([&]() {
			brgemm_f32_avx512(A_ptrs.data(), B_ptrs.data(), C_avx512.data(), batch_size, M, N, K, ldc);
		}, iters);
		std::cout << "  AVX512 : " << t_avx512 << " (speedup x" << (t_ref / t_avx512) << ")" << std::endl;
	}

	return (ok_avx2 && ok_avx512 && ok_auto) ? 0 : 1;
}
