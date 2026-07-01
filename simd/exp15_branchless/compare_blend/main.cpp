#include <emmintrin.h>

#include <stdio.h>

#include "../common/benchmark.hpp"

#define N 1024
#define ROUNDS 200000

static void fill_random_ints(int *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = rand() - RAND_MAX / 2;
    }
}

static void max_branchy(const int *lhs, const int *rhs, int *out, size_t length) {
    for (size_t i = 0; i < length; i++) {
        out[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
}

static void max_compare_blend_sse2(const int *lhs, const int *rhs, int *out, size_t length) {
    size_t i = 0;
    size_t simd_length = length - (length % 4);
    for (; i < simd_length; i += 4) {
        __m128i va = _mm_loadu_si128((const __m128i *)(lhs + i));
        __m128i vb = _mm_loadu_si128((const __m128i *)(rhs + i));
        __m128i mask = _mm_cmpgt_epi32(va, vb);
        __m128i selected_a = _mm_and_si128(mask, va);
        __m128i selected_b = _mm_andnot_si128(mask, vb);
        __m128i result = _mm_or_si128(selected_a, selected_b);
        _mm_storeu_si128((__m128i *)(out + i), result);
    }

    for (; i < length; i++) {
        out[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
}

int main() {
    srand(42);

    int *lhs = simd_exp15::checked_malloc<int>(N);
    int *rhs = simd_exp15::checked_malloc<int>(N);
    int *branchy = simd_exp15::checked_malloc<int>(N);
    int *simd = simd_exp15::checked_malloc<int>(N);

    fill_random_ints(lhs, N);
    fill_random_ints(rhs, N);

    max_branchy(lhs, rhs, branchy, N);
    max_compare_blend_sse2(lhs, rhs, simd, N);
    simd_exp15::validate_equal_buffers(branchy, simd, N);

    double branchy_seconds =
        simd_exp15::benchmark_seconds(ROUNDS, [&]() { max_branchy(lhs, rhs, branchy, N); });

    double simd_seconds = simd_exp15::benchmark_seconds(
        ROUNDS,
        [&]() { max_compare_blend_sse2(lhs, rhs, simd, N); });

    printf("compare + blend example\n");
    printf("  branchy: %.3f s\n", branchy_seconds);
    printf("  simd:    %.3f s\n", simd_seconds);
    printf("  speedup: %.2fx\n", branchy_seconds / simd_seconds);

    free(lhs);
    free(rhs);
    free(branchy);
    free(simd);
    return 0;
}