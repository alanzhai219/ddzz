#include <immintrin.h>

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__clang__)
#define NOINLINE __attribute__((noinline))
#elif defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

NOINLINE void baseline_auto_max(const int *a, const int *b, int *out, size_t n) {
    for (size_t i = 0; i < n; i++) {
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
}

NOINLINE void manual_simd_max(const int *a, const int *b, int *out, size_t n) {
    size_t i = 0;
    size_t simd_n = n - (n % 8);
    for (; i < simd_n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i vmax = _mm256_max_epi32(va, vb);
        _mm256_storeu_si256((__m256i *)(out + i), vmax);
    }
    for (; i < n; i++) {
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
}

NOINLINE void manual_simd_compare_mask(const int *a, const int *b, int *out, size_t n) {
    size_t i = 0;
    size_t simd_n = n - (n % 8);
    for (; i < simd_n; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + i));
        __m256i mask = _mm256_cmpgt_epi32(va, vb);
        __m256i pick_a = _mm256_and_si256(mask, va);
        __m256i pick_b = _mm256_andnot_si256(mask, vb);
        __m256i vout = _mm256_or_si256(pick_a, pick_b);
        _mm256_storeu_si256((__m256i *)(out + i), vout);
    }
    for (; i < n; i++) {
        out[i] = (a[i] > b[i]) ? a[i] : b[i];
    }
}

static uint32_t checksum(const int *x, size_t n) {
    uint32_t s = 0;
    for (size_t i = 0; i < n; i++) {
        s = (s * 131u) + (uint32_t)x[i];
    }
    return s;
}

int main() {
    const size_t n = 1u << 16;
    int *a = (int *)malloc(n * sizeof(int));
    int *b = (int *)malloc(n * sizeof(int));
    int *o0 = (int *)malloc(n * sizeof(int));
    int *o1 = (int *)malloc(n * sizeof(int));
    int *o2 = (int *)malloc(n * sizeof(int));
    if (!a || !b || !o0 || !o1 || !o2) {
        return 1;
    }

    srand(42);
    for (size_t i = 0; i < n; i++) {
        a[i] = rand() - RAND_MAX / 2;
        b[i] = rand() - RAND_MAX / 2;
    }

    baseline_auto_max(a, b, o0, n);
    manual_simd_max(a, b, o1, n);
    manual_simd_compare_mask(a, b, o2, n);

    uint32_t c0 = checksum(o0, n);
    uint32_t c1 = checksum(o1, n);
    uint32_t c2 = checksum(o2, n);
    printf("checksum baseline_auto   = %u\n", c0);
    printf("checksum manual_simd_max = %u\n", c1);
    printf("checksum cmp+mask        = %u\n", c2);

    free(a);
    free(b);
    free(o0);
    free(o1);
    free(o2);
    return (c0 == c1 && c0 == c2) ? 0 : 2;
}
