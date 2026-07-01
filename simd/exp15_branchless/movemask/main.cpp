#include <emmintrin.h>

#include <stdio.h>

#include "../common/benchmark.hpp"

#define N 4096
#define ROUNDS 200000

static void fill_random_bytes(unsigned char *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = (unsigned char)(rand() & 0x7F);
    }
}

static size_t find_first_byte_branchy(const unsigned char *data,
                                      size_t length,
                                      unsigned char target) {
    for (size_t i = 0; i < length; i++) {
        if (data[i] == target) {
            return i;
        }
    }
    return length;
}

static size_t find_first_byte_movemask(const unsigned char *data,
                                       size_t length,
                                       unsigned char target) {
    __m128i vt = _mm_set1_epi8((char)target);
    size_t i = 0;
    size_t simd_length = length - (length % 16);

    for (; i < simd_length; i += 16) {
        __m128i chunk = _mm_loadu_si128((const __m128i *)(data + i));
        __m128i cmp = _mm_cmpeq_epi8(chunk, vt);
        unsigned int mask = (unsigned int)_mm_movemask_epi8(cmp);
        if (mask != 0) {
            return i + (size_t)__builtin_ctz(mask);
        }
    }

    for (; i < length; i++) {
        if (data[i] == target) {
            return i;
        }
    }
    return length;
}

int main() {
    srand(42);

    unsigned char *data = simd_exp15::checked_malloc<unsigned char>(N);

    fill_random_bytes(data, N);
    data[N / 2 + 7] = 0xFF;

    size_t branchy_result = find_first_byte_branchy(data, N, 0xFF);
    size_t simd_result = find_first_byte_movemask(data, N, 0xFF);
    simd_exp15::validate_equal_size(branchy_result, simd_result);

    volatile size_t sink = 0;

    double branchy_seconds = simd_exp15::benchmark_seconds(
        ROUNDS,
        [&]() { sink += find_first_byte_branchy(data, N, 0xFF); });

    double simd_seconds = simd_exp15::benchmark_seconds(
        ROUNDS,
        [&]() { sink += find_first_byte_movemask(data, N, 0xFF); });

    printf("movemask + ctz example\n");
    printf("  branchy: %.3f s\n", branchy_seconds);
    printf("  simd:    %.3f s\n", simd_seconds);
    printf("  speedup: %.2fx\n", branchy_seconds / simd_seconds);
    printf("  sink:    %zu\n", (size_t)sink);

    free(data);
    return 0;
}