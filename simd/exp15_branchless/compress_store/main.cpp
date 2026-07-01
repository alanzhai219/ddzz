#include <tmmintrin.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "../common/benchmark.hpp"

#define N 4096
#define ROUNDS 100000

static unsigned char shuffle_table[256][16];
static unsigned char popcount_table[256];

static void init_tables() {
    for (int mask = 0; mask < 256; mask++) {
        int out_index = 0;
        for (int bit = 0; bit < 8; bit++) {
            if (mask & (1 << bit)) {
                shuffle_table[mask][out_index++] = (unsigned char)bit;
            }
        }
        while (out_index < 16) {
            shuffle_table[mask][out_index++] = 0x80;
        }
        popcount_table[mask] = (unsigned char)__builtin_popcount((unsigned int)mask);
    }
}

static void fill_random_bytes(unsigned char *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = (unsigned char)(rand() & 0xFF);
    }
}

static size_t filter_bytes_branchy(const unsigned char *input,
                                   unsigned char *output,
                                   size_t length,
                                   unsigned char threshold) {
    size_t count = 0;
    for (size_t i = 0; i < length; i++) {
        if (input[i] > threshold) {
            output[count++] = input[i];
        }
    }
    return count;
}

static size_t filter_bytes_compress_store(const unsigned char *input,
                                          unsigned char *output,
                                          size_t length,
                                          unsigned char threshold) {
    size_t out_count = 0;
    size_t i = 0;

    __m128i sign = _mm_set1_epi8((char)0x80);
    __m128i threshold_vec = _mm_set1_epi8((char)(threshold ^ 0x80));

    for (; i + 8 <= length; i += 8) {
        __m128i chunk = _mm_loadl_epi64((const __m128i *)(input + i));
        __m128i adjusted = _mm_xor_si128(chunk, sign);
        __m128i cmp = _mm_cmpgt_epi8(adjusted, threshold_vec);
        unsigned int mask = (unsigned int)_mm_movemask_epi8(cmp) & 0xFFu;
        __m128i shuffle = _mm_loadu_si128((const __m128i *)shuffle_table[mask]);
        __m128i packed = _mm_shuffle_epi8(chunk, shuffle);
        _mm_storeu_si128((__m128i *)(output + out_count), packed);
        out_count += (size_t)popcount_table[mask];
    }

    for (; i < length; i++) {
        if (input[i] > threshold) {
            output[out_count++] = input[i];
        }
    }
    return out_count;
}

int main() {
    srand(42);
    init_tables();

    unsigned char *input = simd_exp15::checked_malloc<unsigned char>(N);
    unsigned char *branchy = simd_exp15::checked_malloc<unsigned char>(N);
    unsigned char *simd = simd_exp15::checked_malloc<unsigned char>(N + 16);

    fill_random_bytes(input, N);

    size_t branchy_count = filter_bytes_branchy(input, branchy, N, 127);
    size_t simd_count = filter_bytes_compress_store(input, simd, N, 127);
    simd_exp15::validate_equal_size(branchy_count, simd_count);
    simd_exp15::validate_equal_buffers(branchy, simd, branchy_count);

    double branchy_seconds = simd_exp15::benchmark_seconds(
        ROUNDS,
        [&]() { branchy_count = filter_bytes_branchy(input, branchy, N, 127); });

    double simd_seconds = simd_exp15::benchmark_seconds(
        ROUNDS,
        [&]() { simd_count = filter_bytes_compress_store(input, simd, N, 127); });

    printf("compress-store example\n");
    printf("  branchy: %.3f s\n", branchy_seconds);
    printf("  simd:    %.3f s\n", simd_seconds);
    printf("  speedup: %.2fx\n", branchy_seconds / simd_seconds);
    printf("  kept:    %zu\n", simd_count);

    free(input);
    free(branchy);
    free(simd);
    return 0;
}