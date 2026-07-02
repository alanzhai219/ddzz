#include <immintrin.h>

#include <stdint.h>
#include <stdio.h>
#include <string>

#include "../common/benchmark.hpp"
#include "../common/pmu_lite.hpp"

#define N (1u << 18)
#define ROUNDS 500
#define PMU_ROUNDS 200

#if defined(__clang__)
#define SIMD_EXP15_NOVEC __attribute__((noinline, optnone))
#elif defined(__GNUC__)
#define SIMD_EXP15_NOVEC __attribute__((noinline, optimize("no-tree-vectorize")))
#else
#define SIMD_EXP15_NOVEC
#endif

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

SIMD_EXP15_NOVEC
static size_t filter_bytes_branchy_no_vectorize(const unsigned char *input,
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

static size_t filter_bytes_compress_store_avx2(const unsigned char *input,
                                               unsigned char *output,
                                               size_t length,
                                               unsigned char threshold) {
    size_t out_count = 0;
    size_t i = 0;

    __m256i sign = _mm256_set1_epi8((char)0x80);
    __m256i threshold_vec = _mm256_set1_epi8((char)(threshold ^ 0x80));

    for (; i + 32 <= length; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i *)(input + i));
        __m256i adjusted = _mm256_xor_si256(chunk, sign);
        __m256i cmp = _mm256_cmpgt_epi8(adjusted, threshold_vec);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(cmp);

        while (mask != 0) {
            unsigned int lane = (unsigned int)__builtin_ctz(mask);
            output[out_count++] = input[i + lane];
            mask &= (mask - 1);
        }
    }

    for (; i < length; i++) {
        if (input[i] > threshold) {
            output[out_count++] = input[i];
        }
    }
    return out_count;
}

static void print_pmu_summary(const char *name, const simd_exp15::pmu::Sample &s) {
    printf("  [pmu] %-16s ns=%lu cycles=%lu instr=%lu ipc=%.2f cpi=%.2f",
           name,
           (unsigned long)s.ns,
           (unsigned long)s.cycles,
           (unsigned long)s.instructions,
           s.ipc(),
           s.cpi());
    if (s.has_branch_instructions && s.has_branch_misses) {
        printf(" br_miss=%.2f%%", s.branch_miss_rate() * 100.0);
    }
    if (s.has_cache_misses) {
        printf(" cache_miss/kI=%.2f", s.cache_miss_per_kinst());
    }
    printf("\n");
}

int main() {
    srand(42);

    unsigned char *input = simd_exp15::checked_malloc<unsigned char>(N);
    unsigned char *branchy_auto = simd_exp15::checked_malloc<unsigned char>(N);
    unsigned char *branchy_novec = simd_exp15::checked_malloc<unsigned char>(N);
    unsigned char *simd = simd_exp15::checked_malloc<unsigned char>(N);

    fill_random_bytes(input, N);

    size_t branchy_auto_count = filter_bytes_branchy(input, branchy_auto, N, 127);
    size_t branchy_novec_count = filter_bytes_branchy_no_vectorize(input, branchy_novec, N, 127);
    size_t simd_count = filter_bytes_compress_store_avx2(input, simd, N, 127);
    simd_exp15::validate_equal_size(branchy_auto_count, branchy_novec_count);
    simd_exp15::validate_equal_size(branchy_auto_count, simd_count);
    simd_exp15::validate_equal_buffers(branchy_auto, branchy_novec, branchy_auto_count);
    simd_exp15::validate_equal_buffers(branchy_auto, simd, branchy_auto_count);

    volatile uint32_t guard = 0;

    double branchy_auto_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        branchy_auto_count = filter_bytes_branchy(input, branchy_auto, N, 127);
        uint32_t sample = branchy_auto_count ? (uint32_t)branchy_auto[guard % branchy_auto_count] : 0u;
        guard = (guard + (uint32_t)branchy_auto_count + sample) & (N - 1);
    });

    double branchy_novec_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        branchy_novec_count = filter_bytes_branchy_no_vectorize(input, branchy_novec, N, 127);
        uint32_t sample = branchy_novec_count ? (uint32_t)branchy_novec[guard % branchy_novec_count] : 0u;
        guard = (guard + (uint32_t)branchy_novec_count + sample) & (N - 1);
    });

    double simd_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        simd_count = filter_bytes_compress_store_avx2(input, simd, N, 127);
        uint32_t sample = simd_count ? (uint32_t)simd[guard % simd_count] : 0u;
        guard = (guard + (uint32_t)simd_count + sample) & (N - 1);
    });

    printf("compress-store fairness benchmark\n");
    printf("  branchy(auto):   %.3f s\n", branchy_auto_seconds);
    printf("  branchy(no-vec): %.3f s\n", branchy_novec_seconds);
    printf("  avx2(compress):  %.3f s\n", simd_seconds);
    printf("  speedup vs no-vec: auto=%.2fx, avx2=%.2fx\n",
           branchy_novec_seconds / branchy_auto_seconds,
           branchy_novec_seconds / simd_seconds);
    printf("  kept:            %zu\n", simd_count);
    printf("  guard:           %u\n", (unsigned)guard);

    volatile uint32_t pmu_guard = 0;
    simd_exp15::pmu::Sample sample;
    std::string pmu_error;

    const bool pmu_auto_ok = simd_exp15::pmu::measure(
        PMU_ROUNDS,
        [&]() {
            size_t count = filter_bytes_branchy(input, branchy_auto, N, 127);
            uint32_t v = count ? (uint32_t)branchy_auto[pmu_guard % count] : 0u;
            pmu_guard = (pmu_guard + (uint32_t)count + v) & (N - 1);
        },
        &sample,
        &pmu_error);
    if (pmu_auto_ok) {
        print_pmu_summary("branchy(auto)", sample);
    } else {
        printf("  [pmu] unavailable: %s\n", pmu_error.c_str());
    }

    if (pmu_auto_ok) {
        if (simd_exp15::pmu::measure(
                PMU_ROUNDS,
                [&]() {
                    size_t count = filter_bytes_branchy_no_vectorize(input, branchy_novec, N, 127);
                    uint32_t v = count ? (uint32_t)branchy_novec[pmu_guard % count] : 0u;
                    pmu_guard = (pmu_guard + (uint32_t)count + v) & (N - 1);
                },
                &sample,
                &pmu_error)) {
            print_pmu_summary("branchy(no-vec)", sample);
        }

        if (simd_exp15::pmu::measure(
                PMU_ROUNDS,
                [&]() {
                    size_t count = filter_bytes_compress_store_avx2(input, simd, N, 127);
                    uint32_t v = count ? (uint32_t)simd[pmu_guard % count] : 0u;
                    pmu_guard = (pmu_guard + (uint32_t)count + v) & (N - 1);
                },
                &sample,
                &pmu_error)) {
            print_pmu_summary("avx2(compress)", sample);
        }
    }
    printf("  pmu_guard:       %u\n", (unsigned)pmu_guard);

    free(input);
    free(branchy_auto);
    free(branchy_novec);
    free(simd);
    return 0;
}