#include <immintrin.h>

#include <stdio.h>
#include <stdint.h>
#include <string>

#include "../common/benchmark.hpp"
#include "../common/pmu_lite.hpp"

#define N (1u << 18)
#define ROUNDS 5000
#define PMU_ROUNDS 500

#if defined(__clang__)
#define SIMD_EXP15_NOVEC __attribute__((noinline, optnone))
#elif defined(__GNUC__)
#define SIMD_EXP15_NOVEC __attribute__((noinline, optimize("no-tree-vectorize")))
#else
#define SIMD_EXP15_NOVEC
#endif

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

SIMD_EXP15_NOVEC
static size_t find_first_byte_branchy_no_vectorize(const unsigned char *data,
                                                   size_t length,
                                                   unsigned char target) {
    for (size_t i = 0; i < length; i++) {
        if (data[i] == target) {
            return i;
        }
    }
    return length;
}

static size_t find_first_byte_movemask_avx2(const unsigned char *data,
                                            size_t length,
                                            unsigned char target) {
    __m256i vt = _mm256_set1_epi8((char)target);
    size_t i = 0;
    size_t simd_length = length - (length % 32);

    for (; i < simd_length; i += 32) {
        __m256i chunk = _mm256_loadu_si256((const __m256i *)(data + i));
        __m256i cmp = _mm256_cmpeq_epi8(chunk, vt);
        unsigned int mask = (unsigned int)_mm256_movemask_epi8(cmp);
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

    unsigned char *data = simd_exp15::checked_malloc<unsigned char>(N);

    fill_random_bytes(data, N);
    data[N / 2 + 7] = 0xFF;

    size_t branchy_auto_result = find_first_byte_branchy(data, N, 0xFF);
    size_t branchy_novec_result = find_first_byte_branchy_no_vectorize(data, N, 0xFF);
    size_t simd_result = find_first_byte_movemask_avx2(data, N, 0xFF);
    simd_exp15::validate_equal_size(branchy_auto_result, branchy_novec_result);
    simd_exp15::validate_equal_size(branchy_auto_result, simd_result);

    volatile uint32_t guard = 0;

    double branchy_auto_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        size_t pos = find_first_byte_branchy(data, N, 0xFF);
        guard = (guard + (uint32_t)pos + (uint32_t)data[pos]) & (N - 1);
    });

    double branchy_novec_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        size_t pos = find_first_byte_branchy_no_vectorize(data, N, 0xFF);
        guard = (guard + (uint32_t)pos + (uint32_t)data[pos]) & (N - 1);
    });

    double simd_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        size_t pos = find_first_byte_movemask_avx2(data, N, 0xFF);
        guard = (guard + (uint32_t)pos + (uint32_t)data[pos]) & (N - 1);
    });

    printf("movemask + ctz fairness benchmark\n");
    printf("  branchy(auto):   %.3f s\n", branchy_auto_seconds);
    printf("  branchy(no-vec): %.3f s\n", branchy_novec_seconds);
    printf("  avx2(movemask):  %.3f s\n", simd_seconds);
    printf("  speedup vs no-vec: auto=%.2fx, avx2=%.2fx\n",
           branchy_novec_seconds / branchy_auto_seconds,
           branchy_novec_seconds / simd_seconds);
    printf("  guard:           %u\n", (unsigned)guard);

    volatile uint32_t pmu_guard = 0;
    simd_exp15::pmu::Sample sample;
    std::string pmu_error;

    const bool pmu_auto_ok = simd_exp15::pmu::measure(
        PMU_ROUNDS,
        [&]() {
            size_t pos = find_first_byte_branchy(data, N, 0xFF);
            pmu_guard = (pmu_guard + (uint32_t)pos + (uint32_t)data[pos]) & (N - 1);
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
                    size_t pos = find_first_byte_branchy_no_vectorize(data, N, 0xFF);
                    pmu_guard = (pmu_guard + (uint32_t)pos + (uint32_t)data[pos]) & (N - 1);
                },
                &sample,
                &pmu_error)) {
            print_pmu_summary("branchy(no-vec)", sample);
        }

        if (simd_exp15::pmu::measure(
                PMU_ROUNDS,
                [&]() {
                    size_t pos = find_first_byte_movemask_avx2(data, N, 0xFF);
                    pmu_guard = (pmu_guard + (uint32_t)pos + (uint32_t)data[pos]) & (N - 1);
                },
                &sample,
                &pmu_error)) {
            print_pmu_summary("avx2(movemask)", sample);
        }
    }
    printf("  pmu_guard:       %u\n", (unsigned)pmu_guard);

    free(data);
    return 0;
}