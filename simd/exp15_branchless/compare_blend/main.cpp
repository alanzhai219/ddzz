#include <immintrin.h>

#include <stdio.h>
#include <stdint.h>
#include <string>

#include "../common/benchmark.hpp"
#include "../common/pmu_lite.hpp"

#define N (1u << 18)
#define ROUNDS 1000
#define PMU_ROUNDS 200

#if defined(__clang__)
#define SIMD_EXP15_NOVEC __attribute__((noinline, optnone))
#elif defined(__GNUC__)
#define SIMD_EXP15_NOVEC __attribute__((noinline, optimize("no-tree-vectorize")))
#else
#define SIMD_EXP15_NOVEC
#endif

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

SIMD_EXP15_NOVEC
static void max_branchy_no_vectorize(const int *lhs, const int *rhs, int *out, size_t length) {
    for (size_t i = 0; i < length; i++) {
        out[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
}

static void max_compare_select_avx2(const int *lhs, const int *rhs, int *out, size_t length) {
    size_t i = 0;
    size_t simd_length = length - (length % 8);
    for (; i < simd_length; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i mask = _mm256_cmpgt_epi32(va, vb);
        __m256i selected_a = _mm256_and_si256(mask, va);
        __m256i selected_b = _mm256_andnot_si256(mask, vb);
        __m256i result = _mm256_or_si256(selected_a, selected_b);
        _mm256_storeu_si256((__m256i *)(out + i), result);
    }

    for (; i < length; i++) {
        out[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
}

static void max_avx2_max_epi32(const int *lhs, const int *rhs, int *out, size_t length) {
    size_t i = 0;
    size_t simd_length = length - (length % 8);
    for (; i < simd_length; i += 8) {
        __m256i va = _mm256_loadu_si256((const __m256i *)(lhs + i));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(rhs + i));
        __m256i vmax = _mm256_max_epi32(va, vb);
        _mm256_storeu_si256((__m256i *)(out + i), vmax);
    }

    for (; i < length; i++) {
        out[i] = lhs[i] > rhs[i] ? lhs[i] : rhs[i];
    }
}

static void print_pmu_summary(const char *name, const simd_exp15::pmu::Sample &s) {
    printf("  [pmu] %-18s ns=%lu cycles=%lu instr=%lu ipc=%.2f cpi=%.2f",
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
    if (s.has_stalled_frontend && s.has_cycles && s.cycles > 0) {
        printf(" fe_stall=%.2f%%", (double)s.stalled_frontend * 100.0 / (double)s.cycles);
    }
    if (s.has_stalled_backend && s.has_cycles && s.cycles > 0) {
        printf(" be_stall=%.2f%%", (double)s.stalled_backend * 100.0 / (double)s.cycles);
    }
    printf("\n");
}

int main() {
    srand(42);

    int *lhs = simd_exp15::checked_malloc<int>(N);
    int *rhs = simd_exp15::checked_malloc<int>(N);
    int *branchy_auto = simd_exp15::checked_malloc<int>(N);
    int *branchy_novec = simd_exp15::checked_malloc<int>(N);
    int *simd_select = simd_exp15::checked_malloc<int>(N);
    int *simd_max = simd_exp15::checked_malloc<int>(N);

    fill_random_ints(lhs, N);
    fill_random_ints(rhs, N);

    max_branchy(lhs, rhs, branchy_auto, N);
    max_branchy_no_vectorize(lhs, rhs, branchy_novec, N);
    max_compare_select_avx2(lhs, rhs, simd_select, N);
    max_avx2_max_epi32(lhs, rhs, simd_max, N);

    simd_exp15::validate_equal_buffers(branchy_auto, branchy_novec, N);
    simd_exp15::validate_equal_buffers(branchy_auto, simd_select, N);
    simd_exp15::validate_equal_buffers(branchy_auto, simd_max, N);

    volatile uint32_t guard = 0;

    double branchy_auto_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        max_branchy(lhs, rhs, branchy_auto, N);
        guard = (guard + (uint32_t)branchy_auto[guard & (N - 1)]) & (N - 1);
    });

    double branchy_novec_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        max_branchy_no_vectorize(lhs, rhs, branchy_novec, N);
        guard = (guard + (uint32_t)branchy_novec[guard & (N - 1)]) & (N - 1);
    });

    double simd_select_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        max_compare_select_avx2(lhs, rhs, simd_select, N);
        guard = (guard + (uint32_t)simd_select[guard & (N - 1)]) & (N - 1);
    });

    double simd_max_seconds = simd_exp15::benchmark_seconds(ROUNDS, [&]() {
        max_avx2_max_epi32(lhs, rhs, simd_max, N);
        guard = (guard + (uint32_t)simd_max[guard & (N - 1)]) & (N - 1);
    });

    printf("compare + blend fairness benchmark\n");
    printf("  branchy(auto):      %.3f s\n", branchy_auto_seconds);
    printf("  branchy(no-vec):    %.3f s\n", branchy_novec_seconds);
    printf("  avx2(compare+mask): %.3f s\n", simd_select_seconds);
    printf("  avx2(max_epi32):    %.3f s\n", simd_max_seconds);
    printf("  speedup vs no-vec:  auto=%.2fx, cmp+mask=%.2fx, max=%.2fx\n",
           branchy_novec_seconds / branchy_auto_seconds,
           branchy_novec_seconds / simd_select_seconds,
           branchy_novec_seconds / simd_max_seconds);
    printf("  guard:              %u\n", (unsigned)guard);

    volatile uint32_t pmu_guard = 0;
    simd_exp15::pmu::Sample sample;
    std::string pmu_error;

    const bool pmu_auto_ok = simd_exp15::pmu::measure(
        PMU_ROUNDS,
        [&]() {
            max_branchy(lhs, rhs, branchy_auto, N);
            pmu_guard = (pmu_guard + (uint32_t)branchy_auto[pmu_guard & (N - 1)]) & (N - 1);
        },
        &sample,
        &pmu_error);
    if (pmu_auto_ok) {
        print_pmu_summary("branchy(auto)", sample);
    } else {
        printf("  [pmu] unavailable:  %s\n", pmu_error.c_str());
    }

    if (pmu_auto_ok) {
        if (simd_exp15::pmu::measure(
                PMU_ROUNDS,
                [&]() {
                    max_branchy_no_vectorize(lhs, rhs, branchy_novec, N);
                    pmu_guard = (pmu_guard + (uint32_t)branchy_novec[pmu_guard & (N - 1)]) & (N - 1);
                },
                &sample,
                &pmu_error)) {
            print_pmu_summary("branchy(no-vec)", sample);
        }

        if (simd_exp15::pmu::measure(
                PMU_ROUNDS,
                [&]() {
                    max_compare_select_avx2(lhs, rhs, simd_select, N);
                    pmu_guard = (pmu_guard + (uint32_t)simd_select[pmu_guard & (N - 1)]) & (N - 1);
                },
                &sample,
                &pmu_error)) {
            print_pmu_summary("avx2(compare+mask)", sample);
        }

        if (simd_exp15::pmu::measure(
                PMU_ROUNDS,
                [&]() {
                    max_avx2_max_epi32(lhs, rhs, simd_max, N);
                    pmu_guard = (pmu_guard + (uint32_t)simd_max[pmu_guard & (N - 1)]) & (N - 1);
                },
                &sample,
                &pmu_error)) {
            print_pmu_summary("avx2(max_epi32)", sample);
        }
    }
    printf("  pmu_guard:          %u\n", (unsigned)pmu_guard);

    free(lhs);
    free(rhs);
    free(branchy_auto);
    free(branchy_novec);
    free(simd_select);
    free(simd_max);
    return 0;
}