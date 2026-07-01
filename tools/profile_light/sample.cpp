#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "profiler_light.hpp"

#define N 100000
#define ROUNDS 2000

struct profile_result {
    double seconds;
    long long branches;
    long long branch_misses;
    long long cache_refs;
    long long cache_misses;
    long long l1d_read_misses;
    long long llc_read_misses;
};

enum profile_index {
    PROFILE_BRANCHES = 0,
    PROFILE_BRANCH_MISSES = 1,
    PROFILE_CACHE_REFS = 2,
    PROFILE_CACHE_MISSES = 3,
    PROFILE_L1D_READ_MISSES = 4,
    PROFILE_LLC_READ_MISSES = 5,
    PROFILE_COUNTER_COUNT = 6,
};

struct profile_group {
    struct perf_counter counters[PROFILE_COUNTER_COUNT];
    long long values[PROFILE_COUNTER_COUNT];
};

static void fill_random_ints(int *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = rand() - RAND_MAX / 2;
    }
}

static void fill_random_bytes(unsigned char *data, size_t length) {
    for (size_t i = 0; i < length; i++) {
        data[i] = (unsigned char)(rand() & 0xFF);
    }
}

static void format_hw_cache_event_name(uint64_t cache_id,
                                       uint64_t op_id,
                                       uint64_t result_id,
                                       char *buffer,
                                       size_t buffer_size) {
    const char *cache_name = "UNKNOWN_CACHE";
    const char *op_name = "UNKNOWN_OP";
    const char *result_name = "UNKNOWN_RESULT";

    switch (cache_id) {
        case PERF_COUNT_HW_CACHE_L1D: cache_name = "L1D"; break;
        case PERF_COUNT_HW_CACHE_L1I: cache_name = "L1I"; break;
        case PERF_COUNT_HW_CACHE_LL: cache_name = "LLC"; break;
        case PERF_COUNT_HW_CACHE_DTLB: cache_name = "DTLB"; break;
        case PERF_COUNT_HW_CACHE_ITLB: cache_name = "ITLB"; break;
        case PERF_COUNT_HW_CACHE_BPU: cache_name = "BPU"; break;
        case PERF_COUNT_HW_CACHE_NODE: cache_name = "NODE"; break;
    }

    switch (op_id) {
        case PERF_COUNT_HW_CACHE_OP_READ: op_name = "READ"; break;
        case PERF_COUNT_HW_CACHE_OP_WRITE: op_name = "WRITE"; break;
        case PERF_COUNT_HW_CACHE_OP_PREFETCH: op_name = "PREFETCH"; break;
    }

    switch (result_id) {
        case PERF_COUNT_HW_CACHE_RESULT_ACCESS: result_name = "ACCESS"; break;
        case PERF_COUNT_HW_CACHE_RESULT_MISS: result_name = "MISS"; break;
    }

    snprintf(buffer,
             buffer_size,
             "%s_%s_%s",
             cache_name,
             op_name,
             result_name);
}

template <typename Fn>
static struct profile_result measure_profile_group(struct profile_group *group, Fn body) {
    clock_t start = clock();

    start_counter_group(&group->counters[0]);

    body();

    stop_counter_group(&group->counters[0],
                       group->counters,
                       PROFILE_COUNTER_COUNT,
                       group->values);

    struct profile_result result;
    result.branches = group->values[PROFILE_BRANCHES];
    result.branch_misses = group->values[PROFILE_BRANCH_MISSES];
    result.cache_refs = group->values[PROFILE_CACHE_REFS];
    result.cache_misses = group->values[PROFILE_CACHE_MISSES];
    result.l1d_read_misses = group->values[PROFILE_L1D_READ_MISSES];
    result.llc_read_misses = group->values[PROFILE_LLC_READ_MISSES];
    result.seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    return result;
}

static struct profile_group create_profile_group(void) {
    struct profile_group group;
    group.counters[PROFILE_BRANCHES] = create_group_counter_leader(
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    group.counters[PROFILE_BRANCH_MISSES] = create_group_counter_member(
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_BRANCH_MISSES,
        &group.counters[PROFILE_BRANCHES]);
    group.counters[PROFILE_CACHE_REFS] = create_group_counter_member(
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_CACHE_REFERENCES,
        &group.counters[PROFILE_BRANCHES]);
    group.counters[PROFILE_CACHE_MISSES] = create_group_counter_member(
        PERF_TYPE_HARDWARE,
        PERF_COUNT_HW_CACHE_MISSES,
        &group.counters[PROFILE_BRANCHES]);
    group.counters[PROFILE_L1D_READ_MISSES] = create_group_counter_member(
        PERF_TYPE_HW_CACHE,
        make_hw_cache_config(PERF_COUNT_HW_CACHE_L1D,
                             PERF_COUNT_HW_CACHE_OP_READ,
                             PERF_COUNT_HW_CACHE_RESULT_MISS),
        &group.counters[PROFILE_BRANCHES]);
    group.counters[PROFILE_LLC_READ_MISSES] = create_group_counter_member(
        PERF_TYPE_HW_CACHE,
        make_hw_cache_config(PERF_COUNT_HW_CACHE_LL,
                             PERF_COUNT_HW_CACHE_OP_READ,
                             PERF_COUNT_HW_CACHE_RESULT_MISS),
        &group.counters[PROFILE_BRANCHES]);

    for (int i = 0; i < PROFILE_COUNTER_COUNT; i++) {
        group.values[i] = 0;
    }
    return group;
}

static void destroy_profile_group(struct profile_group *group) {
    for (int i = 0; i < PROFILE_COUNTER_COUNT; i++) {
        close_counter(&group->counters[i]);
    }
}

static void print_profile(const char *label,
                          const struct profile_result *result,
                          const char *l1d_name,
                          const char *llc_name) {
    double branch_miss_rate = result->branches == 0
                                  ? 0.0
                                  : 100.0 * (double)result->branch_misses /
                                        (double)result->branches;
    double cache_miss_rate = result->cache_refs == 0
                                 ? 0.0
                                 : 100.0 * (double)result->cache_misses /
                                       (double)result->cache_refs;

    printf("%s\n", label);
    printf("  time:          %.3f s\n", result->seconds);
    printf("  branches:      %lld\n", result->branches);
    printf("  branch_misses: %lld (%.2f%%)\n",
           result->branch_misses,
           branch_miss_rate);
    printf("  cache_refs:    %lld\n", result->cache_refs);
    printf("  cache_misses:  %lld (%.2f%%)\n",
           result->cache_misses,
           cache_miss_rate);
    printf("  %s: %lld\n", l1d_name, result->l1d_read_misses);
    printf("  %s: %lld\n", llc_name, result->llc_read_misses);
}

static int count_positive_branchy(const int *data, size_t length) {
    int count = 0;
    for (size_t i = 0; i < length; i++) {
        if (data[i] >= 0) {
            count++;
        }
    }
    return count;
}

static int count_positive_branchless(const int *data, size_t length) {
    int count = 0;
    for (size_t i = 0; i < length; i++) {
        count += (data[i] >= 0);
    }
    return count;
}

static unsigned long long touch_sequential(const unsigned char *data, size_t length) {
    unsigned long long sum = 0;
    for (size_t i = 0; i < length; i++) {
        sum += data[i];
    }
    return sum;
}

static unsigned long long touch_strided(const unsigned char *data,
                                        size_t length,
                                        size_t stride) {
    unsigned long long sum = 0;
    for (size_t i = 0; i < length; i += stride) {
        sum += data[i];
    }
    return sum;
}

int main() {
    srand(42);

    char l1d_name[64];
    char llc_name[64];
    format_hw_cache_event_name(PERF_COUNT_HW_CACHE_L1D,
                               PERF_COUNT_HW_CACHE_OP_READ,
                               PERF_COUNT_HW_CACHE_RESULT_MISS,
                               l1d_name,
                               sizeof(l1d_name));
    format_hw_cache_event_name(PERF_COUNT_HW_CACHE_LL,
                               PERF_COUNT_HW_CACHE_OP_READ,
                               PERF_COUNT_HW_CACHE_RESULT_MISS,
                               llc_name,
                               sizeof(llc_name));

    struct profile_group profiler = create_profile_group();

    int *ints = (int *)malloc(N * sizeof(int));
    unsigned char *bytes = (unsigned char *)malloc((size_t)N * 64);
    if (ints == NULL || bytes == NULL) {
        fprintf(stderr, "malloc failed\n");
        return 1;
    }

    fill_random_ints(ints, N);
    fill_random_bytes(bytes, (size_t)N * 64);

    int branchy_count = count_positive_branchy(ints, N);
    int branchless_count = count_positive_branchless(ints, N);
    if (branchy_count != branchless_count) {
        fprintf(stderr,
                "validation failed: branchy_count=%d branchless_count=%d\n",
                branchy_count,
                branchless_count);
        destroy_profile_group(&profiler);
        free(ints);
        free(bytes);
        return 1;
    }

    volatile int branch_sink = 0;
    struct profile_result branchy = measure_profile_group(
        &profiler,
        [&]() {
            for (int round = 0; round < ROUNDS; round++) {
                branch_sink += count_positive_branchy(ints, N);
            }
        });

    struct profile_result branchless = measure_profile_group(
        &profiler,
        [&]() {
            for (int round = 0; round < ROUNDS; round++) {
                branch_sink += count_positive_branchless(ints, N);
            }
        });

    printf("=== Demo 1: Branch-heavy vs branchless count ===\n");
    print_profile("branchy count_positive", &branchy, l1d_name, llc_name);
    print_profile("branchless count_positive", &branchless, l1d_name, llc_name);
    printf("  sink:          %d\n\n", branch_sink);

    volatile unsigned long long cache_sink = 0;
    struct profile_result sequential = measure_profile_group(
        &profiler,
        [&]() {
            for (int round = 0; round < ROUNDS; round++) {
                cache_sink += touch_sequential(bytes, (size_t)N * 64);
            }
        });

    struct profile_result strided = measure_profile_group(
        &profiler,
        [&]() {
            for (int round = 0; round < ROUNDS; round++) {
                cache_sink += touch_strided(bytes, (size_t)N * 64, 64);
            }
        });

    printf("=== Demo 2: Sequential vs strided memory access ===\n");
    print_profile("sequential touch", &sequential, l1d_name, llc_name);
    print_profile("strided touch", &strided, l1d_name, llc_name);
    printf("  sink:          %llu\n", cache_sink);

    destroy_profile_group(&profiler);
    free(ints);
    free(bytes);
    return 0;
}