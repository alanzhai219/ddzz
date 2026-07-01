#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "branch_predictor_profiler.hpp"
#include "case1_hex.hpp"
#include "case2_filter_positive.hpp"
#include "case3_unrolled_sum.hpp"
#include "case4_compare_date.hpp"
#include "case5_clamp.hpp"
#include "case6_abs.hpp"
#include "case7_mask_select.hpp"
#include "case8_sentinel_search.hpp"
#include "case9_bit_scan.hpp"

#define N 100000

// ================================================================
// 基准测试
// ================================================================

static char hex_chars[] = "0123456789abcdefABCDEF";

struct benchmark_result {
    double seconds;
    long long branches;
    long long misses;
};

static void print_benchmark_result(const char *label,
                                   const struct benchmark_result *result) {
    double miss_rate = result->branches == 0
                           ? 0.0
                           : (100.0 * (double)result->misses / (double)result->branches);
    printf("  %s: %.3f s, branches=%lld, misses=%lld, miss rate=%.2f%%\n",
           label,
           result->seconds,
           result->branches,
           result->misses,
           miss_rate);
}

static void begin_branch_measurement(struct perf_counter *branches,
                                     struct perf_counter *misses,
                                     clock_t *start) {
    *start = clock();
    start_counter(branches);
    start_counter(misses);
}

static struct benchmark_result end_branch_measurement(struct perf_counter *branches,
                                                      struct perf_counter *misses,
                                                      clock_t start) {
    struct benchmark_result result;
    result.misses = stop_counter(misses);
    result.branches = stop_counter(branches);
    result.seconds = (double)(clock() - start) / CLOCKS_PER_SEC;
    return result;
}

template <typename Fn>
static struct benchmark_result run_benchmark(struct perf_counter *branches,
                                             struct perf_counter *misses,
                                             Fn benchmark_body) {
    clock_t start = clock();
    begin_branch_measurement(branches, misses, &start);
    benchmark_body();
    return end_branch_measurement(branches, misses, start);
}

static void fail_validation(const char *case_name, const char *detail) {
    fprintf(stderr, "[validation failed] %s: %s\n", case_name, detail);
    exit(1);
}

static void validate_equal_long_long(const char *case_name,
                                     const char *variant,
                                     long long expected,
                                     long long actual) {
    if (expected != actual) {
        fprintf(stderr,
                "[validation failed] %s: %s expected=%lld actual=%lld\n",
                case_name,
                variant,
                expected,
                actual);
        exit(1);
    }
}

static void validate_equal_int(const char *case_name,
                               const char *variant,
                               int expected,
                               int actual) {
    if (expected != actual) {
        fprintf(stderr,
                "[validation failed] %s: %s expected=%d actual=%d\n",
                case_name,
                variant,
                expected,
                actual);
        exit(1);
    }
}

static void validate_equal_size_t(const char *case_name,
                                  const char *variant,
                                  size_t expected,
                                  size_t actual) {
    if (expected != actual) {
        fprintf(stderr,
                "[validation failed] %s: %s expected=%zu actual=%zu\n",
                case_name,
                variant,
                expected,
                actual);
        exit(1);
    }
}

static void validate_filter_result(const char *case_name,
                                   const int *source,
                                   size_t length) {
    int *branchy = static_cast<int *>(malloc(length * sizeof(int)));
    int *branchless = static_cast<int *>(malloc(length * sizeof(int)));
    int *branchless_bit = static_cast<int *>(malloc(length * sizeof(int)));
    if (branchy == NULL || branchless == NULL || branchless_bit == NULL) {
        fail_validation(case_name, "malloc failed");
    }

    memcpy(branchy, source, length * sizeof(int));
    memcpy(branchless, source, length * sizeof(int));
    memcpy(branchless_bit, source, length * sizeof(int));

    size_t branchy_len = filter_positive_branchy(branchy, length);
    size_t branchless_len = filter_positive_branchless(branchless, length);
    size_t branchless_bit_len = filter_positive_branchless_bit(branchless_bit, length);

    validate_equal_size_t(case_name, "branchless length", branchy_len, branchless_len);
    validate_equal_size_t(case_name,
                          "branchless bit length",
                          branchy_len,
                          branchless_bit_len);

    for (size_t i = 0; i < branchy_len; i++) {
        validate_equal_int(case_name, "branchless value", branchy[i], branchless[i]);
        validate_equal_int(case_name,
                           "branchless bit value",
                           branchy[i],
                           branchless_bit[i]);
    }

    free(branchy);
    free(branchless);
    free(branchless_bit);
}

template <typename T>
static T *allocate_array(const char *case_name, size_t count) {
    T *buffer = static_cast<T *>(malloc(count * sizeof(T)));
    if (buffer == NULL) {
        fail_validation(case_name, "malloc failed");
    }
    return buffer;
}

void generate_random_hex(char *buf, size_t len) {
    for (size_t i = 0; i < len; i++)
        buf[i] = hex_chars[rand() % 22];
    buf[len] = '\0';
}

static void run_case1_hex(struct perf_counter *branch_counter,
                          struct perf_counter *miss_counter) {
    printf("=== 案例1: 查表法 vs if-else ===\n");

    char *hex_input = allocate_array<char>("案例1", N + 1);
    generate_random_hex(hex_input, N);

    long long hex_check_branchy = 0;
    long long hex_check_branchless = 0;
    for (size_t i = 0; i < N; i++) {
        hex_check_branchy += decode_hex_branchy(hex_input[i]);
        hex_check_branchless += decode_hex_branchless((unsigned char)hex_input[i]);
    }
    validate_equal_long_long("案例1", "decode result", hex_check_branchy, hex_check_branchless);

    volatile long long sum1 = 0;
    struct benchmark_result hex_branchy = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    sum1 += decode_hex_branchy(hex_input[i]);
        });

    volatile long long sum2 = 0;
    struct benchmark_result hex_branchless = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    sum2 += decode_hex_branchless((unsigned char)hex_input[i]);
        });

    print_benchmark_result("有分支", &hex_branchy);
    print_benchmark_result("查表法", &hex_branchless);
    printf("  加速比: %.2fx\n\n", hex_branchy.seconds / hex_branchless.seconds);

    free(hex_input);
}

static void run_case2_filter_positive(struct perf_counter *branch_counter,
                                      struct perf_counter *miss_counter) {
    printf("=== 案例2: 条件写入 vs 推测性写入 ===\n");

    int *data1 = allocate_array<int>("案例2", N);
    int *data2 = allocate_array<int>("案例2", N);
    for (int i = 0; i < N; i++) {
        data1[i] = rand() - RAND_MAX / 2;
    }
    memcpy(data2, data1, N * sizeof(int));
    validate_filter_result("案例2", data2, N);

    struct benchmark_result filter_branchy = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++) {
                memcpy(data1, data2, N * sizeof(int));
                filter_positive_branchy(data1, N);
            }
        });

    struct benchmark_result filter_branchless = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++) {
                memcpy(data1, data2, N * sizeof(int));
                filter_positive_branchless(data1, N);
            }
        });

    print_benchmark_result("有分支", &filter_branchy);
    print_benchmark_result("无分支", &filter_branchless);
    printf("  加速比: %.2fx\n\n",
           filter_branchy.seconds / filter_branchless.seconds);

    free(data1);
    free(data2);
}

static void run_case3_unrolled_sum(struct perf_counter *branch_counter,
                                   struct perf_counter *miss_counter) {
    printf("=== 案例3: 简单循环 vs 展开循环 ===\n");

    char *hex_input = allocate_array<char>("案例3", N + 1);
    generate_random_hex(hex_input, N);

    validate_equal_long_long("案例3",
                             "sum result",
                             sum_hex_simple(hex_input, N),
                             sum_hex_unrolled(hex_input, N));

    volatile long long s1 = 0;
    struct benchmark_result loop_simple = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                s1 = sum_hex_simple(hex_input, N);
        });

    volatile long long s2 = 0;
    struct benchmark_result loop_unrolled = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                s2 = sum_hex_unrolled(hex_input, N);
        });

    print_benchmark_result("简单循环", &loop_simple);
    print_benchmark_result("展开循环", &loop_unrolled);
    printf("  加速比:   %.2fx\n\n",
           loop_simple.seconds / loop_unrolled.seconds);

    free(hex_input);
}

static void run_case4_compare_date(struct perf_counter *branch_counter,
                                   struct perf_counter *miss_counter) {
    printf("=== 案例4: 逐字段比较 vs 编码比较 ===\n");

    int *date_y1 = allocate_array<int>("案例4", N);
    int *date_m1 = allocate_array<int>("案例4", N);
    int *date_d1 = allocate_array<int>("案例4", N);
    int *date_y2 = allocate_array<int>("案例4", N);
    int *date_m2 = allocate_array<int>("案例4", N);
    int *date_d2 = allocate_array<int>("案例4", N);
    for (int i = 0; i < N; i++) {
        date_y1[i] = 2000 + rand() % 30;
        date_m1[i] = 1 + rand() % 12;
        date_d1[i] = 1 + rand() % 28;
        date_y2[i] = 2000 + rand() % 30;
        date_m2[i] = 1 + rand() % 12;
        date_d2[i] = 1 + rand() % 28;
    }

    long long date_check_branchy = 0;
    long long date_check_branchless = 0;
    for (size_t i = 0; i < N; i++) {
        int branchy_value = compare_date_branchy(
            date_y1[i], date_m1[i], date_d1[i], date_y2[i], date_m2[i], date_d2[i]);
        int branchless_value = compare_date_branchless(
            date_y1[i], date_m1[i], date_d1[i], date_y2[i], date_m2[i], date_d2[i]);
        int branchy_sign = (branchy_value > 0) - (branchy_value < 0);
        validate_equal_int("案例4", "compare result sign", branchy_sign, branchless_value);
        date_check_branchy += branchy_sign;
        date_check_branchless += branchless_value;
    }
    validate_equal_long_long("案例4",
                             "accumulated compare sign",
                             date_check_branchy,
                             date_check_branchless);

    volatile long long date_sum_branchy = 0;
    struct benchmark_result date_branchy = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    date_sum_branchy += compare_date_branchy(
                        date_y1[i], date_m1[i], date_d1[i], date_y2[i], date_m2[i], date_d2[i]);
        });

    volatile long long date_sum_branchless = 0;
    struct benchmark_result date_branchless = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    date_sum_branchless += compare_date_branchless(
                        date_y1[i], date_m1[i], date_d1[i], date_y2[i], date_m2[i], date_d2[i]);
        });

    print_benchmark_result("逐字段比较", &date_branchy);
    print_benchmark_result("编码比较", &date_branchless);
    printf("  加速比:   %.2fx\n\n",
           date_branchy.seconds / date_branchless.seconds);

    free(date_y1);
    free(date_m1);
    free(date_d1);
    free(date_y2);
    free(date_m2);
    free(date_d2);
}

static void run_case5_clamp(struct perf_counter *branch_counter,
                            struct perf_counter *miss_counter) {
    printf("=== 案例5: clamp 分支 vs 算术 ===\n");

    int *clamp_values = allocate_array<int>("案例5", N);
    int *clamp_lo = allocate_array<int>("案例5", N);
    int *clamp_hi = allocate_array<int>("案例5", N);
    for (int i = 0; i < N; i++) {
        int lo = (rand() % 2001) - 1000;
        int width = 1 + rand() % 200;
        clamp_lo[i] = lo;
        clamp_hi[i] = lo + width;
        clamp_values[i] = (rand() % 4001) - 2000;
    }

    long long clamp_check_branchy = 0;
    long long clamp_check_branchless = 0;
    long long clamp_check_branchless_bit = 0;
    for (size_t i = 0; i < N; i++) {
        int branchy_value = clamp_branchy(clamp_values[i], clamp_lo[i], clamp_hi[i]);
        int branchless_value = clamp_branchless(clamp_values[i], clamp_lo[i], clamp_hi[i]);
        int branchless_bit_value =
            clamp_branchless_bit(clamp_values[i], clamp_lo[i], clamp_hi[i]);
        validate_equal_int("案例5", "cmov result", branchy_value, branchless_value);
        validate_equal_int("案例5", "bit result", branchy_value, branchless_bit_value);
        clamp_check_branchy += branchy_value;
        clamp_check_branchless += branchless_value;
        clamp_check_branchless_bit += branchless_bit_value;
    }
    validate_equal_long_long("案例5",
                             "cmov accumulated result",
                             clamp_check_branchy,
                             clamp_check_branchless);
    validate_equal_long_long("案例5",
                             "bit accumulated result",
                             clamp_check_branchy,
                             clamp_check_branchless_bit);

    volatile long long clamp_sum_branchy = 0;
    struct benchmark_result clamp_branchy_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    clamp_sum_branchy += clamp_branchy(clamp_values[i], clamp_lo[i], clamp_hi[i]);
        });

    volatile long long clamp_sum_branchless = 0;
    struct benchmark_result clamp_branchless_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    clamp_sum_branchless +=
                        clamp_branchless(clamp_values[i], clamp_lo[i], clamp_hi[i]);
        });

    volatile long long clamp_sum_branchless_bit = 0;
    struct benchmark_result clamp_branchless_bit_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    clamp_sum_branchless_bit +=
                        clamp_branchless_bit(clamp_values[i], clamp_lo[i], clamp_hi[i]);
        });

    print_benchmark_result("有分支", &clamp_branchy_result);
    print_benchmark_result("cmov/三元", &clamp_branchless_result);
    print_benchmark_result("位运算", &clamp_branchless_bit_result);
    printf("  加速比(有分支/cmov): %.2fx\n",
           clamp_branchy_result.seconds / clamp_branchless_result.seconds);
    printf("  加速比(有分支/位运算): %.2fx\n\n",
           clamp_branchy_result.seconds / clamp_branchless_bit_result.seconds);

    free(clamp_values);
    free(clamp_lo);
    free(clamp_hi);
}

static void run_case6_abs(struct perf_counter *branch_counter,
                          struct perf_counter *miss_counter) {
    printf("=== 案例6: abs 分支 vs 位运算 ===\n");

    int *abs_values = allocate_array<int>("案例6", N);
    for (int i = 0; i < N; i++) {
        abs_values[i] = rand() - RAND_MAX / 2;
    }

    long long abs_check_branchy = 0;
    long long abs_check_branchless = 0;
    for (size_t i = 0; i < N; i++) {
        int branchy_value = abs_branchy(abs_values[i]);
        int branchless_value = abs_branchless(abs_values[i]);
        validate_equal_int("案例6", "abs result", branchy_value, branchless_value);
        abs_check_branchy += branchy_value;
        abs_check_branchless += branchless_value;
    }
    validate_equal_long_long("案例6",
                             "accumulated abs result",
                             abs_check_branchy,
                             abs_check_branchless);

    volatile long long abs_sum_branchy = 0;
    struct benchmark_result abs_branchy_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    abs_sum_branchy += abs_branchy(abs_values[i]);
        });

    volatile long long abs_sum_branchless = 0;
    struct benchmark_result abs_branchless_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    abs_sum_branchless += abs_branchless(abs_values[i]);
        });

    print_benchmark_result("有分支", &abs_branchy_result);
    print_benchmark_result("位运算", &abs_branchless_result);
    printf("  加速比: %.2fx\n\n",
           abs_branchy_result.seconds / abs_branchless_result.seconds);

    free(abs_values);
}

static void run_case7_mask_select(struct perf_counter *branch_counter,
                                  struct perf_counter *miss_counter) {
    printf("=== 案例7: 掩码选择 ===\n");

    int *conds = allocate_array<int>("案例7", N);
    int *true_values = allocate_array<int>("案例7", N);
    int *false_values = allocate_array<int>("案例7", N);
    for (int i = 0; i < N; i++) {
        conds[i] = rand() & 1;
        true_values[i] = rand() - RAND_MAX / 2;
        false_values[i] = rand() - RAND_MAX / 2;
    }

    long long select_check_branchy = 0;
    long long select_check_branchless = 0;
    for (size_t i = 0; i < N; i++) {
        int branchy_value = select_branchy(conds[i], true_values[i], false_values[i]);
        int branchless_value =
            select_branchless_mask(conds[i], true_values[i], false_values[i]);
        validate_equal_int("案例7", "mask result", branchy_value, branchless_value);
        select_check_branchy += branchy_value;
        select_check_branchless += branchless_value;
    }
    validate_equal_long_long("案例7",
                             "accumulated select result",
                             select_check_branchy,
                             select_check_branchless);

    volatile long long select_sum_branchy = 0;
    struct benchmark_result select_branchy_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    select_sum_branchy +=
                        select_branchy(conds[i], true_values[i], false_values[i]);
        });

    volatile long long select_sum_branchless = 0;
    struct benchmark_result select_branchless_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    select_sum_branchless +=
                        select_branchless_mask(conds[i], true_values[i], false_values[i]);
        });

    print_benchmark_result("有分支", &select_branchy_result);
    print_benchmark_result("掩码选择", &select_branchless_result);
    printf("  加速比: %.2fx\n\n",
           select_branchy_result.seconds / select_branchless_result.seconds);

    free(conds);
    free(true_values);
    free(false_values);
}

static void run_case8_sentinel_search(struct perf_counter *branch_counter,
                                      struct perf_counter *miss_counter) {
    printf("=== 案例8: Sentinel 查找 ===\n");

    unsigned char *search_data = allocate_array<unsigned char>("案例8", N + 1);
    unsigned char target = 0xFF;
    for (int i = 0; i < N; i++) {
        search_data[i] = (unsigned char)(rand() & 0x7F);
    }

    validate_equal_size_t("案例8",
                          "search result",
                          find_byte_branchy(search_data, N, target),
                          find_byte_sentinel(search_data, N, target));

    volatile long long search_sum_branchy = 0;
    struct benchmark_result search_branchy_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 2000; t++)
                search_sum_branchy += find_byte_branchy(search_data, N, target);
        });

    volatile long long search_sum_sentinel = 0;
    struct benchmark_result search_sentinel_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 2000; t++)
                search_sum_sentinel += find_byte_sentinel(search_data, N, target);
        });

    print_benchmark_result("边界检查", &search_branchy_result);
    print_benchmark_result("sentinel", &search_sentinel_result);
    printf("  加速比: %.2fx\n\n",
           search_branchy_result.seconds / search_sentinel_result.seconds);

    free(search_data);
}

static void run_case9_bit_scan(struct perf_counter *branch_counter,
                               struct perf_counter *miss_counter) {
    printf("=== 案例9: 位图扫描 vs ctz ===\n");

    uint32_t *masks = allocate_array<uint32_t>("案例9", N);
    for (int i = 0; i < N; i++) {
        masks[i] = ((uint32_t)rand() << 16) ^ (uint32_t)rand();
        masks[i] |= 1u << (rand() % 32);
    }

    long long bit_scan_check_branchy = 0;
    long long bit_scan_check_branchless = 0;
    for (size_t i = 0; i < N; i++) {
        int branchy_value = first_set_bit_branchy(masks[i]);
        int branchless_value = first_set_bit_branchless(masks[i]);
        validate_equal_int("案例9", "ctz result", branchy_value, branchless_value);
        bit_scan_check_branchy += branchy_value;
        bit_scan_check_branchless += branchless_value;
    }
    validate_equal_long_long("案例9",
                             "accumulated bit-scan result",
                             bit_scan_check_branchy,
                             bit_scan_check_branchless);

    volatile long long bit_scan_sum_branchy = 0;
    struct benchmark_result bit_scan_branchy_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    bit_scan_sum_branchy += first_set_bit_branchy(masks[i]);
        });

    volatile long long bit_scan_sum_branchless = 0;
    struct benchmark_result bit_scan_branchless_result = run_benchmark(
        branch_counter,
        miss_counter,
        [&]() {
            for (int t = 0; t < 1000; t++)
                for (size_t i = 0; i < N; i++)
                    bit_scan_sum_branchless += first_set_bit_branchless(masks[i]);
        });

    print_benchmark_result("逐位扫描", &bit_scan_branchy_result);
    print_benchmark_result("ctz", &bit_scan_branchless_result);
    printf("  加速比: %.2fx\n\n",
           bit_scan_branchy_result.seconds / bit_scan_branchless_result.seconds);

    free(masks);
}

int main() {
    srand(42);

    struct perf_counter branch_counter =
        create_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    struct perf_counter miss_counter =
        create_counter(PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

    run_case1_hex(&branch_counter, &miss_counter);
    run_case2_filter_positive(&branch_counter, &miss_counter);
    run_case3_unrolled_sum(&branch_counter, &miss_counter);
    run_case4_compare_date(&branch_counter, &miss_counter);
    run_case5_clamp(&branch_counter, &miss_counter);
    run_case6_abs(&branch_counter, &miss_counter);
    run_case7_mask_select(&branch_counter, &miss_counter);
    run_case8_sentinel_search(&branch_counter, &miss_counter);
    run_case9_bit_scan(&branch_counter, &miss_counter);

    return 0;
}
