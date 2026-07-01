#include <stdio.h>
#include <string.h>

#include "branch_predictor_profiler.hpp"

// ============ 使用示例 ============
int main() {
    // 创建两个计数器：总分支数 和 预测失败数
    struct perf_counter branches = create_counter(
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS);
    struct perf_counter misses = create_counter(
        PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES);

    // === 开始测量 ===
    start_counter(&branches);
    start_counter(&misses);

    // 你要测量的代码
    volatile int sum = 0;
    int data[10000];
    srand(42);
    for (int i = 0; i < 10000; i++)
        data[i] = rand() % 256 - 128;

    for (int i = 0; i < 10000; i++) {
        if (data[i] >= 0)  // 难以预测的分支
            sum += data[i];
    }

    // === 结束测量 ===
    long long branch_count = stop_counter(&branches);
    long long miss_count = stop_counter(&misses);

    printf("总分支数:     %lld\n", branch_count);
    printf("预测失败数:   %lld\n", miss_count);
    printf("失败率:       %.2f%%\n", 100.0 * miss_count / branch_count);
    printf("sum = %d\n", sum);

    close(branches.fd);
    close(misses.fd);
    return 0;
}
