#include <papi.h>
#include <stdio.h>

int main() {
    int events[2] = {PAPI_BR_MSP, PAPI_BR_CN};  // 预测失败, 条件分支总数
    long long values[2];

    PAPI_library_init(PAPI_VER_CURRENT);

    PAPI_start_counters(events, 2);

    // ... 你的代码 ...

    PAPI_stop_counters(values, 2);

    printf("分支预测失败: %lld\n", values[0]);
    printf("条件分支总数: %lld\n", values[1]);
    return 0;
}
