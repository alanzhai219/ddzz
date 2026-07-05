# PAPI (Performance API) 使用教程

## 1. PAPI 简介

PAPI (Performance Application Programming Interface) 是一个跨平台的硬件性能计数器访问库，
提供了统一接口来访问现代 CPU 上的硬件性能监控单元 (PMU)。

### 核心概念

```
 应用程序
    │
    ▼
┌──────────────┐
│  High-Level  │  ← 简单封装，开箱即用 (PAPI_flops, PAPI_ipc ...)
│    API       │
├──────────────┤
│  Low-Level   │  ← 灵活控制，自定义事件组合
│    API       │
├──────────────┤
│  Framework   │  ← 平台无关的核心层
├──────────────┤
│  Components  │  ← CPU / GPU / 网络 / 存储 等子系统
└──────────────┘
```

### 安装路径 (本项目)

```
3rdparty/papi/src/install/
├── include/
│   ├── papi.h              # 主头文件
│   └── papiStdEventDefs.h  # 预设事件定义
├── lib/
│   ├── libpapi.a           # 静态库
│   ├── libpapi.so          # 动态库
│   ├── libpfm.a            # perfmon 辅助库
│   └── libpfm.so
├── bin/                    # 命令行工具
│   ├── papi_avail          # 查看可用事件
│   ├── papi_native_avail   # 查看原生事件
│   └── ...
└── share/
    └── papi/
        └── papi_events.csv # 事件列表
```

### 编译链接

```bash
gcc -o my_program my_program.c \
    -I 3rdparty/papi/src/install/include \
    -L 3rdparty/papi/src/install/lib \
    -lpapi -lpfm \
    -Wl,-rpath,3rdparty/papi/src/install/lib
```

---

## 2. 快速上手：High-Level API

High-Level API 提供了一组无需手动管理 EventSet 的便捷函数。

### 2.1 可用的 High-Level 函数

| 函数 | 功能 |
|------|------|
| `PAPI_flops_rate()` | 测量 MFLOPS 速率 |
| `PAPI_flips_rate()` | 测量固定点指令速率 |
| `PAPI_ipc()` | 测量 IPC (每周期指令数) |
| `PAPI_epc()` | 测量 EPC (每条指令周期数) |
| `PAPI_accum()` | 累计计数 (不重置) |
| `PAPI_num_counters()` | 获取可用计数器数量 |

### 2.2 示例：测量 MFLOPS

```c
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

int main() {
    float real_time, proc_time, mflops;
    long long flpops;
    int retval;

    /* 首次调用初始化并启动计数 */
    retval = PAPI_flops_rate(PAPI_FP_OPS,
                             &real_time, &proc_time,
                             &flpops, &mflops);
    if (retval < PAPI_OK) {
        printf("PAPI_flops_rate 失败: %d\n", retval);
        exit(1);
    }

    /* --- 被测代码 --- */
    volatile double x = 1.0;
    for (int i = 0; i < 1000000; i++) {
        x = x * 1.0001 + 0.0001;
    }
    /* --------------- */

    /* 再次调用获取结果 */
    retval = PAPI_flops_rate(PAPI_FP_OPS,
                             &real_time, &proc_time,
                             &flpops, &mflops);
    if (retval < PAPI_OK) {
        printf("PAPI_flops_rate 失败: %d\n", retval);
        exit(1);
    }

    printf("Real time:  %.6f s\n", real_time);
    printf("Proc time:  %.6f s\n", proc_time);
    printf("FLOPs:      %lld\n", flpops);
    printf("MFLOPS:     %.2f\n", mflops);

    return 0;
}
```

---

## 3. Low-Level API 详解

Low-Level API 提供完全控制，适合精确的性能分析。

### 3.1 核心调用流程

```
┌─────────────────────────┐
│ PAPI_library_init()     │  ← 初始化 PAPI 库
├─────────────────────────┤
│ PAPI_create_eventset()  │  ← 创建事件集
├─────────────────────────┤
│ PAPI_add_event()        │  ← 添加事件 (可多次调用)
├─────────────────────────┤
│ PAPI_start()            │  ← 开始计数
├─────────────────────────┤
│   /* 被测代码 */        │
├─────────────────────────┤
│ PAPI_stop()             │  ← 停止计数，读取结果
├─────────────────────────┤
│ PAPI_cleanup_eventset() │  ← 清空事件集
├─────────────────────────┤
│ PAPI_destroy_eventset() │  ← 销毁事件集
├─────────────────────────┤
│ PAPI_shutdown()         │  ← 关闭 PAPI 库
└─────────────────────────┘
```

### 3.2 核心函数说明

#### PAPI_library_init
```c
int PAPI_library_init(int version);
// version: 传入 PAPI_VER_CURRENT
// 返回: PAPI_VER_CURRENT 表示成功，否则失败
```

#### PAPI_create_eventset
```c
int PAPI_create_eventset(int *EventSet);
// 返回: PAPI_OK 表示成功
```

#### PAPI_add_event / PAPI_add_events
```c
int PAPI_add_event(int EventSet, int EventCode);
int PAPI_add_events(int EventSet, int *EventCodes, int number);
// 返回: PAPI_OK 成功
//       PAPI_ECNFLCT (-8) 硬件计数器资源冲突
//       PAPI_ENOEVNT (-7) 事件不存在
```

#### PAPI_start / PAPI_stop / PAPI_read
```c
int PAPI_start(int EventSet);
int PAPI_stop(int EventSet, long long *values);
int PAPI_read(int EventSet, long long *values);
// values 数组长度等于添加的事件数量
// PAPI_read 可以不停止计数就读取当前值
// PAPI_stop 停止计数并读取累加值
```

### 3.3 完整示例：IPC + Cache Miss 分析

```c
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

#define N 256

static double A[N][N], B[N][N], C[N][N];

static void init_matrices() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = 1.0;
            B[i][j] = 2.0;
            C[i][j] = 0.0;
        }
}

static void matmul() {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}

int main() {
    int retval, EventSet = PAPI_NULL;
    /* 4 个事件 -> 4 个结果值 */
    long long values[4];

    /* 1. 初始化 */
    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI_library_init: %d\n", retval);
        exit(1);
    }

    /* 2. 创建事件集 */
    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_create_eventset: %d\n", retval);
        exit(1);
    }

    /* 3. 添加事件 */
    /* ⚠️ 硬件计数器有数量限制 (本机: 10 个) */
    /* ⚠️ 某些事件共享计数器资源，会返回 PAPI_ECNFLCT */
    PAPI_add_event(EventSet, PAPI_TOT_CYC);  // 总周期数
    PAPI_add_event(EventSet, PAPI_TOT_INS);  // 总指令数
    PAPI_add_event(EventSet, PAPI_L1_DCM);   // L1 数据缓存缺失次数
    PAPI_add_event(EventSet, PAPI_L2_TCM);   // L2 缓存缺失总次数

    /* 4. 开始计数 */
    init_matrices();
    PAPI_start(EventSet);

    /* 5. 被测代码 */
    matmul();

    /* 6. 停止并读取 */
    PAPI_stop(EventSet, values);

    /* 7. 输出结果 */
    long long cycles   = values[0];
    long long insns    = values[1];
    long long l1_miss  = values[2];
    long long l2_miss  = values[3];

    double ipc = (cycles > 0) ? (double)insns / cycles : 0.0;

    printf("========================================\n");
    printf("  PAPI Low-Level API 测量结果\n");
    printf("========================================\n");
    printf("  工作负载:     %dx%d 矩阵乘法\n", N, N);
    printf("  总周期数:     %lld\n", cycles);
    printf("  总指令数:     %lld\n", insns);
    printf("  IPC:          %.4f\n", ipc);
    printf("  L1 Cache Miss: %lld\n", l1_miss);
    printf("  L2 Total Miss: %lld\n", l2_miss);
    printf("========================================\n");

    /* 8. 清理 */
    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();

    return 0;
}
```

---

## 4. 常用 Preset Events 参考

### 4.1 通用性能事件

| Event Code | 含义 | 说明 |
|------------|------|------|
| `PAPI_TOT_CYC` | 总周期数 | CPU 总共消耗的时钟周期 |
| `PAPI_TOT_INS` | 总指令数 | 完成执行的指令总数 |
| `PAPI_TOT_IIS` | 发射指令数 | 发射到执行单元的指令数 |
| `PAPI_LD_INS` | Load 指令数 | |
| `PAPI_SR_INS` | Store 指令数 | |
| `PAPI_INT_INS` | 整数指令数 | |
| `PAPI_FP_INS` | 浮点指令数 | |
| `PAPI_FP_OPS` | 浮点操作数 | 用于 flops/flips 计算 |

### 4.2 缓存事件

| Event Code | 含义 |
|------------|------|
| `PAPI_L1_DCM` | L1 数据缓存缺失 |
| `PAPI_L1_ICM` | L1 指令缓存缺失 |
| `PAPI_L1_TCM` | L1 缓存总缺失 |
| `PAPI_L1_LDM` | L1 读缺失 |
| `PAPI_L1_STM` | L1 写缺失 |
| `PAPI_L2_DCM` | L2 数据缓存缺失 |
| `PAPI_L2_ICM` | L2 指令缓存缺失 |
| `PAPI_L2_TCM` | L2 缓存总缺失 |
| `PAPI_L2_LDM` | L2 读缺失 |
| `PAPI_L2_STM` | L2 写缺失 |
| `PAPI_L3_TCM` | L3 缓存总缺失 |
| `PAPI_L3_LDM` | L3 读缺失 |

### 4.3 分支预测事件

| Event Code | 含义 |
|------------|------|
| `PAPI_BR_CN` | 条件分支指令 |
| `PAPI_BR_TKN` | 条件分支被采纳 |
| `PAPI_BR_NTK` | 条件分支未被采纳 |
| `PAPI_BR_MSP` | 分支误预测 |
| `PAPI_BR_PRC` | 分支正确预测 |
| `PAPI_BR_UCN` | 无条件分支 |

### 4.4 停顿/延迟事件

| Event Code | 含义 |
|------------|------|
| `PAPI_STL_ICY` | 无指令发射的周期数 |
| `PAPI_FUL_ICY` | 满指令发射的周期数 |
| `PAPI_STL_CCY` | 无指令完成的周期数 |
| `PAPI_FUL_CCY` | 满指令完成的周期数 |
| `PAPI_MEM_SCY` | 等待内存访问的停顿周期 |
| `PAPI_MEM_RCY` | 等待内存读的停顿周期 |
| `PAPI_MEM_WCY` | 等待内存写的停顿周期 |

### 4.5 TLB 事件

| Event Code | 含义 |
|------------|------|
| `PAPI_TLB_DM` | 数据 TLB 缺失 |
| `PAPI_TLB_IM` | 指令 TLB 缺失 |

### 4.6 查看可用事件

```bash
# 查看所有可用预设事件
papi_avail

# 查看原生 (native) 事件
papi_native_avail

# 查看特定事件详情
papi_avail -e PAPI_TOT_CYC
```

---

## 5. 错误处理与调试

### 5.1 常见错误码

| 错误码 | 值 | 含义 | 常见原因 |
|--------|----|------|----------|
| `PAPI_OK` | 0 | 成功 | |
| `PAPI_EINVAL` | -1 | 参数无效 | 传入了 PAPI_NULL |
| `PAPI_ENOMEM` | -2 | 内存不足 | |
| `PAPI_ESYS` | -3 | 系统调用失败 | perf_event_open 失败 |
| `PAPI_ENOEVNT` | -7 | 事件不存在 | 该 CPU 不支持此事件 |
| `PAPI_ECNFLCT` | -8 | 计数器冲突 | 事件共享硬件资源 |
| `PAPI_ENOTRUN` | -9 | EventSet 未运行 | stop 前未 start |
| `PAPI_EISRUN` | -10 | EventSet 正在运行 | 重复 start |
| `PAPI_ENOINIT` | -16 | 未初始化 | 忘记调用 init |
| `PAPI_EPERM` | -15 | 权限不足 | 需要 root 或 perf_event_paranoid |

### 5.2 错误处理模板

```c
#define CHECK(expr, msg) do {                              \
    int _ret = (expr);                                      \
    if (_ret != PAPI_OK) {                                  \
        fprintf(stderr, "%s: %s (%d)\n",                    \
                msg, PAPI_strerror(_ret), _ret);            \
        exit(1);                                            \
    }                                                       \
} while(0)

// 使用
CHECK(PAPI_library_init(PAPI_VER_CURRENT), "library_init");
CHECK(PAPI_create_eventset(&EventSet), "create_eventset");
```

### 5.3 调试信息

```c
#include "papi.h"

// ...

/* 获取硬件信息 */
const PAPI_hw_info_t *hw_info;
hw_info = PAPI_get_hardware_info();
printf("CPU 型号:    %s\n", hw_info->model_string);
printf("CPU 核心数:  %d\n", hw_info->totalcpus);

/* 获取组件信息 (硬件计数器数量) */
const PAPI_component_info_t *cmp = PAPI_get_component_info(0);
printf("硬件计数器:  %d\n", cmp->num_cntrs);
printf("最大多路复用: %d\n", cmp->num_mpx_cntrs);

/* 获取事件信息 */
PAPI_event_info_t ev_info;
PAPI_get_event_info(PAPI_TOT_CYC, &ev_info);
printf("事件: %s\n", ev_info.symbol);
printf("描述: %s\n", ev_info.long_descr);
```

---

## 6. 多路复用 (Multiplexing)

当需要监控的事件数量超过硬件计数器数量时，PAPI 通过时间片轮转 (multiplexing)
来"虚拟扩展"计数器数量。

### 6.1 工作原理

```
硬件计数器: 4 个
需要监控事件: 8 个

时刻 0:   事件 A B C D  (使用计数器 0-3)
时刻 1:   事件 E F G H  (切换，使用计数器 0-3)
时刻 2:   事件 A B C D  (再次切换)
  ...
```

### 6.2 示例代码

```c
#include <stdio.h>
#include <stdlib.h>
#include "papi.h"

int main() {
    int retval, EventSet = PAPI_NULL;

    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) exit(1);

    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) exit(1);

    /* 启用多路复用 */
    retval = PAPI_set_multiplex(EventSet);
    if (retval != PAPI_OK) {
        fprintf(stderr, "Multiplex not supported: %d\n", retval);
        exit(1);
    }

    /* 添加超过硬件计数器数量的事件 */
    int events[] = {
        PAPI_TOT_CYC, PAPI_TOT_INS,
        PAPI_L1_DCM,  PAPI_L2_TCM,
        PAPI_TLB_DM,  PAPI_BR_MSP,
        PAPI_LD_INS,  PAPI_SR_INS,
    };
    int num_events = sizeof(events) / sizeof(events[0]);

    retval = PAPI_add_events(EventSet, events, num_events);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_add_events: %s (%d)\n",
                PAPI_strerror(retval), retval);
        exit(1);
    }

    /* 开始计数 */
    PAPI_start(EventSet);

    /* 被测代码 */
    volatile double x = 0.0;
    for (int i = 0; i < 10000000; i++) x += i * 0.001;
    (void)x;

    long long values[num_events];
    PAPI_stop(EventSet, values);

    printf("多路复用测量结果 (num_counters + multiplexing):\n");
    printf("  PAPI_TOT_CYC:  %lld\n", values[0]);
    printf("  PAPI_TOT_INS:  %lld\n", values[1]);
    printf("  PAPI_L1_DCM:   %lld\n", values[2]);
    printf("  PAPI_L2_TCM:   %lld\n", values[3]);
    printf("  PAPI_TLB_DM:   %lld\n", values[4]);
    printf("  PAPI_BR_MSP:   %lld\n", values[5]);
    printf("  PAPI_LD_INS:   %lld\n", values[6]);
    printf("  PAPI_SR_INS:   %lld\n", values[7]);

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();
    return 0;
}
```

### 6.3 多路复用注意事项

- 用 `PAPI_set_multiplex(EventSet)` 开启
- 用 `PAPI_get_opt(PAPI_MAX_MPX_CTRS, ...)` 获取最大多路复用事件数
- 多路复用时计数器值不累加，每次切换会记录并在最终结果中按时间比例缩放
- 短时间运行的多路复用结果可能不够准确

---

## 7. Overflow / Profiling

PAPI 支持在计数器达到阈值时触发回调函数，用于采样分析。

### 7.1 工作原理

```
计数器值
  │
  │        ┌── 触发回调 ──┐
  │        │              │
  │   ─────┤              ├────── 阈值 (overflow threshold)
  │        │              │
  │   ─────┘              └────── 计数器重置
  │
  └──────────────────────────────→ 时间
```

### 7.2 示例代码

```c
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include "papi.h"

#define THRESHOLD 1000000

/* 溢出回调函数 */
void overflow_handler(int EventSet, void *address,
                      long long overflow_vector, void *context)
{
    /* address: 发生溢出时的指令地址 (EIP) */
    fprintf(stderr, "Overflow at %p! vector=0x%llx\n",
            address, overflow_vector);

    /* ⚠️ 回调中不应使用 printf (信号不安全)，这里仅为演示 */
    /* ⚠️ 产品代码中应写入 buffer 或使用原子操作 */
}

int main() {
    int retval, EventSet = PAPI_NULL;

    retval = PAPI_library_init(PAPI_VER_CURRENT);
    if (retval != PAPI_VER_CURRENT) exit(1);

    retval = PAPI_create_eventset(&EventSet);
    if (retval != PAPI_OK) exit(1);

    /* 添加事件 */
    PAPI_add_event(EventSet, PAPI_TOT_INS);

    /* 注册回调 */
    retval = PAPI_overflow(EventSet, PAPI_TOT_INS,
                           THRESHOLD, 0, overflow_handler);
    if (retval != PAPI_OK) {
        fprintf(stderr, "PAPI_overflow: %s (%d)\n",
                PAPI_strerror(retval), retval);
        exit(1);
    }

    /* 开始带溢出的计数 */
    PAPI_start(EventSet);

    /* 被测代码 — 大量指令 */
    volatile double x = 1.0;
    for (int i = 0; i < 50000000; i++) {
        x = x * 1.0000001 + 0.0000001;
    }
    (void)x;

    long long value;
    PAPI_stop(EventSet, &value);

    printf("Total instructions: %lld\n", value);

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_shutdown();
    return 0;
}
```

### 7.3 设置 Domain 与 Granularity

```c
/* 设置监控域 (用户态/内核态) */
int domain = PAPI_DOM_USER | PAPI_DOM_KERNEL;
PAPI_set_opt(PAPI_DEFDOM, &domain);  /* 全局默认域 */
PAPI_set_domain(EventSet, domain);   /* 针对特定 EventSet */

/* 设置粒度 (线程/进程/系统) */
int granul = PAPI_GRN_THR;  /* 线程级别 */
PAPI_set_granularity(EventSet, granul);
```

---

## 8. 命令行工具

### 8.1 papi_avail — 查看可用事件

```bash
# 列出所有预设事件及其可用性
papi_avail

# 查看特定事件
papi_avail -e PAPI_TOT_CYC

# 查看硬件信息
papi_avail -d
```

### 8.2 papi_native_avail — 原生事件

```bash
# 列出 CPU 原生事件 (如 perf 事件)
papi_native_avail
```

### 8.3 papi_command_line — 命令行测量

```bash
# 直接测量任意程序
papi_command_line PAPI_TOT_CYC PAPI_TOT_INS ./my_program
```

### 8.4 papi_cost — 测量 PAPI 自身开销

```bash
papi_cost
```

---

## 9. 实战技巧

### 9.1 事件选择策略

| 分析目标 | 推荐事件组合 |
|----------|-------------|
| 总体性能 | `PAPI_TOT_CYC`, `PAPI_TOT_INS` → IPC |
| 缓存分析 | `PAPI_L1_DCM`, `PAPI_L2_TCM`, `PAPI_TLB_DM` |
| 分支预测 | `PAPI_BR_CN`, `PAPI_BR_MSP` → 误预测率 |
| 内存带宽 | `PAPI_LD_INS`, `PAPI_SR_INS`, `PAPI_MEM_WCY` |
| 指令混合 | `PAPI_INT_INS`, `PAPI_FP_INS`, `PAPI_BR_CN` |

### 9.2 避免常见陷阱

1. **事件冲突** (`PAPI_ECNFLCT`)
   - 某些事件共享硬件计数器资源
   - 解决：减少事件数量、更换事件组合、或使用多路复用

2. **权限问题** (`PAPI_EPERM`)
   ```bash
   # 检查当前设置
   cat /proc/sys/kernel/perf_event_paranoid
   # 临时降低权限要求 (需要 root)
   echo 0 > /proc/sys/kernel/perf_event_paranoid
   ```

3. **虚拟化环境**
   - 虚拟机可能不暴露硬件 PMU → 使用 `PAPI_get_hw_info()` 检查
   - 部分事件在虚拟环境中不可用

4. **多线程程序**
   - 需要在每个线程中调用 `PAPI_thread_init()` 和 `PAPI_register_thread()`
   - 每个线程创建独立的 EventSet

### 9.3 多线程模板

```c
#include <pthread.h>
#include "papi.h"

void *thread_func(void *arg) {
    int EventSet = PAPI_NULL;

    /* 在线程中注册 */
    PAPI_thread_init(pthread_self);
    PAPI_register_thread();

    /* 创建线程专属 EventSet */
    PAPI_create_eventset(&EventSet);
    PAPI_add_event(EventSet, PAPI_TOT_CYC);

    PAPI_start(EventSet);
    /* ... 工作负载 ... */
    long long val;
    PAPI_stop(EventSet, &val);
    printf("Thread %lu cycles: %lld\n", pthread_self(), val);

    PAPI_cleanup_eventset(EventSet);
    PAPI_destroy_eventset(&EventSet);
    PAPI_unregister_thread();
    return NULL;
}
```

---

## 10. 总结

| API 级别 | 适用场景 | 优点 | 缺点 |
|----------|----------|------|------|
| **High-Level** | 快速上手、简单测量 | 代码简洁 | 事件不可定制 |
| **Low-Level** | 精确分析、自由组合 | 完全控制 | 代码量更多 |
| **Multiplexing** | 事件数 > 计数器数 | 突破硬件限制 | 可能不精确 |
| **Overflow** | 采样分析、热点检测 | 最小扰动 | 实现复杂 |

### 推荐学习路径

1. 先使用 `papi_avail` 了解平台支持的事件
2. 用 High-Level API 快速获得 MFLOPS/IPC
3. 用 Low-Level API 精确分析特定瓶颈
4. 事件选太多时引入 Multiplexing
5. 需要采样分析时使用 Overflow

### 相关资源

- PAPI 官网: https://icl.utk.edu/papi/
- 本项目示例代码: `3rdparty/papi/sample/`
- 预设事件列表: `3rdparty/papi/src/install/share/papi/papi_events.csv`