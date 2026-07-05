# PAPI 可视化性能分析指南

> 从数据采集到图形化分析 —— 函数级 Timeline 热点追踪与性能指标可视化

---

## 目录

1. [项目概览](#1-项目概览)
2. [环境准备](#2-环境准备)
3. [示例程序说明](#3-示例程序说明)
4. [Timeline Profiler 深入解析](#4-timeline-profiler-深入解析)
5. [可视化脚本使用](#5-可视化脚本使用)
6. [trace.json 数据格式](#6-tracejson-数据格式)
7. [性能分析方法论](#7-性能分析方法论)
8. [自定义扩展指南](#8-自定义扩展指南)

---

## 1. 项目概览

本指南基于已安装的 PAPI 7.3.0 库，提供一套完整的 **函数级性能 Timeline 追踪 + 图形化可视化** 方案。

### 1.1 核心文件

```
3rdparty/papi/sample/
├── papi_low_level.c          # Low-Level API 入门示例
├── papi_multiplex.c          # 多路复用示例
├── papi_timeline_demo.c      # ★ 函数级 Timeline Profiler
├── visualize_timeline.py     # ★ Python 可视化渲染器
├── Makefile                  # 统一编译脚本
└── trace_timeline.png        # 输出图表示例
```

### 1.2 工作流程

```
papi_timeline_demo  (C 程序，使用 PROF_TRACE 宏包装函数)
        │
        │  PAPI_read() 在每个函数入口/出口采集 PMU 计数器
        │  clock_gettime() 记录时间戳
        ▼
   trace.json  (JSON 格式的 Timeline 数据)
        │
        │  Python matplotlib 解析
        ▼
trace_timeline.png  (Gantt 图 + 性能指标汇总表)
```

### 1.3 采集的硬件性能事件

| 事件 | 含义 | 用途 |
|------|------|------|
| `PAPI_TOT_CYC` | 总 CPU 周期数 | 计算 IPC |
| `PAPI_TOT_INS` | 完成执行的指令数 | 计算 IPC |
| `PAPI_L1_DCM` | L1 数据缓存缺失 | 诊断缓存不友好访问 |
| `PAPI_L2_TCM` | L2 缓存总缺失 | 诊断内存层级瓶颈 |
| `PAPI_TLB_DM` | 数据 TLB 缺失 | 诊断页表遍历开销 |
| `PAPI_BR_MSP` | 分支误预测 | 诊断控制流问题 |

---

## 2. 环境准备

### 2.1 PAPI 库路径

本项目 PAPI 已安装在：

```bash
3rdparty/papi/src/install/
├── include/papi.h
├── lib/libpapi.a
├── lib/libpapi.so
├── lib/libpfm.a
├── lib/libpfm.so
└── bin/papi_avail
```

### 2.2 编译所有示例

```bash
cd 3rdparty/papi/sample
make all
```

编译输出：
```
gcc -O2 -I ../src/install/include -o papi_low_level papi_low_level.c \
    -L ../src/install/lib -Wl,-rpath,../src/install/lib -lpapi -lpfm
gcc -O2 -I ../src/install/include -o papi_multiplex papi_multiplex.c ...
gcc -O2 -I ../src/install/include -o papi_timeline_demo papi_timeline_demo.c ...
```

### 2.3 Python 依赖

```bash
pip install matplotlib
```

### 2.4 权限检查

非 root 用户需确保 `perf_event_paranoid <= 1`：

```bash
cat /proc/sys/kernel/perf_event_paranoid
# 0 或 1 → OK
# 2 或以上 → 需要 root 或调整内核参数
```

---

## 3. 示例程序说明

### 3.1 Low-Level API 示例

**文件**: `papi_low_level.c`

经典的 PAPI Low-Level API 调用流程：

```
PAPI_library_init(PAPI_VER_CURRENT)
    ↓
PAPI_create_eventset(&EventSet)
    ↓
PAPI_add_event(EventSet, PAPI_TOT_CYC)
PAPI_add_event(EventSet, PAPI_TOT_INS)
PAPI_add_event(EventSet, PAPI_L1_DCM)
PAPI_add_event(EventSet, PAPI_L2_TCM)
    ↓
PAPI_start(EventSet)
    ↓
matmul(256)              ← 被测代码 (256×256 矩阵乘法)
    ↓
PAPI_stop(EventSet, values)
    ↓
PAPI_cleanup/destroy/shutdown
```

**运行**:
```bash
./papi_low_level
```

**输出示例**:
```
========================================
  PAPI Low-Level API Sample Results
========================================
  Workload:       256x256 matrix multiply
  Total Cycles:   1220
  Total Instrs:   298
  IPC:            0.2443
  L1 Data Cache Misses:  13
  L2 Total Cache Misses: 153
========================================
```

### 3.2 多路复用示例

**文件**: `papi_multiplex.c`

当事件数 > 硬件计数器数时 (本机 10 个)，PAPI 通过 `PAPI_set_multiplex()` 时间片轮转：

```
硬件计数器: 10, Max Multiplex: 384
Multiplexing enabled.  ← PAPI_assign_eventset_component + PAPI_set_multiplex

Added 8 events: PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_L1_DCM, PAPI_L2_TCM,
                PAPI_TLB_DM, PAPI_BR_MSP, PAPI_LD_INS, PAPI_SR_INS
```

**运行**:
```bash
./papi_multiplex
```

---

## 4. Timeline Profiler 深入解析

### 4.1 架构设计

```
┌──────────────────────────────────────────────────────┐
│                   papi_timeline_demo.c                │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │        Mini Profiler Engine (内嵌)           │     │
│  │                                             │     │
│  │  prof_init()     — 初始化 PAPI + 事件注册    │     │
│  │  tr_begin(name)  — 记录入口 snapshot         │     │
│  │  tr_end()        — 记录出口,计算 delta        │     │
│  │  prof_json()     — 导出 JSON trace文件        │     │
│  │  prof_done()     — 清理资源                   │     │
│  └─────────────────────────────────────────────┘     │
│                                                      │
│  PROF_TRACE("matmul_naive") { matmul_naive(512); }    │
│       ↑                                              │
│       └── for 循环技巧实现 scope-based RAII 自动追踪  │
└──────────────────────────────────────────────────────┘
```

### 4.2 PROF_TRACE 宏原理

```c
#define PROF_TRACE(name) \
    for (int _t = (tr_begin(name), 0); _t < 1; tr_end(), _t++)
```

展开后的等价伪代码：

```c
{
    tr_begin("matmul_naive");      // 第 0 次循环前执行
    for (int _t = 0; _t < 1; ) {
        matmul_naive(512);         // 用户代码
        tr_end();                  // 第 0 次循环后执行
        _t++;                      // _t = 1, 循环退出
    }
}
```

**优势**: 利用 C 的 `for` 循环语法，实现了类似 C++ RAII 的自动追踪效果——
无论函数正常返回还是提前 `return`/`break`，`tr_end()` 都会被调用。

### 4.3 模拟的热点函数

| 函数 | 特点 | 预期瓶颈 |
|------|------|----------|
| `init_matrix` | 512² 个 double 初始化写入 | 内存写入带宽, TLB |
| `matmul_naive` | Naive 三重循环矩阵乘法,k 在内层 → B 按列访问 | L1/L2 Cache Miss 极高 |
| `compute_row_sums` | 顺序累加,数据依赖 | IPC 低 (无指令级并行) |
| `branching_kernel` | 多层嵌套 if-else | 分支预测不友好 |

### 4.4 运行 Trace

```bash
./papi_timeline_demo
```

输出：

```
Profiler initialized with 6 events.
row_sums=inf                   ← 累加结果 (很大)
trace.json written (4 traces)  ← 4 个函数追踪记录
```

### 4.5 trace.json 输出

```json
{
  "event_names": ["PAPI_TOT_CYC","PAPI_TOT_INS","PAPI_L1_DCM",
                  "PAPI_L2_TCM","PAPI_TLB_DM","PAPI_BR_MSP"],
  "traces": [
    {
      "name": "init_matrix",
      "start_us": 0,
      "dur_us": 1622,
      "cycles": 27268,
      "insns": 2065166,
      "ipc": 75.7359,
      "PAPI_L1_DCM": 564,
      "PAPI_L2_TCM": 1498,
      "PAPI_TLB_DM": 6853,
      "PAPI_BR_MSP": 0
    },
    {
      "name": "matmul_naive",
      "start_us": 1627,
      "dur_us": 186528,
      "cycles": 789053408,
      "insns": 939415704,
      "ipc": 1.1906,
      "PAPI_L1_DCM": 134100064,
      "PAPI_L2_TCM": 241852587,
      "PAPI_TLB_DM": 19267,
      "PAPI_BR_MSP": 264189
    },
    ...
  ]
}
```

**关键数据解读**:

- `matmul_naive` 占总时间 99% (`186528 / 188445 us`)
- L1 data cache miss: **1.34 亿次** — 明确指示缓存不友好的列访问模式
- L2 cache miss: **2.42 亿次** — L1 的大量缺失传递到 L2
- IPC: 1.19 — OoO 核心有一定并行度，但被内存瓶颈限制

---

## 5. 可视化脚本使用

### 5.1 基本用法

```bash
# 默认: 读取 trace.json, 输出 trace_timeline.png
python3 visualize_timeline.py

# 指定输入
python3 visualize_timeline.py trace.json

# 指定输入和输出
python3 visualize_timeline.py trace.json my_chart.png
```

### 5.2 图表结构

生成的 PNG 文件包含两个部分：

#### 上半部分: Gantt 图

```
PAPI Function Timeline Profiler
Time (seconds) ──────────────────────────────────────►

┌──────────────────────────────────────────────┐
│ init_matrix        IPC=75.74  Cycles=27K     │  ← 极短, IPC 极高(很少指令/周期)
└──┼───────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ matmul_naive       IPC=1.19   Cycles=789M  L1miss=134M   │  ← 占绝对主导
└──────────────────────────────────────────────────────────┘

  ┌────────┐
  │ rowsums│ IPC=0.49  Cycles=1.9M                          ← 短, IPC 低
  └────────┘
     ┌┐
     └┘ branching  IPC=0.52                                  ← 极短

颜色: 红色(低IPC) → 绿色(高IPC)
```

#### 下半部分: 汇总表

```
┌──────────────────┬──────────┬──────────┬──────────┬────────┬───────────┬──────────┬──────────┬────────────┐
│    Function      │  Time(s) │  Cycles  │  Insns   │  IPC   │ L1D Miss  │ L2 Miss  │TLB Miss  │BranchMisp  │
├──────────────────┼──────────┼──────────┼──────────┼────────┼───────────┼──────────┼──────────┼────────────┤
│   init_matrix    │  0.0016  │   27K    │   2.07M  │ 75.74  │    564    │   1.50K  │  6.85K   │     0      │
│   matmul_naive   │  0.1865  │  789M    │  939M    │ 1.1906 │  134M     │  242M    │  19K     │   264K     │
│ compute_row_sums │  0.0003  │   1.90M  │  927K    │ 0.4889 │   47K     │  110K    │    28    │     0      │
│ branching_kernel │  0.0000  │   3.70K  │   1.92K  │ 0.5182 │    42     │    34    │     2    │     0      │
└──────────────────┴──────────┴──────────┴──────────┴────────┴───────────┴──────────┴──────────┴────────────┘
```

### 5.3 图表含义

| 视觉元素 | 含义 |
|----------|------|
| 条形长度 | 函数执行耗时 |
| 条形颜色 | IPC (绿=高, 红=低) |
| 条形标注 | 函数名 + IPC + Cycles + L1Miss |
| 颜色条 (colorbar) | IPC 数值映射 |

---

## 6. trace.json 数据格式

### 6.1 顶层结构

```jsonc
{
  "event_names": ["event1", "event2", ...],  // 采集的事件列表
  "traces": [                                 // 函数追踪记录数组
    { /* 第 1 个函数 */ },
    { /* 第 2 个函数 */ },
    ...
  ]
}
```

### 6.2 Trace 记录结构

```jsonc
{
  "name":          "function_name",   // 函数名 (PROF_TRACE 的参数)
  "start_us":      1627,              // 相对于首个 trace 的偏移 (微秒)
  "dur_us":        186528,            // 该函数执行时长 (微秒)

  // 核心指标
  "cycles":        789053408,         // PAPI_TOT_CYC delta
  "insns":         939415704,         // PAPI_TOT_INS delta
  "ipc":           1.1906,            // insns / cycles (计算得出)

  // 扩展事件 (键名为 PAPI event name)
  "PAPI_L1_DCM":   134100064,
  "PAPI_L2_TCM":   241852587,
  "PAPI_TLB_DM":   19267,
  "PAPI_BR_MSP":   264189
}
```

### 6.3 自定义扩展

可以修改 `papi_timeline_demo.c` 中的事件数组来采集不同指标：

```c
int events[] = {
    PAPI_TOT_CYC,   // 必须 (用于 IPC 计算)
    PAPI_TOT_INS,   // 必须
    PAPI_L1_DCM,    // 按需
    // ... 添加更多事件
};
```

可视化脚本会自动适配任意数量的事件，只要 `event_names` 和 `traces[*]` 中的键名一致即可。

---

## 7. 性能分析方法论

### 7.1 阅读 Gantt 图找热点

1. **找最长条形** — 这就是耗时最多的函数，优化它收益最大
2. **看颜色** — 红色 (低 IPC) 表示该函数执行效率低下
3. **读标注** — 关注 L1Miss/Cycles 比值，高比值 = 内存瓶颈

### 7.2 典型瓶颈诊断

| 现象 | 诊断 | 优化方向 |
|------|------|----------|
| IPC < 0.5, L1Miss 很高 | Cache 不友好访问模式 | Loop interchange, blocking, prefetch |
| IPC < 0.5, L1Miss 不高 | 指令依赖链过长 | 循环展开, SIMD, 减少数据依赖 |
| BranchMisp 很高 | 分支预测失败 | 消除不可预测分支, 使用 CMOV |
| TLB Miss 很高 | 工作集过大或跨页访问 | 大页 (Huge Page), 数据重排 |
| IPC ≈ 1-2 (OoO 核) | 内存延迟为主 | 预取, 缓存阻塞 (cache blocking) |

### 7.3 从示例数据中总结

以 `matmul_naive` 为例：

```
IPC = 1.19                       ← OoO 核正常范围，但未饱和
L1DCM / Insns = 134M / 939M = 14.3%  ← 极高! 每 7 条指令就有 1 次 L1 miss
L2TCM / Insns = 242M / 939M = 25.8%  ← L2 也扛不住
TLB / Insns   = 19K  / 939M = 0.002% ← TLB 问题不大

结论: 典型的 Cache-unfriendly matmul (k 在内层循环访问 B[k][j] 导致 stride 跳变)
修复: 将循环重排为 i-k-j, 或使用 loop tiling / blocking
```

---

## 8. 自定义扩展指南

### 8.1 为你的代码添加追踪

只需 3 步：

```c
#include "papi.h"   // + 内嵌 profiler 代码 (直接从 papi_timeline_demo.c 复制)

int main() {
    // 1. 初始化
    prof_init(events, names, n_ev);

    // 2. 用 PROF_TRACE 包装所有关键函数
    PROF_TRACE("load_data")    { load_data(); }
    PROF_TRACE("compute")      { compute(); }
    PROF_TRACE("save_results") { save_results(); }

    // 3. 导出
    prof_json("trace.json");
    prof_done();
    return 0;
}
```

### 8.2 添加自定义事件

```c
// 想加 FLOPS 测量? 添加:
PAPI_FP_OPS,    // 浮点操作数

// 想加 memory bandwidth? 添加:
PAPI_LD_INS,    // Load 指令数
PAPI_SR_INS,    // Store 指令数

// 注意: 事件过多需启用 multiplexing (prof_init 已自动处理)
```

### 8.3 扩展到多线程

```c
// 每个线程独立创建 profiler 状态
// 可将 thread_id 写入 trace name:
PROF_TRACE("thread_0_matmul") { matmul(); }
// ...
// 然后导出时按线程合并 JSON
```

---

## 附录: 一键运行脚本

```bash
#!/bin/bash
# run_timeline_analysis.sh — 一键采集 + 可视化

cd 3rdparty/papi/sample

echo "=== Compiling ==="
make papi_timeline_demo

echo "=== Profiling ==="
./papi_timeline_demo

echo "=== Visualizing ==="
python3 visualize_timeline.py trace.json trace_timeline.png

echo "=== Done: trace_timeline.png ==="
ls -lh trace_timeline.png
```

---

*Generated by PAPI Visualization Guide v1.0*