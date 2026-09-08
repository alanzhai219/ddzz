# Linux Perf Profiler V2

单文件、header-only 的 Linux perf profiler。日常使用只需要一个宏。

## 在代码中插入一行

在需要测量的作用域开头加入：

```cpp
LINUX_PERF_PROFILE("vector_sin_update");
```

它会创建一个局部 RAII 对象：进入作用域时开始计数，离开作用域时自动结束并记录。正常返回、提前 `return` 和抛出异常时都会自动结束，不需要手动调用 `finish()` 或 `Init()`。

只需在源文件中包含：

```cpp
#include "linux_perf_advance.hpp"
```

## 编译

```sh
g++ -std=c++11 -O2 -pthread sample_profile.cpp -o sample_profile -lrt
```

## 运行

```sh
LINUX_PERF=dump ./sample_profile
LINUX_PERF=dump=10 ./sample_profile
taskset -c 2,3 env LINUX_PERF=dump:switch-cpu:cpus=2,3 ./sample_profile
LINUX_PERF=dump:BRANCH_MISSES:CPU_MIGRATIONS ./sample_profile
LINUX_PERF=dump:0x10d1 ./sample_profile
```

结果写入当前目录的 `perf_dump.json`：

- `dump`：记录所有命中的 profile scope。
- `dump=10`：每个线程最多记录 10 个 profile scope。
- `switch-cpu:cpus=2,3`：附加记录 CPU 2、3 的上下文切换时间线。

## 通过环境变量添加事件

事件直接写为符号名或 Linux perf 原生 raw config；不支持 `MY_EVENT=...` 形式的自定义别名：

- `BRANCH_MISSES`：直接传入已登记的常用事件名，名称碰撞成功后自动识别为 hardware 或 software event。
- `0x10d1`：未匹配到 hardware/software 事件，因此直接作为 Linux perf 原生 raw config 使用。

解析时依次匹配 hardware 和 software 符号名；均未匹配时，单个整数默认作为 raw event 的 `perf_event_attr::config`。不使用 `hw,`、`sw,` 或 `raw,` 前缀，也不接受 `event,umask,cmask` 多值格式。`dump=N` 和 `cpus=...` 仍使用 `=`。

`switch-cpu:cpus=2,3` 只跟踪 CPU 2、3，同时只为在该 CPU 集合内初始化的线程注册 scope trace。程序如果没有绑定 CPU，初始化时可能运行在其他 CPU，此时 JSON 中只有上下文切换事件，不会出现包含 HW/SW 指标的 scope 事件。因此示例使用 `taskset -c 2,3` 将程序约束到同一 CPU 集合。HW/SW 指标位于 scope 类型 `X` 事件的 `args` 字段中。

需要 Linux `perf_event_open` 权限；受限环境中可能需要调整 `perf_event_paranoid` 或授予 `CAP_PERFMON`。
