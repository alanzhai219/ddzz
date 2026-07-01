# profile_light

这个目录放一个非常轻量的 Linux perf 计数封装，用 `perf_event_open` 直接测硬件事件，不依赖 `perf stat` 命令行。

核心文件：

- `profiler_light.hpp`: 轻量封装，负责创建、启动、停止计数器
- `sample.cpp`: 最小可运行 sample，展示怎么测 branch / cache 事件

## 新增 helper

现在头文件里除了通用的：

- `create_counter(type, config)`
- `create_group_counter_leader(type, config)`
- `create_group_counter_member(type, config, leader)`

还新增了：

- `make_hw_cache_config(cache_id, op_id, result_id)`

这个 helper 专门用于 `PERF_TYPE_HW_CACHE` 事件，内部会自动把：

- cache 层级
- 访问类型
- 结果类型

编码到 `config` 里。

例如：

```cpp
create_counter(
   PERF_TYPE_HW_CACHE,
   make_hw_cache_config(PERF_COUNT_HW_CACHE_L1D,
                        PERF_COUNT_HW_CACHE_OP_READ,
                        PERF_COUNT_HW_CACHE_RESULT_MISS));
```

就表示“L1D read miss”。

如果你需要把这些组合格式化成展示名，比如 `LLC_READ_MISS`，更适合在 sample 或调用方本地做，而不是留在公共头文件里。

## Grouped counters

现在这个目录也支持 grouped counters。

为什么要加这层：

- 多个独立 counter 分别 start/stop，会有轻微时间偏差
- PMU 资源不足时，更容易出现 multiplexing 和测量不一致
- group leader 可以让一组事件在同一时间窗口内启停和读取

典型用法是：

```cpp
struct perf_counter leader =
   create_group_counter_leader(PERF_TYPE_HARDWARE,
                        PERF_COUNT_HW_BRANCH_INSTRUCTIONS);

struct perf_counter branch_miss =
   create_group_counter_member(PERF_TYPE_HARDWARE,
                        PERF_COUNT_HW_BRANCH_MISSES,
                        &leader);
```

sample 现在已经切到 grouped 方式读取 6 个事件，减少了多个独立 counter 分开读造成的误差。

另外，group read 现在启用了 `PERF_FORMAT_ID`：

- 读取结果不再依赖“第几个 counter 就是第几个值”这种固定顺序假设
- 会按 event id 把读回来的值映射回对应 counter

这样即使后面调整 member 注册顺序，grouped read 也不会因为顺序耦合而出错。

## 当前保留的接口层次

这一版去掉了高层 RAII 类，头文件里只保留两层更直接的接口：

- 单事件：`create_counter` / `start_counter` / `stop_counter`
- 分组事件：`create_group_counter_leader` / `create_group_counter_member` / `start_counter_group` / `stop_counter_group`

sample 现在直接展示底层 grouped API 的实际用法，这样接口面更小，也更容易看清 `perf_event_open` 的真实调用关系。

## 能测什么

只要底层 PMU 支持，这个封装可以测任何 `perf_event_open` 能暴露的事件。最常见的是：

- `PERF_COUNT_HW_BRANCH_INSTRUCTIONS`
- `PERF_COUNT_HW_BRANCH_MISSES`
- `PERF_COUNT_HW_CACHE_REFERENCES`
- `PERF_COUNT_HW_CACHE_MISSES`

如果你要测更细粒度的 cache 事件，比如 L1D read miss、LLC miss、DTLB miss，可以继续基于 `PERF_TYPE_HW_CACHE` 扩展 `config` 组合逻辑。

现在 sample 已经直接演示了：

- `L1D read miss`
- `LLC read miss`
- grouped read of branch/cache events
- `PERF_FORMAT_ID` 映射回原始 counter 顺序

## Sample 做了什么

`sample.cpp` 里有两个 demo：

1. `count_positive_branchy` vs `count_positive_branchless`
   这个例子主要看 `branches` 和 `branch_misses` 的差异。

2. `touch_sequential` vs `touch_strided`
   这个例子主要看 `cache_refs` 和 `cache_misses` 的差异。

这样一个 sample 里就能同时看到：

- 分支预测失败带来的代价
- 内存访问模式变化带来的 cache miss 代价

## 编译运行

```bash
cd /home/xiuchuan/workspace/ddzz/tools/profile_light
g++ -O2 -std=c++17 sample.cpp -o sample
./sample
```

如果当前机器或权限不允许打开某个 PMU 事件，`perf_event_open` 会失败并直接报错退出。

## 输出如何解读

sample 会打印：

- `time`
- `branches`
- `branch_misses`
- `cache_refs`
- `cache_misses`
- `l1d_read_miss`
- `llc_read_miss`

其中：

- `branch_miss_rate = branch_misses / branches`
- `cache_miss_rate = cache_misses / cache_refs`

细粒度 cache 事件名现在会统一打印成类似：

- `L1D_READ_MISS`
- `LLC_READ_MISS`

如果你要自己拼 `PERF_TYPE_HW_CACHE` 的 grouped member，最直接的写法是：

```cpp
struct perf_counter leader =
   create_group_counter_leader(PERF_TYPE_HARDWARE,
                               PERF_COUNT_HW_BRANCH_INSTRUCTIONS);

struct perf_counter l1d_read_miss =
   create_group_counter_member(PERF_TYPE_HW_CACHE,
                               make_hw_cache_config(PERF_COUNT_HW_CACHE_L1D,
                                                    PERF_COUNT_HW_CACHE_OP_READ,
                                                    PERF_COUNT_HW_CACHE_RESULT_MISS),
                               &leader);
```

建议不要只看单一指标。

典型解读方式：

1. 如果 `branch_misses` 明显下降，而 `time` 也下降，说明控制流重写是有效的。
2. 如果 `cache_misses` 明显上升，同时 `time` 变差，说明数据访问模式更伤 cache。
3. 如果 miss rate 有变化但时间变化不大，说明当前瓶颈可能不在这个事件上。

## 适用范围和限制

- 这是 Linux-only 方案，依赖 `<linux/perf_event.h>`。
- 硬件事件支持度和精度依赖具体 CPU。
- 同时开太多 counter 时，可能出现 PMU multiplexing；grouped counters 只能减少误差，不能保证资源一定足够。
- `PERF_COUNT_HW_CACHE_MISSES` 是抽象事件，不同架构上的含义可能有细微差异。

## 后续可扩展方向

如果你要继续扩这个目录，比较自然的方向有：

1. 增加 `PERF_FORMAT_TOTAL_TIME_ENABLED/RUNNING` 支持，在 multiplexing 时做更稳妥的归一化。
2. 增加 per-thread / pinned CPU 选项，减少调度噪声。
3. 增加更高层的 sample runner，把多组 profile 配置和输出格式统一掉。