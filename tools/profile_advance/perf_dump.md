# perf_dump.json 格式说明

下面对 `perf_dump.json`（位于本工作区的 `ddzz/tools/profile_advance/perf_dump.json`）的结构和字段含义做简要说明，以便用于查看或解析性能跟踪数据。

**概览**

- **schemaVersion**: 顶层整数，表示 trace 文件的版本。
- **traceEvents**: 事件数组，每个元素是一个 trace 事件对象（兼容 Chrome/Chromium Trace Event 格式的子集）。

**常见事件字段**

- **ph**: 事件阶段（phase），常见值：
  - `X`：Complete event（带 `ts` 和 `dur`，表示有时长的事件）。
  - `i`：Instant event（即时事件，通常与 `s` 一起出现表示作用域）。
  - 还有其他 TraceEvent 相位，但本文件示例只用了 `X` 和 `i`。
- **name**: 事件名称（字符串）。
- **cat**: 分类（category），用于分组事件（可选）。
- **pid**: 进程 ID（或进程标签）。
- **tid**: 线程 ID（或线程标签）。
- **ts**: 事件起始时间戳，单位通常为微秒（μs）。示例中 `ts` 值为 `48.673`、`28791.562`，与 `dur` 一起表明时间尺度是微秒级（例如 `dur: 28402.65` ≈ 28.4 ms）。
- **dur**: 事件持续时间（仅对于 `X` 完整事件存在），单位与 `ts` 相同。
- **args**: 键值对象，可包含任意额外度量或元数据（数值、字符串、数组等）。示例中的 `args` 包含 CPU 使用率、频率、CPI、硬件计数器、以及 `Extra Data` 数组等。
- **s**: 当 `ph` 为 `i`（instant）时，`s` 表示作用域（例如 `g`=global, `p`=process, `t`=thread）。

**示例事件解析**

示例片段（来自原 JSON）:

```json
{"ph": "X", "name": "vector_sin_update_0", "cat":"teaching", "pid": 60204, "tid": 60204, "ts": 48.673, "dur": 28402.65, "args": {"CPU Usage":1.00009, "CPU Freq(GHz)":3.95526, "CPI":0.262796, "HW_CPU_CYCLES":"112,350,259", "HW_INSTRUCTIONS":"427,518,101", "HW_CACHE_MISSES":"17,631", "SW_CONTEXT_SWITCHES":"0", "SW_TASK_CLOCK":"28,405,294", "SW_PAGE_FAULTS":"2", "L2_MISS":"745", "Extra Data":[262144,32,3.14159]}}
```

- 该事件为 `X`（完整事件），表示 `vector_sin_update_0` 在时间 `ts=48.673µs` 开始，持续 `dur=28402.65µs`（约 28.4 ms）。
- `args` 中的数字和字符串是自定义采样/计数（例如 `CPU Usage`、`CPI`、硬件计数器等）。注意有些计数被序列化为带逗号的字符串（如 `"112,350,259"`），在解析时可能需要先移除千位分隔符再转为整数。
- `Extra Data` 是一个数组，可能用于保存额外上下文（示例值 `[262144, 32, 3.14159]`），其含义需参考产生数据的代码/文档以获得准确解释。

**使用建议**

- 直接查看：将 JSON 文件在 Chrome/Chromium 中打开（地址栏访问 `chrome://tracing`，选择 `Load` 并加载该 JSON）可以获得可视化的时间线视图。 
- 程序解析：若只需提取数值，可用 Python 快速加载并遍历 `traceEvents`：

```python
import json

with open('ddzz/tools/profile_advance/perf_dump.json', 'r') as f:
    data = json.load(f)

for ev in data.get('traceEvents', []):
    print(ev.get('name'), ev.get('ph'), ev.get('ts'), ev.get('dur'), ev.get('args'))
```

- 单位注意：文件示例的 `ts`/`dur` 值最合理的解释是“微秒 (µs)”。如果你在其他工具中看到以毫秒为单位的值，请先确认来源工具的单位约定。

**附注**

- 此说明基于提供的 JSON 片段和常见 Trace Event 约定。若需要对 `args` 中特定字段（例如 `Extra Data` 或自定义计数器）提供精确语义，请提供生成该文件的代码片段或文档，我可以进一步补充说明或自动抽取字段说明。

---

文件已生成：`ddzz/tools/profile_advance/perf_dump.md`
