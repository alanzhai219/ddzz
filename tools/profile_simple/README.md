# profile_simple

Single-header, zero-dependency Linux `perf_event` wrapper for micro-benchmarking C/C++ code.

## What it does

`SimplePerf` reads **hardware performance counters** (cycles, instructions, cache misses, …) via the Linux `perf_event_open` syscall — the same kernel interface that powers `perf stat`. No libraries, no root, no sampling overhead. Just counter reads around the code you care about.

## Quick start

```bash
g++ -O2 -std=c++11 -o example example.cpp
./example
```

> If you get "perf_event_open failed", run:  
> `echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid`

## API overview

### 1. Include

```cpp
#include "simple_perf.h"
```

### 2. Create

```cpp
SimplePerf perf;                           // default: cycles + instructions
SimplePerf perf({{"CPU_CYCLES", 0},        // or pick your own set
                 {"INSTRUCTIONS", 0},
                 {"CACHE_MISSES", 0}});
```

### 3. Measure

```cpp
perf.start();
my_function();
auto& evs = perf.stop();   // map<string, uint64_t>

printf("cycles = %lu\n", (unsigned long)evs.at("CPU_CYCLES"));
```

### 4. Or use the one-liner helper

```cpp
perf.profile(
    [&]() { my_function(); },
    "label",
    /* n_repeats  */ 5,
    /* loop_count */ 1,
    /* ops        */ 2e9,   // total FLOPs (0 = skip)
    /* bytes      */ 4e8    // total bytes  (0 = skip)
);
```

Prints a coloured summary per repeat:

```
[   label] 1/5 123456(ns) CPU_CYCLES/INSTRUCTIONS/(per-iter): 456789.0 123456.0 CPI: 3.70 2.41(GHz) 16.2(GOP/s) 3.2(GB/s) 5.0(OP/B)
```

---

## Design

### Architecture

```
┌──────────────────────────────────────────────┐
│                  SimplePerf                  │
│                                              │
│  ┌─────────┐ ┌─────────┐      ┌─────────┐   │
│  │ Event 0 │ │ Event 1 │ ...  │ Event N │   │
│  │(leader) │ │(member) │      │(member) │   │
│  │  fd[0]  │ │  fd[1]  │      │  fd[N]  │   │
│  └────┬────┘ └────┬────┘      └────┬────┘   │
│       │           │                │         │
│       └───────────┴─── group ──────┘         │
│                   │                          │
│            perf_event_open()                 │
│            ioctl(RESET/ENABLE/DISABLE)       │
│            read() → ReadBuf                  │
│                                              │
│  start()  ──▶ reset+enable all, record t0    │
│  stop()   ──▶ read clock, disable, read buf  │
│              returns map<string,uint64_t>     │
└──────────────────────────────────────────────┘
```

### Key concepts

| Concept | Explanation |
|---|---|
| **Event group** | All counters share one group leader (`fd[0]`). A single `ioctl` on the leader resets/enables/disables the whole group atomically, so counters stay in sync. |
| **`perf_event_open` syscall** | Kernel interface to create a counter file descriptor. Parameters: event type+config, pid (0 = self), cpu (-1 = any), group_fd, flags. |
| **`read_format`** | `PERF_FORMAT_GROUP \| PERF_FORMAT_ID` — one `read()` on the leader returns all counter values with their IDs, in a single struct. |
| **`CLOCK_MONOTONIC_RAW`** | Wall-clock timestamp unaffected by NTP adjustments, used to compute elapsed nanoseconds. |
| **`exclude_kernel` / `exclude_hv`** | Only count events in user-space, avoiding noise from kernel and hypervisor execution. |

### Event mnemonics

| Mnemonic | `perf_event_attr` mapping |
|---|---|
| `"CPU_CYCLES"` / `"C"` | `PERF_TYPE_HARDWARE`, `PERF_COUNT_HW_CPU_CYCLES` |
| `"INSTRUCTIONS"` / `"I"` | `PERF_TYPE_HARDWARE`, `PERF_COUNT_HW_INSTRUCTIONS` |
| `"STALLED_CYCLES_FRONTEND"` | `PERF_TYPE_HARDWARE`, `PERF_COUNT_HW_STALLED_CYCLES_FRONTEND` |
| `"STALLED_CYCLES_BACKEND"` | `PERF_TYPE_HARDWARE`, `PERF_COUNT_HW_STALLED_CYCLES_BACKEND` |
| `"CACHE_MISSES"` | `PERF_TYPE_HARDWARE`, `PERF_COUNT_HW_CACHE_MISSES` |
| `"BRANCH_MISSES"` | `PERF_TYPE_HARDWARE`, `PERF_COUNT_HW_BRANCH_MISSES` |
| any other string | `PERF_TYPE_RAW`, user-supplied `raw_config` value |

### Lifecycle

```
constructor           → perf_event_open() for each event, join group
start()               → ioctl RESET + ENABLE (group-wide), record timestamp
  ... user code ...
stop()                → record timestamp, ioctl DISABLE, read() counters
                        returns map with "ns" + each event name → value
destructor            → close() all file descriptors
```

### Differences from the original `LinuxSimplePerf`

| Aspect | Original (`simple_perf.hpp`) | Standalone (`simple_perf.h`) |
|---|---|---|
| Dependencies | Requires `misc.hpp` (ASSERT macro) | Self-contained — inline assert |
| Max events | Hard-coded `TOTAL_EVENTS = 6` macro, used in destructor and read buffer | `MAX_EVENTS = 8` constant; destructor iterates `m_events` vector (safe for any count) |
| Copyability | Implicitly copyable (dangerous — double-close of fds) | Explicitly `= delete` on copy |
| `operator()` | Overloads `operator()` for profiling | Named method `profile()` for clarity |
| Repeat numbering | 0-based (`No.0/5`) | 1-based (`1/5`) |
| Extra events | — | `CACHE_MISSES`, `BRANCH_MISSES` mnemonics added |

## Prerequisites

- Linux kernel ≥ 2.6.31 (perf_event support)
- `/proc/sys/kernel/perf_event_paranoid` ≤ 2 (or 0 for full access)
- C++11 compiler (GCC, Clang)

## Files

| File | Purpose |
|---|---|
| `simple_perf.h` | Header-only library |
| `example.cpp` | Usage sample (three demos) |
| `README.md` | This file |
