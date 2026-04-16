#pragma once

#ifdef __cplusplus

// ============================================================================
// Section: Platform Includes and Low-Level Intrinsics
// System headers, perf_event_open syscall wrapper, and X86_RAW_EVENT macro.
// ============================================================================

#include <linux/perf_event.h>
#include <time.h>
#include <sched.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <fcntl.h>           /* For O_* constants */
#include <sys/stat.h>        /* For mode constants */

#if __GLIBC__ == 2 && __GLIBC_MINOR__ < 30
#define gettid() syscall(SYS_gettid)
#endif

inline int perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
	return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

#ifdef __x86_64__
#include <x86intrin.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <atomic>
#include <sys/mman.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdarg>
#include <deque>
#include <mutex>
#include <set>
#include <iomanip>
#include <functional>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>


/*
RAW HARDWARE EVENT DESCRIPTOR
       Even when an event is not available in a symbolic form within perf right now, it can be encoded in a per processor specific way.

       For instance For x86 CPUs NNN represents the raw register encoding with the layout of IA32_PERFEVTSELx MSRs (see [Intel® 64 and IA-32 Architectures Software Developer's Manual Volume 3B: System Programming Guide] Figure 30-1
       Layout of IA32_PERFEVTSELx MSRs) or AMD's PerfEvtSeln (see [AMD64 Architecture Programmer's Manual Volume 2: System Programming], Page 344, Figure 13-7 Performance Event-Select Register (PerfEvtSeln)).

       Note: Only the following bit fields can be set in x86 counter registers: event, umask, edge, inv, cmask. Esp. guest/host only and OS/user mode flags must be setup using EVENT MODIFIERS.

 event 7:0
 umask 15:8
 edge  18
 inv   23
 cmask 31:24
*/
#define X86_RAW_EVENT(EventSel, UMask, CMask) ((CMask << 24) | (UMask << 8) | (EventSel))

namespace LinuxPerf {

// ============================================================================
// Design overview (4 core components):
//
// 1. PerfConfig     -- parses the LINUX_PERF environment variable.
// 2. PerfCounterGroup    -- per-thread HW/SW counter group with ProfileScope RAII.
// 3. CpuContextSwitchTracker -- per-CPU context-switch ring buffer for timeline view.
// 4. TraceFileWriter       -- aggregates all dumpers, writes a single perf_dump.json.
// ============================================================================

// ============================================================================
// Section: Logging Utilities
// Diagnostic printf/perror helpers used throughout the library.
// ============================================================================

// Two-level stringize: #x stringizes literally without macro expansion,
// so STRINGIFY(EXPAND(__LINE__)) is needed to get "42" instead of "__LINE__".
#define STRINGIFY_LITERAL(x) #x
#define STRINGIFY_EXPAND(x) STRINGIFY_LITERAL(x)

#define LINUX_PERF_ "\e[33m[LINUX_PERF:" STRINGIFY_EXPAND(__LINE__) "]\e[0m "

inline void log_printf(const char* format, ...) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
}

inline void log_message(const std::string& message) {
    std::cout << message;
}

inline void log_perror(const char* message) {
    perror(message);
}

inline void abort_with_perror(const char* message) {
    log_perror(message);
    abort();
}

// ============================================================================
// Section: Timestamp Infrastructure
// get_time_ns(), read_tsc(), read_pmc(), and TimestampConverter for converting
// raw monotonic timestamps (nanoseconds) to microseconds for trace output.
// ============================================================================

inline uint64_t get_time_ns() {
    struct timespec tp0;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &tp0) != 0) {
        abort_with_perror(LINUX_PERF_"clock_gettime(CLOCK_MONOTONIC_RAW,...) failed!");
    }
    return (tp0.tv_sec * 1000000000) + tp0.tv_nsec;
}

#ifdef __x86_64__
inline uint64_t read_tsc(void) {
    return __rdtsc();
}
inline uint64_t read_pmc(int index) {
    return __rdpmc(index);
}
#endif

#ifdef __aarch64__
// SPDX-License-Identifier: GPL-2.0
inline uint64_t read_tsc(void) {
    uint64_t val;
    /*
     * According to ARM DDI 0487F.c, from Armv8.0 to Armv8.5 inclusive, the
     * system counter is at least 56 bits wide; from Armv8.6, the counter
     * must be 64 bits wide.  So the system counter could be less than 64
     * bits wide and it is attributed with the flag 'cap_user_time_short'
     * is true.
     */
    asm volatile("mrs %0, cntvct_el0" : "=r" (val));
    return val;
}
inline uint64_t read_pmc(int index) {
    (void)index;
    uint64_t val;
    asm volatile("mrs %0, PMCCNTR_EL0" : "=r"(val));
    return val;
}
#endif

// Converts raw CLOCK_MONOTONIC_RAW nanosecond timestamps to microseconds
// relative to a base point (captured at construction time).
// Despite the legacy name "tsc", this now uses clock_gettime directly,
// avoiding the ~1s calibration sleep that a real TSC approach would need.
struct TimestampConverter {
    uint64_t ticks_per_second;
    uint64_t base_tick;

    // Convert an absolute timestamp to microseconds relative to base.
    double ticks_to_usec(uint64_t ticks) const {
        if (ticks < base_tick) {
            return 0;
        }
        return (ticks - base_tick) * 1000000.0 / ticks_per_second;
    }

    // Convert a [start, end] timestamp pair to a duration in microseconds.
    double duration_usec(uint64_t start_ticks, uint64_t end_ticks) const {
        if (end_ticks < start_ticks) {
            return 0;
        }
        return (end_ticks - start_ticks) * 1000000.0 / ticks_per_second;
    }

    TimestampConverter() {
        // Use CLOCK_MONOTONIC_RAW as unified time source (unit: nanoseconds).
        // This avoids the ~1s calibration sleep that a TSC-based approach would need,
        // and stays consistent with the clockid used in perf_event_attr.
        ticks_per_second = 1000000000; // ns
        base_tick = get_time_ns();
    }
};

// ============================================================================
// Section: Trace Output -- Dumper Interface and JSON Writer
// ITraceEventDumper interface, TraceFileWriter singleton (Chrome Trace format
// output), and CrossLibrarySingleton for shared-memory based process-wide
// singleton that survives across multiple .so boundaries.
// ============================================================================

// Interface that any component must implement to contribute trace events
// to the final perf_dump.json output file.
class ITraceEventDumper {
public:
    virtual ~ITraceEventDumper() = default;
    virtual void dump_json(std::ofstream& fw, TimestampConverter& tsc) = 0;
};

// Aggregates all registered ITraceEventDumper instances and writes a single
// perf_dump.json in Chrome Trace Event Format (viewable in chrome://tracing
// or Perfetto).  Uses a cross-library singleton to ensure exactly one writer
// per process, even when this header is compiled into multiple shared libraries.
struct TraceFileWriter {
    std::mutex writer_mutex;
    std::set<ITraceEventDumper*> registered_dumpers;
    const char* output_filename = "perf_dump.json";
    bool needs_finalization = true;
    std::ofstream output_stream;
    std::atomic_int registered_dumper_count{0};
    TimestampConverter timestamp_converter;

    ~TraceFileWriter() {
        if (needs_finalization) {
            flush_and_close();
        }
    }

    // Write all registered dumpers' data to the output JSON file, then close it.
    void flush_and_close() {
        if (!needs_finalization) {
            return;
        }
        std::lock_guard<std::mutex> guard(writer_mutex);
        // Re-check under lock to prevent double finalization from concurrent threads.
        if (!needs_finalization || registered_dumpers.empty()) {
            return;
        }

        // start dump
        output_stream.open(output_filename, std::ios::out);
        output_stream << "{\n";
        output_stream << "\"schemaVersion\": 1,\n";
        output_stream << "\"traceEvents\": [\n";
        output_stream.flush();

        for (auto& pthis : registered_dumpers) {
            pthis->dump_json(output_stream, timestamp_converter);
        }
        registered_dumpers.clear();

        output_stream << R"({
            "name": "Profiler End",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Trace OV Profiler",
            "ts":)"
           << timestamp_converter.ticks_to_usec(get_time_ns()) << "}";
        output_stream << "]\n";
        output_stream << "}\n";
        auto total_size = output_stream.tellp();
        output_stream.close();
        needs_finalization = false;

        std::cout << LINUX_PERF_"Dumped ";

        if (total_size < 1024) {
            std::cout << total_size << " bytes ";
        } else if (total_size < 1024*1024) {
            std::cout << total_size/1024 << " KB ";
        } else {
            std::cout << total_size/(1024 * 1024) << " MB ";
        }
        std::cout << " to " << output_filename << std::endl;
    }

    int register_dumper(ITraceEventDumper* pthis) {
        std::lock_guard<std::mutex> guard(writer_mutex);
        std::ostringstream ss;
        auto serial_id = registered_dumper_count.fetch_add(1);
        ss << LINUX_PERF_"#" << serial_id << "(" << pthis << ") : is registered." << std::endl;
        log_message(ss.str());
        registered_dumpers.emplace(pthis);
        return serial_id;
    }

    // Cross-library singleton via POSIX shared memory.
    //
    // Problem: a C++ static-local singleton in a header-only library gets a
    // separate instance in each .so that includes the header.  We need exactly
    // one TraceFileWriter per process.
    //
    // Solution: use shm_open to create a process-wide shared memory region that
    // stores a pointer to the single heap-allocated instance.
    //
    // Protocol:
    //   1. All users in the same process shm_open the same named region.
    //   2. The first to CAS init_lock_pid owns initialization (creates the object).
    //   3. Others spin on init_done_pid until the owner signals completion.
    //   4. Reference counting via reference_count controls destruction.
    template<class T>
    struct CrossLibrarySingleton {
        static constexpr const char * shm_name = "/linuxperf_shm01";

        // Layout of the shared memory region (fits in one page).
        struct SharedMemoryBlock {
            T * instance_ptr;
            int64_t reference_count;
            pid_t init_lock_pid;    // PID that claimed the initialization lock
            pid_t init_done_pid;    // set to PID when initialization is complete
        };

        SharedMemoryBlock* shared_block;
        T * instance_ptr;

        CrossLibrarySingleton() {
            int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
            if (ftruncate(fd, 4096) != 0) {
                abort_with_perror("ftruncate failed!");
            }
            pid_t pid = getpid();
            shared_block = reinterpret_cast<SharedMemoryBlock*>(mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
            if (__atomic_exchange_n(&shared_block->init_lock_pid, pid, __ATOMIC_SEQ_CST) != pid) {
                // we are the first one that set init_lock_pid to pid, do initialization
                shared_block->instance_ptr = new T();
                __atomic_store_n(&shared_block->reference_count, 0, __ATOMIC_SEQ_CST);
                __atomic_store_n(&shared_block->init_done_pid, pid, __ATOMIC_SEQ_CST);
            } else {
                // spin until the owner has done
                while (__atomic_load_n(&shared_block->init_done_pid, __ATOMIC_SEQ_CST) != pid);
            }
            __atomic_add_fetch(&shared_block->reference_count, 1, __ATOMIC_SEQ_CST);
            close(fd);

            instance_ptr = shared_block->instance_ptr;
        }
        T& obj() {
            return *(instance_ptr);
        }
        ~CrossLibrarySingleton() {
            if (__atomic_sub_fetch(&shared_block->reference_count, 1, __ATOMIC_SEQ_CST) == 0) {
                delete shared_block->instance_ptr;
                munmap(shared_block, 4096);
            }
            shm_unlink(shm_name);
        }
    };

    static TraceFileWriter& get() {
        static CrossLibrarySingleton<TraceFileWriter> inst;
        return inst.obj();
    }
};

// ============================================================================
// Section: String Utilities
// ============================================================================

inline std::vector<std::string> str_split(const std::string& s, std::string delimiter) {
    std::vector<std::string> ret;
    if (delimiter.empty()) {
        ret.push_back(s);
        return ret;
    }
    size_t last = 0;
    size_t next = 0;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        ret.push_back(s.substr(last, next-last));
        last = next + delimiter.size();
    }
    ret.push_back(s.substr(last));
    return ret;
}

// Parse a list of 1-3 hex/decimal values into an X86_RAW_EVENT config.
// Accepts any single-character delimiter (e.g. "-" from env var, "," from EventSpec).
inline uint64_t parse_raw_event_values(const std::vector<std::string>& parts) {
    if (parts.empty()) { return 0; }
    auto evsel = std::strtoull(parts[0].c_str(), nullptr, 0);
    if (parts.size() == 1) { return evsel; }
    auto umask = std::strtoull(parts[1].c_str(), nullptr, 0);
    uint64_t cmask = 0;
    if (parts.size() >= 3) {
        cmask = std::strtoull(parts[2].c_str(), nullptr, 0);
    }
    return X86_RAW_EVENT(evsel, umask, cmask);
}

// ============================================================================
// Section: perf_event_attr Helpers and Ring Buffer Primitives
// Factory for perf_event_attr, RAII try-lock, ring buffer read/release helpers.
// ============================================================================

inline perf_event_attr make_perf_event_attr(uint32_t type,
                                            uint64_t config,
                                            bool exclude_kernel,
                                            bool include_total_time,
                                            bool pinned) {
    perf_event_attr pea;
    memset(&pea, 0, sizeof(perf_event_attr));
    pea.type = type;
    pea.size = sizeof(perf_event_attr);
    pea.config = config;
    pea.disabled = 1;
    pea.exclude_kernel = exclude_kernel ? 1 : 0;
    pea.exclude_hv = 1;
    pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
    if (include_total_time) {
        pea.read_format |= PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    }
    if (pinned) {
        pea.pinned = 1;
    }
    return pea;
}

// RAII guard that performs a single atomic try-lock (not a spin).
// If the lock was already held, owns_lock() returns false and the
// destructor is a no-op.
class AtomicTryLockGuard {
public:
    explicit AtomicTryLockGuard(std::atomic<int>& lock) : lock_(lock), owns_(lock_.exchange(1) == 0) {}
    ~AtomicTryLockGuard() {
        if (owns_) {
            lock_.store(0);
        }
    }
    bool owns_lock() const { return owns_; }

private:
    std::atomic<int>& lock_;
    bool owns_;
};

// Read a value of type T from the perf ring buffer at the given offset,
// handling wrap-around via (offset % data_size).  Advances offset by sizeof(T).
template<typename T>
T& read_ring_buffer(perf_event_mmap_page& meta, uint64_t& offset) {
    auto offset0 = offset;
    offset += sizeof(T);
    return *reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(&meta) + meta.data_offset + (offset0)%meta.data_size);
}

template<typename Event>
void release_perf_event_resource(Event& ev, size_t mmap_length) {
    if (ev.pmeta && ev.pmeta != MAP_FAILED) {
        munmap(ev.pmeta, mmap_length);
        ev.pmeta = nullptr;
    }
    if (ev.fd >= 0) {
        close(ev.fd);
        ev.fd = -1;
    }
}

template<typename EventContainer>
void release_perf_event_resources(EventContainer& events, size_t mmap_length) {
    for (auto& ev : events) {
        release_perf_event_resource(ev, mmap_length);
    }
}

inline perf_event_mmap_page* map_perf_event_metadata_or_abort(int fd, size_t mmap_length, const char* error_message) {
    auto* pmeta = reinterpret_cast<perf_event_mmap_page*>(
            mmap(NULL, mmap_length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
    if (pmeta == MAP_FAILED) {
        close(fd);
        abort_with_perror(error_message);
    }
    return pmeta;
}

// Cursor for sequentially reading fields from a perf ring buffer.
// The offset increases monotonically; physical wrap-around is handled
// internally by read_ring_buffer via (offset % data_size).
struct RingBufferReader {
    perf_event_mmap_page& meta;
    uint64_t offset;

    RingBufferReader(perf_event_mmap_page& meta, uint64_t offset) : meta(meta), offset(offset) {}

    template<typename T>
    T read() {
        return read_ring_buffer<T>(meta, offset);
    }
};

// ============================================================================
// Section: Context Switch Record Parsing
// Deserializes a single PERF_RECORD_SWITCH_CPU_WIDE entry from the ring buffer.
// ============================================================================

struct ContextSwitchRecord {
    uint32_t type = 0;
    uint16_t misc = 0;
    uint16_t size = 0;
    uint32_t next_prev_pid = 0;
    uint32_t next_prev_tid = 0;
    uint32_t pid = 0;
    uint32_t tid = 0;
    uint64_t time = 0;
    uint32_t cpu = 0;
    uint32_t reserved0 = 0;

    static ContextSwitchRecord parse(RingBufferReader& cursor) {
        ContextSwitchRecord record;
        record.type = cursor.read<__u32>();
        record.misc = cursor.read<__u16>();
        record.size = cursor.read<__u16>();
        if (record.type == PERF_RECORD_SWITCH_CPU_WIDE) {
            record.next_prev_pid = cursor.read<__u32>();
            record.next_prev_tid = cursor.read<__u32>();
        }
        record.pid = cursor.read<__u32>();
        record.tid = cursor.read<__u32>();
        record.time = cursor.read<uint64_t>();
        record.cpu = cursor.read<__u32>();
        record.reserved0 = cursor.read<__u32>();
        return record;
    }
};

// ============================================================================
// Section: Type-Erased Extra Data Storage
// Allows attaching arbitrary typed arguments (int, float, double, pointer,
// vector) to trace events.  Each value is tagged with a single char
// ('i', 'f', 'p', 'v') and serialized into the Chrome Trace "args" JSON.
// ============================================================================

template<int Size>
struct TypeErasedArgStorage {
    union TaggedValue {
        double f;
        int64_t i;
        void* p;
    };
    TaggedValue values[Size];
    char types[Size];       // tag per slot: 'i'=int, 'f'=float, 'p'=pointer, 'v'=vector, '\0'=end

    TypeErasedArgStorage() : types() {}

    // Compile-time type tag deduction.
    template<typename T>
    static char get_type(T) {
        return std::is_pointer<T>::value   ? 'p'
             : std::is_floating_point<T>::value ? 'f'
             : std::is_integral<T>::value       ? 'i'
             : '\0';
    }

    template<typename T>
    void set_data(int& index, T* value) {
        types[index] = get_type(value);
        values[index].p = value;
        index++;
    }

    void set_data(int& index, float value) {
        types[index] = get_type(value);
        values[index].f = value;
        index++;
    }

    void set_data(int& index, double value) {
        types[index] = get_type(value);
        values[index].f = value;
        index++;
    }

    template<typename T>
    void set_data(int& index, T value) {
        static_assert(std::is_integral<T>::value, "extra data only supports integral, floating, pointer, or vector types");
        types[index] = get_type(value);
        values[index].i = value;
        index++;
    }

    template<typename T>
    void set_data(int& index, const std::vector<T>& value) {
        static_assert(std::is_integral<T>::value || std::is_floating_point<T>::value,
                "vector extra data only supports integral or floating point values");
        types[index] = 'v';
        values[index].i = value.size();
        index++;
        for (auto& item : value) {
            set_data(index, item);
        }
    }

    void set_terminator(int index) {
        types[index] = '\0';
    }

    template <typename ... Values>
    void set_datas(Values... vals) {
        static_assert(Size >= sizeof...(vals), "too many extra data values");
        int index = 0;
        int unused[] = {0, (set_data(index, vals), 0)...};
        (void)unused;
        set_terminator(index);
    }

    void append_json(std::ostream& os) const {
        if (types[0] == 0) {
            return;
        }

        const char* sep = "";
        os << ",\"Extra Data\":[";
        for (size_t i = 0; i < Size && types[i] != 0; i++) {
            if (types[i] == 'f') {
                os << sep << values[i].f;
            } else if (types[i] == 'i') {
                os << sep << values[i].i;
            } else if (types[i] == 'p') {
                os << sep << "\"" << values[i].p << "\"";
            } else if (types[i] == 'v') {
                const auto vec_size = values[i].i;
                os << sep << "\"(";
                const char* sep2 = "";
                i++;
                const size_t i_end = i + vec_size;
                for (; i < i_end; i++) {
                    if (types[i] == 'f') {
                        os << sep2 << values[i].f;
                    }
                    if (types[i] == 'i') {
                        os << sep2 << values[i].i;
                    }
                    sep2 = ",";
                }
                i--;
                os << ")\"";
            } else {
                os << sep << "\"?\"";
            }
            sep = ",";
        }
        os << "]";
    }
};

// ============================================================================
// Section: Environment Variable Configuration
// Parses the LINUX_PERF environment variable into dump limits, CPU masks,
// raw PMU event configs, and context-switch mode.
//
// Format:  LINUX_PERF=opt1:opt2:...
//   dump           -- enable JSON dump (unlimited)
//   dump=N         -- limit to N dumps per thread
//   cpus=C1,C2,... -- restrict dump/context-switch to listed CPUs
//   switch-cpu     -- enable per-CPU context-switch timeline
//   NAME=0xCONFIG  -- add a raw PMU counter (hex or EventSel-UMask-CMask)
// ============================================================================

struct PerfConfig {
    // --- data members ---
    int64_t dump_limit = 0;                                     // 0 = dump disabled
    cpu_set_t cpu_affinity_mask;
    bool context_switch_enabled = false;
    std::vector<std::pair<std::string, uint64_t>> custom_pmu_events;

    // Parse a hex or dash-separated PMU event config string.
    // Formats: "0x10d1" or "EventSel-UMask-CMask"
    uint64_t parse_pmu_event_hex(const std::string& raw_evt) const {
        auto delimiter = (raw_evt.find('-') != std::string::npos) ? "-" : "";
        auto parts = delimiter[0] ? str_split(raw_evt, delimiter) : std::vector<std::string>{raw_evt};
        return parse_raw_event_values(parts);
    }

    void parse_cpu_list(const std::string& value) {
        // cpus=56 or cpus=56,57,59
        auto cpus = str_split(value, ",");
        CPU_ZERO(&cpu_affinity_mask);
        for (auto& cpu : cpus) {
            CPU_SET(std::atoi(cpu.c_str()), &cpu_affinity_mask);
        }
    }

    void enable_context_switch_tracking() {
        // get cpu_affinity_mask as early as possible
        context_switch_enabled = true;
        CPU_ZERO(&cpu_affinity_mask);
        if (sched_getaffinity(getpid(), sizeof(cpu_set_t), &cpu_affinity_mask)) {
            abort_with_perror(LINUX_PERF_"sched_getaffinity failed:");
        }
    }

    void parse_option(const std::string& opt) {
        auto items = str_split(opt, "=");
        if (items.size() == 2) {
            const std::string& key = items[0];
            const std::string& value = items[1];
            if (key == "dump") {
                // limit the number of dumps per thread
                dump_limit = strtoll(value.c_str(), nullptr, 0);
            } else if (key == "cpus") {
                // thread affinity can be changed by runtime libs; allow explicit cpu list.
                parse_cpu_list(value);
            } else {
                auto config = parse_pmu_event_hex(value);
                if (config > 0) {
                    custom_pmu_events.emplace_back(key, config);
                }
            }
            return;
        }

        if (items.size() == 1) {
            if (items[0] == "switch-cpu") {
                enable_context_switch_tracking();
            }
            if (items[0] == "dump") {
                dump_limit = std::numeric_limits<int64_t>::max(); // no limit to number of dumps
            }
        }
    }

    void log_parsed_config() const {
        for (auto& cfg : custom_pmu_events) {
            log_printf(LINUX_PERF_" config: %s=0x%lx\n", cfg.first.c_str(), cfg.second);
        }
        if (context_switch_enabled) {
            log_printf(LINUX_PERF_" config: switch_cpu\n");
        }
        if (dump_limit) {
            log_printf(LINUX_PERF_" config: dump=%ld\n", dump_limit);
        }
        if (CPU_COUNT(&cpu_affinity_mask)) {
            std::ostringstream ss;
            ss << LINUX_PERF_ << " config: cpus=";
            for (int cpu = 0; cpu < (int)sizeof(cpu_set_t) * 8; cpu++) {
                if (CPU_ISSET(cpu, &cpu_affinity_mask)) {
                    ss << cpu << ",";
                }
            }
            ss << std::endl;
            log_message(ss.str());
        }
    }

    PerfConfig() {
        CPU_ZERO(&cpu_affinity_mask);
        // env var defined raw events
        const char* str_raw_config = std::getenv("LINUX_PERF");
        if (!str_raw_config) {
            log_printf(LINUX_PERF_" LINUX_PERF is unset, example: LINUX_PERF=dump:switch-cpu:L2_MISS=0x10d1\n");
            return;
        }

        // options are separated by ':' as PATH
        auto options = str_split(str_raw_config, ":");
        for (auto& opt : options) {
            parse_option(opt);
        }
        log_parsed_config();
    }

    bool should_dump_on_cpu(int cpu) {
        if (dump_limit == 0) {
            return false;
        }
        if (CPU_COUNT(&cpu_affinity_mask)) {
            return CPU_ISSET(cpu, &cpu_affinity_mask);
        }
        return true;
    }

    static PerfConfig& get() {
        static PerfConfig inst;
        return inst;
    }
};

// ============================================================================
// Section: Per-CPU Context Switch Timeline
// Tracks CPU-wide context switches to build a timeline showing which thread
// was running on which CPU at any point.  Uses PERF_RECORD_SWITCH_CPU_WIDE
// records delivered via a memory-mapped ring buffer (one per monitored CPU).
// The resulting time slices appear in the JSON output as duration events on
// virtual "CPU<N>" threads, enabling visual correlation with per-thread scopes.
// ============================================================================

struct CpuContextSwitchTracker : public ITraceEventDumper {
    bool tracking_enabled;

    // State for one monitored CPU's perf event fd and ring buffer.
    struct PerCpuState {
        int fd;
        perf_event_mmap_page * pmeta;
        int cpu;
        uint64_t switch_in_timestamp;       // when the current TID switched in
        uint64_t switch_in_tid;             // which TID switched in
        uint64_t last_event_timestamp;      // timestamp of the last processed record

        PerCpuState(int fd, perf_event_mmap_page * pmeta): fd(fd), pmeta(pmeta) {}
    };
    std::vector<PerCpuState> per_cpu_states;

    static perf_event_attr build_context_switch_attr() {
        perf_event_attr pea = make_perf_event_attr(PERF_TYPE_HARDWARE,
                                                   PERF_COUNT_HW_REF_CPU_CYCLES,
                                                   true,
                                                   true,
                                                   true);
        pea.disabled = 0;

        // Generate PERF_RECORD_SWITCH records into ring-buffer for timeline view.
        pea.context_switch = 1;
        pea.sample_id_all = 1;
        pea.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_TID | PERF_SAMPLE_CPU;
        pea.use_clockid = 1;
        pea.clockid = CLOCK_MONOTONIC_RAW;
        return pea;
    }

    static size_t ring_buffer_mmap_size() {
        return sysconf(_SC_PAGESIZE) * (1024 + 1);
    }

    void open_cpu_monitor(int cpu) {
        perf_event_attr pea = build_context_switch_attr();
        const size_t mmap_length = ring_buffer_mmap_size();

        // measures all processes/threads on the specified CPU (pid=-1 = CPU-wide)
        const int ctx_switch_fd = perf_event_open(&pea, -1, cpu, -1, 0);
        if (ctx_switch_fd < 0) {
            abort_with_perror(LINUX_PERF_"CpuContextSwitchTracker perf_event_open failed (check /proc/sys/kernel/perf_event_paranoid please)");
        }

        auto* ctx_switch_pmeta = map_perf_event_metadata_or_abort(
                ctx_switch_fd, mmap_length, LINUX_PERF_"mmap perf_event_mmap_page failed:");

        log_printf(LINUX_PERF_"perf_event_open CPU_WIDE context_switch on cpu %d, ctx_switch_fd=%d\n", cpu, ctx_switch_fd);
        per_cpu_states.emplace_back(ctx_switch_fd, ctx_switch_pmeta);
        per_cpu_states.back().switch_in_timestamp = get_time_ns();
        per_cpu_states.back().last_event_timestamp = get_time_ns();
        per_cpu_states.back().cpu = cpu;
    }

    bool should_enable(const cpu_set_t& mask) {
        const long number_of_processors = sysconf(_SC_NPROCESSORS_ONLN);
        log_printf(LINUX_PERF_"sizeof(cpu_set_t):%lu: _SC_NPROCESSORS_ONLN=%ld CPU_COUNT=%d\n",
                   sizeof(cpu_set_t), number_of_processors, CPU_COUNT(&mask));
        if (CPU_COUNT(&mask) >= number_of_processors) {
            log_printf(LINUX_PERF_" no affinity is set, will not enable CpuContextSwitchTracker\n");
            return false;
        }
        return true;
    }

    void open_monitors_for_cpus(const cpu_set_t& mask) {
        for (int cpu = 0; cpu < (int)sizeof(cpu_set_t) * 8; cpu++) {
            if (!CPU_ISSET(cpu, &mask)) {
                continue;
            }
            open_cpu_monitor(cpu);
        }
    }

    // Flush any currently-running (switch-in but no switch-out yet) slices
    // by using "now" as a temporary end point.  Called at dump time to avoid
    // losing the final in-progress slice.
    void flush_active_slices() {
        for (auto& ev : per_cpu_states) {
            if (ev.switch_in_timestamp == 0) {
                continue;
            }
            recorded_slices.emplace_back();
            auto* pd = &recorded_slices.back();
            pd->tid = ev.switch_in_tid;
            pd->cpu = ev.cpu;
            pd->tsc_start = ev.switch_in_timestamp;
            pd->tsc_end = get_time_ns();
            ev.switch_in_timestamp = 0;
        }
    }

    // Check whether any ring buffer is more than half full.
    // Only drain when this returns true, to reduce profiling overhead.
    bool is_ring_buffer_half_full() const {
        for (auto& ev : per_cpu_states) {
            const auto& mmap_meta = *ev.pmeta;
            const auto used_size = (mmap_meta.data_head - mmap_meta.data_tail) % mmap_meta.data_size;
            if (used_size > (mmap_meta.data_size >> 1)) {
                return true;
            }
        }
        return false;
    }

    void log_record_type_constants() const {
        log_printf("PERF_RECORD_SWITCH = %d\n", PERF_RECORD_SWITCH);
        log_printf("PERF_RECORD_SWITCH_CPU_WIDE = %d\n", PERF_RECORD_SWITCH_CPU_WIDE);
        log_printf("PERF_RECORD_MISC_SWITCH_OUT = %d\n", PERF_RECORD_MISC_SWITCH_OUT);
        log_printf("PERF_RECORD_MISC_SWITCH_OUT_PREEMPT  = %d\n", PERF_RECORD_MISC_SWITCH_OUT_PREEMPT);
    }

    // Record a completed time slice: [switch_in_timestamp, switch_out_time].
    void record_switch_out(PerCpuState& ev, uint32_t tid, uint32_t cpu, uint64_t time, __u16 misc) {
        recorded_slices.emplace_back();
        auto* pd = &recorded_slices.back();
        pd->tid = tid;
        pd->cpu = cpu;
        pd->preempt = (misc & PERF_RECORD_MISC_SWITCH_OUT_PREEMPT);
        pd->tsc_start = ev.switch_in_timestamp;
        pd->tsc_end = time;

        if (verbose_logging) {
            log_printf("\t  cpu: %u tid: %u  %lu (+%lu)\n", cpu, tid, ev.switch_in_timestamp, time - ev.switch_in_timestamp);
        }
        ev.switch_in_timestamp = 0;
    }

    // Parse one record from the ring buffer and update the switch-in/out state machine.
    void parse_one_record(PerCpuState& ev, perf_event_mmap_page& mmap_meta, uint64_t& head0, uint64_t head1) {
        const auto h0 = head0;
        RingBufferReader cursor(mmap_meta, head0);
        const ContextSwitchRecord record = ContextSwitchRecord::parse(cursor);
        (void)record.reserved0;
        (void)record.next_prev_pid;
        (void)record.pid;

        if (record.tid > 0 && verbose_logging) {
            log_printf("event: %lu/%lu\ttype,misc,size=(%u,%u,%u) cpu%u,next_prev_tid=%u,tid=%u  time:(%lu), (+%lu)\n",
                       h0, head1, record.type, record.misc, record.size, record.cpu,
                       record.next_prev_tid, record.tid, record.time, record.time - ev.last_event_timestamp);
        }

        // State machine:
        //   switch-in  → save timestamp and tid
        //   switch-out → form [start, end] time slice
        if (record.type == PERF_RECORD_SWITCH_CPU_WIDE && record.tid > 0) {
            if (record.misc & PERF_RECORD_MISC_SWITCH_OUT || record.misc & PERF_RECORD_MISC_SWITCH_OUT_PREEMPT) {
                record_switch_out(ev, record.tid, record.cpu, record.time, record.misc);
            } else {
                ev.switch_in_timestamp = record.time;
                ev.switch_in_tid = record.tid;
            }
        }

        ev.last_event_timestamp = record.time;
        head0 = h0 + record.size;
    }

    // Consume all available records from one CPU's ring buffer.
    // After draining, advances data_tail so the kernel can reuse the space.
    void drain_ring_buffer(PerCpuState& ev) {
        auto& mmap_meta = *ev.pmeta;
        uint64_t head0 = mmap_meta.data_tail;
        const uint64_t head1 = mmap_meta.data_head;

        if (head0 == head1) {
            return;
        }
        if (verbose_logging) {
            log_record_type_constants();
        }

        while (head0 < head1) {
            parse_one_record(ev, mmap_meta, head0, head1);
        }

        if (head0 != head1) {
            log_printf("head0(%lu) != head1(%lu)\n", head0, head1);
            abort_with_perror("ring buffer head mismatch");
        }

        // update tail so kernel can keep generate event records
        mmap_meta.data_tail = head0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    CpuContextSwitchTracker() {
        tracking_enabled = PerfConfig::get().context_switch_enabled;
        if (tracking_enabled) {
            // make sure TimestampConverter in TraceFileWriter is the very first thing to initialize
            TraceFileWriter::get().register_dumper(this);

            // open fd for each CPU
            const cpu_set_t mask = PerfConfig::get().cpu_affinity_mask;
            if (!should_enable(mask)) {
                tracking_enabled = false;
                return;
            }

            open_monitors_for_cpus(mask);
            my_pid = getpid();
            my_tid = gettid();
        }
    }

    ~CpuContextSwitchTracker() {
        if (tracking_enabled) {
            TraceFileWriter::get().flush_and_close();
        }
        release_perf_event_resources(per_cpu_states, ring_buffer_mmap_size());
    }

    // A time slice where a particular TID was running on a particular CPU.
    struct TimeSlice {
        uint64_t tsc_start;
        uint64_t tsc_end;
        uint32_t tid;
        uint32_t cpu;
        bool preempt;   // preempt means current TID preempts previous thread
    };

    std::deque<TimeSlice> recorded_slices;

    void dump_json(std::ofstream& fw, TimestampConverter& tsc) override {
        if (!tracking_enabled) {
            return;
        }

        drain_all_ring_buffers();

        auto data_size = recorded_slices.size();
        if (!data_size) {
            return;
        }

        flush_active_slices();

        auto pid = 9999;    // fake pid for CPU
        auto cat = "TID";

        // TID is used for CPU id instead
        for (auto& d : recorded_slices) {
            auto duration = tsc.duration_usec(d.tsc_start, d.tsc_end);
            auto start = tsc.ticks_to_usec(d.tsc_start);
            auto cpu_id = d.cpu;

            fw << "{\"ph\": \"X\", \"name\": \"" << d.tid << "\", \"cat\":\"" << cat << "\","
                << "\"pid\": " << pid << ", \"tid\": \"CPU" << cpu_id <<  "\","
                << "\"ts\": " << std::setprecision (15) << start << ", \"dur\": " << duration << "},\n";
        }
    }

    bool verbose_logging = false;
    uint32_t my_pid = 0;
    uint32_t my_tid = 0;
    std::atomic<int> update_lock{0};

    // Drain all per-CPU ring buffers.  Only one thread may do this at a time
    // (guarded by atomic try-lock); others simply skip if the lock is held.
    void drain_all_ring_buffers() {
        // only one thread can update ring-buffer at one time
        AtomicTryLockGuard lock_guard(update_lock);
        if (!lock_guard.owns_lock()) {
            return;
        }

        if (!is_ring_buffer_half_full()) {
            return;
        }

        for (auto& ev : per_cpu_states) {
            drain_ring_buffer(ev);
        }
    }

    static CpuContextSwitchTracker& get() {
        static CpuContextSwitchTracker inst;
        return inst;
    }
};

// ============================================================================
// Section: Per-Thread Hardware/Software Counter Group
//
// PerfCounterGroup is the core profiling class.  It manages a perf_event
// group for the current thread and provides:
//   - Adding HW/SW/RAW counters to the group
//   - Unified enable/disable/reset
//   - Two counter read paths: fast (rdpmc userspace) and slow (read() syscall)
//   - begin_scope() / ProfileScope RAII for scoped measurements
//   - JSON trace output via ITraceEventDumper
//
// One instance per thread (thread_local in get()).
// ============================================================================

struct PerfCounterGroup : public ITraceEventDumper {
    static constexpr size_t kReadBufferU64Count = 512;

    int leader_fd = -1;                 // fd of the group leader event
    uint64_t read_format;

    // Describes a single counter within the group.
    struct CounterDescriptor {
        int fd = -1;
        uint64_t id = 0;
        uint64_t pmc_index = 0;        // >0 if rdpmc fast path is available
        perf_event_mmap_page* pmeta = nullptr;
        std::string name = "?";
        char format[32];
    };
    std::vector<CounterDescriptor> counter_descriptors;

    uint64_t read_buffer[kReadBufferU64Count]; // 4KB
    uint64_t pmc_bit_width;
    uint64_t pmc_value_mask;            // bitmask applied to rdpmc result
    uint64_t values[32];
    uint32_t tsc_time_shift;
    uint32_t tsc_time_mult;

    // Well-known event indices (into counter_descriptors vector).
    // -1 means the event is not present in this group.
    int task_clock_index = -1;
    int cpu_cycles_index = -1;
    int instructions_index = -1;

    // A snapshot of counter values for one profiled scope.
    // Contains start/end timestamps, counter deltas, and optional extra data.
    struct ScopedSnapshot {
        uint64_t tsc_start;
        uint64_t tsc_end;
        std::string title;
        std::string cat;
        int32_t id;
        static const int data_size = 16; // 4(fixed) + 8(PMU) + 4(software)
        uint64_t data[data_size] = {0};
        TypeErasedArgStorage<data_size> extra_data;

        ScopedSnapshot(std::string title, std::string cat = {})
            : title(std::move(title)), cat(std::move(cat)) {
            start();
        }
        void start() {
            tsc_start = get_time_ns();
        }
        void stop() {
            tsc_end = get_time_ns();
        }
    };

    bool json_dump_enabled = false;
    int64_t remaining_dump_quota = 0;
    std::deque<ScopedSnapshot> recorded_snapshots;
    int dumper_serial_id;

    using CustomArgsSerializer = std::function<void(std::ostream& fw, double usec, uint64_t* counters)>;
    CustomArgsSerializer custom_args_serializer;

    // --- JSON serialization helpers ---

    void emit_custom_args(std::stringstream& ss, double duration, const ScopedSnapshot& d) const {
        if (custom_args_serializer) {
            custom_args_serializer(ss, duration, const_cast<uint64_t*>(d.data));
        }
    }

    // Emit derived metrics: CPU usage, frequency, CPI.
    void emit_derived_metrics(std::stringstream& ss, double duration, const ScopedSnapshot& d) const {
        if (task_clock_index >= 0) {
            // PERF_COUNT_SW_TASK_CLOCK is in nano-seconds.
            ss << "\"CPU Usage\":" << (d.data[task_clock_index] * 1e-3) / duration << ",";
        }
        if (cpu_cycles_index >= 0) {
            if (task_clock_index >= 0 && d.data[task_clock_index] > 0) {
                ss << "\"CPU Freq(GHz)\":" << static_cast<double>(d.data[cpu_cycles_index]) / d.data[task_clock_index] << ",";
            } else {
                ss << "\"CPU Freq(GHz)\":" << static_cast<double>(d.data[cpu_cycles_index]) * 1e-3 / duration << ",";
            }
            if (instructions_index >= 0 && d.data[instructions_index] > 0) {
                ss << "\"CPI\":" << static_cast<double>(d.data[cpu_cycles_index]) / d.data[instructions_index] << ",";
            }
        }
    }

    void emit_counter_values(std::stringstream& ss, const ScopedSnapshot& d) const {
        const std::locale prev_locale = ss.imbue(std::locale(""));
        const char * sep = "";
        for (size_t i = 0; i < counter_descriptors.size() && i < d.data_size; i++) {
            ss << sep << "\"" << counter_descriptors[i].name << "\":\"" << d.data[i] << "\"";
            sep = ",";
        }
        ss.imbue(prev_locale);
    }

    void emit_extra_data(std::stringstream& ss, const ScopedSnapshot& d) const {
        d.extra_data.append_json(ss);
    }

    std::string serialize_args_json(double duration, const ScopedSnapshot& d) const {
        std::stringstream ss;
        emit_custom_args(ss, duration, d);
        emit_derived_metrics(ss, duration, d);
        emit_counter_values(ss, d);
        emit_extra_data(ss, d);
        return ss.str();
    }

    void write_trace_event_header(std::ofstream& fw, const ScopedSnapshot& d, TimestampConverter& tsc, double start, double duration) const {
        if (d.id < 0) {
            // Async (flow) event: begin/end pair linked by id
            fw << "{\"ph\": \"b\", \"name\": \"" << d.title << "\", \"cat\":\"" << d.cat << "\","
               << "\"pid\": " << my_pid << ", \"id\": " << (-d.id) << ","
               << "\"ts\": " << std::setprecision(15) << start << "},";

            fw << "{\"ph\": \"e\", \"name\": \"" << d.title << "\", \"cat\":\"" << d.cat << "\","
               << "\"pid\": " << my_pid << ", \"id\": " << (-d.id) << ","
               << "\"ts\": " << std::setprecision(15) << tsc.ticks_to_usec(d.tsc_end) << ",";
            return;
        }

        // Complete duration event
        fw << "{\"ph\": \"X\", \"name\": \"" << d.title << "_" << d.id << "\", \"cat\":\"" << d.cat << "\","
           << "\"pid\": " << my_pid << ", \"tid\": " << my_tid << ","
           << "\"ts\": " << std::setprecision(15) << start << ", \"dur\": " << duration << ",";
    }

    void write_trace_event(std::ofstream& fw, const ScopedSnapshot& d, TimestampConverter& tsc) const {
        const double duration = tsc.duration_usec(d.tsc_start, d.tsc_end);
        const double start = tsc.ticks_to_usec(d.tsc_start);
        write_trace_event_header(fw, d, tsc, start, duration);
        fw << "\"args\":{" << serialize_args_json(duration, d) << "}},\n";
    }

    void dump_json(std::ofstream& fw, TimestampConverter& tsc) override {
        if (!json_dump_enabled) {
            return;
        }
        auto data_size = recorded_snapshots.size();
        if (!data_size) {
            return;
        }

        for (auto& d : recorded_snapshots) {
            write_trace_event(fw, d, tsc);
        }
        recorded_snapshots.clear();
        std::cout << LINUX_PERF_"#" << dumper_serial_id << "(" << this << ") finalize: dumped " << data_size << std::endl;
    }

    uint64_t operator[](size_t i) {
        if (i < counter_descriptors.size()) {
            return values[i];
        } else {
            log_printf(LINUX_PERF_"PerfCounterGroup: operator[] with index %lu oveflow (>%lu)\n", i, counter_descriptors.size());
            abort();
        }
    }

    PerfCounterGroup() = default;

    size_t active_counter_count() const {
        return std::min(counter_descriptors.size(), static_cast<size_t>(ScopedSnapshot::data_size));
    }

    // Returns true if all counters can be read via rdpmc (zero syscall overhead).
    bool can_use_rdpmc() const {
        return non_rdpmc_event_count == 0;
    }

    // Capture counter values at scope start (either via rdpmc or read syscall).
    void capture_start_counters(ScopedSnapshot& snapshot) {
        const size_t num_counters = active_counter_count();
        if (can_use_rdpmc()) {
            for (size_t i = 0; i < num_counters; i++) {
                if (counter_descriptors[i].pmc_index) {
                    snapshot.data[i] = read_pmc(counter_descriptors[i].pmc_index - 1);
                }
            }
            return;
        }

        read_all_counters();
        for (size_t i = 0; i < num_counters; i++) {
            snapshot.data[i] = values[i];
        }
    }

    // Capture counter values at scope end, compute deltas, optionally export to a map.
    uint64_t* capture_end_counters(ScopedSnapshot& snapshot, std::map<std::string, uint64_t>* ext_data = nullptr) {
        const size_t num_counters = active_counter_count();

        snapshot.stop();
        if (can_use_rdpmc()) {
            for (size_t i = 0; i < num_counters; i++) {
                if (counter_descriptors[i].pmc_index) {
                    snapshot.data[i] = (read_pmc(counter_descriptors[i].pmc_index - 1) - snapshot.data[i]) & pmc_value_mask;
                } else {
                    snapshot.data[i] = 0;
                }
            }
        } else {
            read_all_counters();
            for (size_t i = 0; i < num_counters; i++) {
                snapshot.data[i] = values[i] - snapshot.data[i];
            }
        }

        if (ext_data) {
            (*ext_data)["ns"] = snapshot.tsc_end - snapshot.tsc_start;
            for (size_t i = 0; i < num_counters; i++) {
                (*ext_data)[counter_descriptors[i].name] = snapshot.data[i];
            }
        }

        return snapshot.data;
    }

    void initialize_dump_state() {
        remaining_dump_quota = PerfConfig::get().dump_limit;
        json_dump_enabled = PerfConfig::get().should_dump_on_cpu(sched_getcpu());
        dumper_serial_id = 0;
        if (json_dump_enabled) {
            dumper_serial_id = TraceFileWriter::get().register_dumper(this);
        }
    }

    void initialize_thread_identity() {
        my_pid = getpid();
        my_tid = gettid();
    }

    // Specifies which perf event to open: type (HW/SW/RAW), config code, and display name.
    struct EventSpec {
        EventSpec(std::string str) {
            parse_from_string(str);
        }
        EventSpec(uint32_t type, uint64_t config, const char * name = "?") : type(type), config(config), name(name) {}

        static bool try_parse_named_config(const std::string& str, uint32_t& type, uint64_t& config) {
            if (str == "HW_CPU_CYCLES" || str == "cycles") {
                type = PERF_TYPE_HARDWARE;
                config = PERF_COUNT_HW_CPU_CYCLES;
                return true;
            }
            if (str == "HW_INSTRUCTIONS" || str == "instructions") {
                type = PERF_TYPE_HARDWARE;
                config = PERF_COUNT_HW_INSTRUCTIONS;
                return true;
            }
            if (str == "SW_PAGE_FAULTS" || str == "pagefaults") {
                type = PERF_TYPE_SOFTWARE;
                config = PERF_COUNT_SW_PAGE_FAULTS;
                return true;
            }
            return false;
        }

        void parse_raw_config(const std::string& str) {
            type = PERF_TYPE_RAW;
            auto items = str_split(str, "=");
            if (items.size() != 2) {
                throw std::runtime_error(std::string("Unknown Perf config: ") + str);
            }
            name = items[0];
            config = parse_raw_event_values(str_split(items[1], ","));
        }

        void parse_from_string(const std::string& str) {
            name = str;
            if (!try_parse_named_config(str, type, config)) {
                parse_raw_config(str);
            }
        }
        uint32_t type;
        uint64_t config;
        std::string name;
    };

    void open_configured_counter(const EventSpec& config) {
        if (config.type == PERF_TYPE_SOFTWARE) {
            open_software_counter(config.config);
        } else if (config.type == PERF_TYPE_HARDWARE) {
            open_hardware_counter(config.config);
        } else if (config.type == PERF_TYPE_RAW) {
            open_raw_counter(config.config);
        }
        counter_descriptors.back().name = config.name;
        snprintf(counter_descriptors.back().format, sizeof(counter_descriptors.back().format), "%%%lulu, ", config.name.size());
    }

    uint32_t my_pid = 0;
    uint32_t my_tid = 0;

    PerfCounterGroup(const std::vector<EventSpec> type_configs, CustomArgsSerializer fn = {}) : custom_args_serializer(fn) {
        for(auto& tc : type_configs) {
            open_configured_counter(tc);
        }

        // env var defined raw events
        for (auto raw_cfg : PerfConfig::get().custom_pmu_events) {
            open_configured_counter({PERF_TYPE_RAW, raw_cfg.second, raw_cfg.first.c_str()});
        }

        initialize_dump_state();
        initialize_thread_identity();

        enable();
    }

    ~PerfCounterGroup() {
        if (json_dump_enabled) {
            TraceFileWriter::get().flush_and_close();
        }
        disable();
        release_perf_event_resources(counter_descriptors, sysconf(_SC_PAGESIZE));
    }

    // Refresh the rdpmc index from the mmap metadata page.
    // Uses the kernel seqlock protocol to ensure a consistent read.
    void refresh_rdpmc_index(CounterDescriptor& ev) {
        if (ev.pmc_index != 0 || !ev.pmeta->cap_user_rdpmc) {
            return;
        }

        uint32_t seqlock;
        do {
            seqlock = ev.pmeta->lock;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            ev.pmc_index = ev.pmeta->index;
            pmc_bit_width = ev.pmeta->pmc_width;
            pmc_value_mask = 1;
            pmc_value_mask = (pmc_value_mask << pmc_bit_width) - 1;
            if (ev.pmeta->cap_user_time) {
                tsc_time_shift = ev.pmeta->time_shift;
                tsc_time_mult = ev.pmeta->time_mult;
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
        } while (ev.pmeta->lock != seqlock || (seqlock & 1));
    }

    void reset_and_enable() {
        ioctl(leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    void initialize_rdpmc_state() {
        // PMC index is only valid when being enabled.
        non_rdpmc_event_count = 0;
        for (auto& ev : counter_descriptors) {
            refresh_rdpmc_index(ev);
            if (ev.pmc_index == 0) {
                non_rdpmc_event_count ++;
            }
        }
    }

    void open_raw_counter(uint64_t config, bool pinned=false) {
        perf_event_attr pea = make_perf_event_attr(PERF_TYPE_RAW,
                                                   config,
                                                   true,
                                                   leader_fd == -1,
                                                   pinned && leader_fd == -1);
        open_counter(&pea);
    }

    void open_hardware_counter(uint64_t config, bool pinned=false) {
        perf_event_attr pea = make_perf_event_attr(PERF_TYPE_HARDWARE,
                                                   config,
                                                   true,
                                                   leader_fd == -1,
                                                   pinned && leader_fd == -1);
        open_counter(&pea);
    }

    void open_software_counter(uint64_t config) {
        // some SW events are counted in kernel, so keep exclude_kernel=false.
        perf_event_attr pea = make_perf_event_attr(PERF_TYPE_SOFTWARE,
                                                   config,
                                                   false,
                                                   false,
                                                   false);
        open_counter(&pea);
    }

    // Set clock source to CLOCK_MONOTONIC_RAW for consistency within the group.
    void set_clock_source(perf_event_attr* pev_attr) {
        pev_attr->use_clockid = 1;
        pev_attr->clockid = CLOCK_MONOTONIC_RAW;
    }

    int open_perf_fd(perf_event_attr* pev_attr, pid_t pid, int cpu) {
        bool has_retried_with_exclude_kernel = false;
        while (true) {
            const int fd = perf_event_open(pev_attr, pid, cpu, leader_fd, 0);
            if (fd >= 0) {
                return fd;
            }
            if (!pev_attr->exclude_kernel && !has_retried_with_exclude_kernel) {
                log_printf(LINUX_PERF_"perf_event_open(type=%d,config=%lld) with exclude_kernel=0 failed (due to /proc/sys/kernel/perf_event_paranoid is 2),  set exclude_kernel=1 and retry...\n",
                           pev_attr->type, pev_attr->config);
                pev_attr->exclude_kernel = 1;
                has_retried_with_exclude_kernel = true;
                continue;
            }
            log_printf(LINUX_PERF_"perf_event_open(type=%d,config=%lld) failed", pev_attr->type, pev_attr->config);
            abort_with_perror("");
        }
    }

    perf_event_mmap_page* mmap_event_page(int fd, size_t mmap_length) {
        return map_perf_event_metadata_or_abort(fd, mmap_length, LINUX_PERF_"mmap perf_event_mmap_page failed:");
    }

    // The first event opened becomes the group leader.
    void promote_to_group_leader(CounterDescriptor& ev, const perf_event_attr* pev_attr) {
        if (leader_fd == -1) {
            leader_fd = ev.fd;
            read_format = pev_attr->read_format;
        }
    }

    // Track indices of well-known events (cycles, instructions, task_clock, ref_cycles)
    // so we can compute derived metrics (CPU frequency, CPI, CPU usage).
    void record_well_known_indices(const perf_event_attr* pev_attr, size_t event_index) {
        if (pev_attr->type == PERF_TYPE_SOFTWARE && pev_attr->config == PERF_COUNT_SW_TASK_CLOCK) {
            task_clock_index = event_index;
        }
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_CPU_CYCLES) {
            cpu_cycles_index = event_index;
        }
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_INSTRUCTIONS) {
            instructions_index = event_index;
        }
    }

    void commit_counter(CounterDescriptor& ev, const perf_event_attr* pev_attr) {
        promote_to_group_leader(ev, pev_attr);
        record_well_known_indices(pev_attr, counter_descriptors.size());
        counter_descriptors.push_back(ev);
    }

    // Open a perf event fd, mmap its metadata page, and add it to the group.
    void open_counter(perf_event_attr* pev_attr, pid_t pid = 0, int cpu = -1) {
        CounterDescriptor ev;

        const size_t mmap_length = sysconf(_SC_PAGESIZE) * 1;
        set_clock_source(pev_attr);

        ev.fd = open_perf_fd(pev_attr, pid, cpu);
        ioctl(ev.fd, PERF_EVENT_IOC_ID, &ev.id);
        ev.pmeta = mmap_event_page(ev.fd, mmap_length);
        commit_counter(ev, pev_attr);
    }

    bool counters_active = false;
    uint32_t non_rdpmc_event_count;     // count of events that cannot use rdpmc fast path

    void enable() {
        if (counters_active) {
            return;
        }
        reset_and_enable();
        initialize_rdpmc_state();
        counters_active = true;
    }

    uint64_t tsc_ticks_to_nanoseconds(uint64_t cyc) {
        uint64_t quot, rem;
        quot  = cyc >> tsc_time_shift;
        rem   = cyc & (((uint64_t)1 << tsc_time_shift) - 1);
        return quot * tsc_time_mult + ((rem * tsc_time_mult) >> tsc_time_shift);
    }

    void disable() {
        if (!counters_active) {
            return;
        }

        ioctl(leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

        for(auto& ev : counter_descriptors) {
            ev.pmc_index = 0;
        }
        counters_active = false;
    }

    // Measure a lambda: read counters before and after, return counter deltas.
    // Optionally prints a formatted summary line if `name` is non-empty.
    // Measure a lambda: read counters before and after, return counter deltas.
    // Optionally prints a formatted summary line if `name` is non-empty.
    template<class FN>
    std::vector<uint64_t> measure(FN fn, std::string name = {}, int64_t loop_cnt = 0, std::function<void(uint64_t, uint64_t*, char*&)> addinfo = {}) {
        ScopedSnapshot snap("measure");
        capture_start_counters(snap);

        auto tsc0 = read_tsc();
        fn();
        auto tsc1 = read_tsc();

        capture_end_counters(snap);

        const size_t cnt = active_counter_count();
        std::vector<uint64_t> pmc(snap.data, snap.data + cnt);

        if (!name.empty()) {
            char log_buff[1024];
            char * log = log_buff;
            char * const log_end = log_buff + sizeof(log_buff) - 1;
            auto safe_snprintf = [&log, log_end](const char* fmt, ...) {
                if (log >= log_end) {
                    return;
                }
                va_list ap;
                va_start(ap, fmt);
                int n = vsnprintf(log, static_cast<size_t>(log_end - log), fmt, ap);
                va_end(ap);
                if (n > 0) {
                    log += n;
                }
            };
            safe_snprintf("\e[33m");
            for(size_t i = 0; i < cnt; i++) {
                safe_snprintf(counter_descriptors[i].format, pmc[i]);
            }
            auto duration_ns = tsc_ticks_to_nanoseconds(tsc1 - tsc0);

            safe_snprintf("\e[0m [%16s] %.3f us", name.c_str(), duration_ns/1e3);
            if (cpu_cycles_index >= 0) {
                safe_snprintf(" CPU:%.2f(GHz)", 1.0 * pmc[cpu_cycles_index] / duration_ns);
                if (instructions_index >= 0) {
                    safe_snprintf(" CPI:%.2f", 1.0 * pmc[cpu_cycles_index] / pmc[instructions_index]);
                }
                if (loop_cnt > 0) {
                    safe_snprintf(" CPK:%.1fx%ld", 1.0 * pmc[cpu_cycles_index] / loop_cnt, loop_cnt);
                }
            }
            if (addinfo) {
                addinfo(duration_ns, &pmc[0], log);
            }
            safe_snprintf("\n");
            log_message(log_buff);
        }
        return pmc;
    }

    // Read all counter values from the group leader fd via read() syscall.
    // The kernel returns (nr, [time_enabled], [time_running], {value, id}*nr).
    // We match each {value, id} pair back to its counter descriptor by id.
    void read_all_counters(bool verbose = false) {
        for(size_t i = 0; i < counter_descriptors.size(); i++) values[i] = 0;

        if (::read(leader_fd, read_buffer, sizeof(read_buffer)) == -1) {
            abort_with_perror(LINUX_PERF_"read perf event failed:");
        }

        uint64_t * readv = read_buffer;
        auto nr = *readv++;
        if (verbose) {
            log_printf("number of counters:\t%lu\n", nr);
        }
        if (read_format & PERF_FORMAT_TOTAL_TIME_ENABLED) {
            auto val = *readv++;
            if (verbose) { log_printf("time_enabled:\t%lu\n", val); }
        }
        if (read_format & PERF_FORMAT_TOTAL_TIME_RUNNING) {
            auto val = *readv++;
            if (verbose) { log_printf("time_running:\t%lu\n", val); }
        }

        for (size_t i = 0; i < nr; i++) {
            auto value = *readv++;
            auto id = *readv++;
            for (size_t k = 0; k < counter_descriptors.size(); k++) {
                if (id == counter_descriptors[k].id) {
                    values[k] = value;
                }
            }
        }

        if (verbose) {
            for (size_t k = 0; k < counter_descriptors.size(); k++) {
                log_printf("\t[%lu]: %lu\n", k, values[k]);
            }
        }
    }

    //================================================================================
    // Profiler API with json_dump capability
    //================================================================================

    // RAII guard for a profiled scope.  Construction captures start counters;
    // destruction (or explicit finish()) captures end counters and computes deltas.
    // Move-only to prevent double-finish.  The optional sampling lock prevents
    // nested Profile() calls from recording, reducing overhead in hot paths.
    struct ProfileScope {
        PerfCounterGroup* pevg = nullptr;
        ScopedSnapshot* pd = nullptr;
        bool do_unlock = false;
        size_t num_events = 0;
        ProfileScope() = default;
        ProfileScope(PerfCounterGroup* pevg, ScopedSnapshot* pd, bool do_unlock = false) : pevg(pevg), pd(pd), do_unlock(do_unlock) {}

        // Move only
        ProfileScope(const ProfileScope&) = delete;
        ProfileScope& operator=(const ProfileScope&) = delete;

        ProfileScope(ProfileScope&& other)
            : pevg(other.pevg)
            , pd(other.pd)
            , do_unlock(other.do_unlock)
            , num_events(other.num_events)
        {
            other.pevg = nullptr;
            other.pd = nullptr;
            other.do_unlock = false;
            other.num_events = 0;
        }

        ProfileScope& operator=(ProfileScope&& other) {
            if (&other != this) {
                if (pevg) {
                    finish();
                }
                pevg = other.pevg;
                pd = other.pd;
                do_unlock = other.do_unlock;
                num_events = other.num_events;
                other.pevg = nullptr;
                other.pd = nullptr;
                other.do_unlock = false;
                other.num_events = 0;
            }
            return *this;
        }

        uint64_t* finish(std::map<std::string, uint64_t>* ext_data = nullptr) {
            if (do_unlock) {
                PerfCounterGroup::sampling_lock() --;
            }
            num_events = 0;
            if (!pevg || !pd) {
                return nullptr;
            }

            num_events = pevg->active_counter_count();
            uint64_t* result = pevg->capture_end_counters(*pd, ext_data);

            pevg = nullptr;
            return result;
        }

        ~ProfileScope() {
            finish();
        }
    };

    ScopedSnapshot* begin_scope(const std::string& title, int id = 0, const std::string& cat = "") {
        if (sampling_lock().load() != 0) {
            return nullptr;
        }
        if (remaining_dump_quota == 0) {
            return nullptr;
        }
        remaining_dump_quota --;

        CpuContextSwitchTracker::get().drain_all_ring_buffers();

        recorded_snapshots.emplace_back(title, cat);
        auto* pd = &recorded_snapshots.back();
        pd->id = id;

        capture_start_counters(*pd);

        return pd;
    }

    static PerfCounterGroup& get() {
        thread_local PerfCounterGroup pevg({
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "HW_CACHE_MISSES"},
            //{PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES, "HW_REF_CPU_CYCLES"},
            {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "SW_CONTEXT_SWITCHES"},
            {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "SW_TASK_CLOCK"},
            {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, "SW_PAGE_FAULTS"}
        });
        return pevg;
    }

    // Global atomic lock that affects all threads.
    // When non-zero, begin_scope() returns nullptr (skips profiling).
    // Used by sampling probability to suppress nested measurements.
    static std::atomic_int& sampling_lock() {
        static std::atomic_int lock{0};
        return lock;
    }
};

using ProfileScope = PerfCounterGroup::ProfileScope;

// ============================================================================
// Section: Public API
//
// Profile(title, ...)       -- returns a ProfileScope RAII guard that captures
//                              counters for the current scope.
// Profile(probability, ...) -- same, but randomly skips (1-p) fraction of scopes
//                              to reduce overhead in high-frequency paths.
// Init()                    -- call once from main thread to start context-switch
//                              tracking and register the main thread.
// ============================================================================

inline bool should_skip_sample(float sampling_probability) {
    return (std::rand() % 1000) * 0.001f >= sampling_probability;
}

inline void suppress_nested_sampling(bool disable_profile) {
    if (disable_profile) {
        PerfCounterGroup::sampling_lock() ++;
    }
}

// Shared implementation behind the public Profile overloads.
template <typename ... Args>
ProfileScope make_profile_scope(const std::string& title,
                                const std::string* category,
                                bool use_sampling_probability,
                                float sampling_probability,
                                int id,
                                Args&&... args) {
    auto& pevg = PerfCounterGroup::get();
    auto* pd = category ? pevg.begin_scope(title, id, *category) : pevg.begin_scope(title, id);
    if (pd) {
        pd->extra_data.set_datas(std::forward<Args>(args)...);
    }

    bool disable_profile = false;
    if (use_sampling_probability) {
        disable_profile = should_skip_sample(sampling_probability);
        suppress_nested_sampling(disable_profile);
    }
    return ProfileScope(&pevg, pd, disable_profile);
}

// per-thread event group with default events pre-selected
// args can be:      int/float/vector
template <typename ... Args>
ProfileScope Profile(const std::string& title, int id = 0, Args&&... args) {
    return make_profile_scope(title, nullptr, false, 0.0f, id, std::forward<Args>(args)...);
}

template <typename ... Args>
ProfileScope Profile(const std::string& title, const std::string& category, int id = 0, Args&&... args) {
    return make_profile_scope(title, &category, false, 0.0f, id, std::forward<Args>(args)...);
}

// overload accept sampling_probability, which can be used to disable profile in scope
template <typename ... Args>
ProfileScope Profile(float sampling_probability, const std::string& title, int id = 0, Args&&... args) {
    return make_profile_scope(title, nullptr, true, sampling_probability, id, std::forward<Args>(args)...);
}

template <typename ... Args>
ProfileScope Profile(float sampling_probability, const std::string& title, const std::string& category, int id = 0, Args&&... args) {
    return make_profile_scope(title, &category, true, sampling_probability, id, std::forward<Args>(args)...);
}

inline int Init() {
    // this is for capture all context switching events
    CpuContextSwitchTracker::get();

    // this is for making main threads the first process
    auto dummy = Profile("start");
    return 0;
}

} // namespace LinuxPerf

// ============================================================================
// Section: C API Bridge
// extern "C" functions for calling from plain C code.  Wraps the C++
// ProfileScope in an opaque void* handle.
// ============================================================================

#ifdef LINUX_PERF_C_API
inline void fill_extra_data_from_va_list(LinuxPerf::PerfCounterGroup::ScopedSnapshot* pd, int count, va_list ap) {
    if (!pd) {
        return;
    }
    for (int j = 0; j < count; j++) {
        pd->extra_data.set_data(j, va_arg(ap, int));
    }
    pd->extra_data.set_terminator(count);
}

inline void* begin_c_api_scope(const char * title,
                                 const char * category,
                                 float sampling_probability,
                                 bool use_sampling_probability,
                                 int count,
                                 va_list ap) {
    auto& pevg = LinuxPerf::PerfCounterGroup::get();
    auto* pd = pevg.begin_scope(title, 0, category);
    fill_extra_data_from_va_list(pd, count, ap);

    bool disable_profile = false;
    if (use_sampling_probability) {
        disable_profile = LinuxPerf::should_skip_sample(sampling_probability);
        LinuxPerf::suppress_nested_sampling(disable_profile);
    }
    return reinterpret_cast<void*>(new LinuxPerf::ProfileScope{&pevg, pd, disable_profile});
}

extern "C" void* linux_perf_profile_start(const char * title, const char * category, int count, ...) {
    va_list ap;
    va_start(ap, count);
    void* profile = begin_c_api_scope(title, category, 0.0f, false, count, ap);
    va_end(ap);
    return profile;
}
extern "C" void* linux_perf_profile_start_prob(const char * title, const char * category, float sampling_probability, int count, ...) {
    va_list ap;
    va_start(ap, count);
    void* profile = begin_c_api_scope(title, category, sampling_probability, true, count, ap);
    va_end(ap);
    return profile;
}

extern "C" void linux_perf_profile_end(void * p) {
    delete reinterpret_cast<LinuxPerf::ProfileScope *>(p);
}
#endif
#else ///#ifdef __cplusplus
// C_API
extern void* linux_perf_profile_start(const char * title, const char * category, int count, ...);
extern void* linux_perf_profile_start_prob(const char * title, const char * category, float sampling_probability, int count, ...);
extern void linux_perf_profile_end(void* p);
#endif ///#ifdef __cplusplus
