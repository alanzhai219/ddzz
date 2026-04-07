#pragma once

#ifdef __cplusplus

#include <linux/perf_event.h>
#include <time.h>
//#include <linux/time.h>
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
#include <chrono>
#include <thread>
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

       For instance For x86 CPUs NNN represents the raw register encoding with the layout of IA32_PERFEVTSELx MSRs (see [Intel® 64 and IA-32 Architectures Software Developer’s Manual Volume 3B: System Programming Guide] Figure 30-1
       Layout of IA32_PERFEVTSELx MSRs) or AMD’s PerfEvtSeln (see [AMD64 Architecture Programmer’s Manual Volume 2: System Programming], Page 344, Figure 13-7 Performance Event-Select Register (PerfEvtSeln)).

       Note: Only the following bit fields can be set in x86 counter registers: event, umask, edge, inv, cmask. Esp. guest/host only and OS/user mode flags must be setup using EVENT MODIFIERS.

 event 7:0
 umask 15:8
 edge  18
 inv   23
 cmask 31:24
*/
#define X86_RAW_EVENT(EventSel, UMask, CMask) ((CMask << 24) | (UMask << 8) | (EventSel))

namespace LinuxPerf {

// Design overview:
// 1. PerfRawConfig parses the LINUX_PERF environment variable.
// 2. PerfEventGroup measures per-thread counters and optional trace scopes.
// 3. PerfEventContextSwitch captures CPU-wide context-switch records for timeline view.
// 4. PerfEventJsonDumper collects all dumpers and writes a single perf_dump.json.

#define _LINE_STRINGIZE(x) _LINE_STRINGIZE2(x)
#define _LINE_STRINGIZE2(x) #x
#define LINE_STRING _LINE_STRINGIZE(__LINE__)

#define LINUX_PERF_ "\e[33m[LINUX_PERF:" LINE_STRING "]\e[0m "

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

struct TscCounter {
    uint64_t tsc_ticks_per_second;
    uint64_t tsc_ticks_base;
    double tsc_to_usec(uint64_t tsc_ticks) const {
        if (tsc_ticks < tsc_ticks_base) {
            return 0;
        }
        return (tsc_ticks - tsc_ticks_base) * 1000000.0 / tsc_ticks_per_second;
    }
    double tsc_to_usec(uint64_t tsc_ticks0, uint64_t tsc_ticks1) const {
        if (tsc_ticks1 < tsc_ticks0) {
            return 0;
        }
        return (tsc_ticks1 - tsc_ticks0) * 1000000.0 / tsc_ticks_per_second;
    }
    TscCounter() {
        // Use CLOCK_MONOTONIC_RAW as unified time source (unit: nanoseconds).
        // This avoids the ~1s calibration sleep that a TSC-based approach would need,
        // and stays consistent with the clockid used in perf_event_attr.
        tsc_ticks_per_second = 1000000000; // ns
        tsc_ticks_base = get_time_ns();
    }
};

class IPerfEventDumper {
public:
    virtual ~IPerfEventDumper() = default;
    virtual void dump_json(std::ofstream& fw, TscCounter& tsc) = 0;
};

struct PerfEventJsonDumper {
    std::mutex g_mutex;
    std::set<IPerfEventDumper*> all_dumpers;
    const char* dump_file_name = "perf_dump.json";
    bool dump_file_over = false;
    bool not_finalized = true;
    std::ofstream fw;
    std::atomic_int totalProfilerManagers{0};
    TscCounter tsc;

    ~PerfEventJsonDumper() {
        if (not_finalized) {
            finalize();
        }
    }

    void finalize() {
        if (!not_finalized) {
            return;
        }
        std::lock_guard<std::mutex> guard(g_mutex);
        // Re-check under lock to prevent double finalization from concurrent threads.
        if (!not_finalized || dump_file_over || all_dumpers.empty()) {
            return;
        }

        // start dump
        fw.open(dump_file_name, std::ios::out);
        fw << "{\n";
        fw << "\"schemaVersion\": 1,\n";
        fw << "\"traceEvents\": [\n";
        fw.flush();

        for (auto& pthis : all_dumpers) {
            pthis->dump_json(fw, tsc);
        }
        all_dumpers.clear();

        fw << R"({
            "name": "Profiler End",
            "ph": "i",
            "s": "g",
            "pid": "Traces",
            "tid": "Trace OV Profiler",
            "ts":)"
           << tsc.tsc_to_usec(get_time_ns()) << "}";
        fw << "]\n";
        fw << "}\n";
        auto total_size = fw.tellp();
        fw.close();
        dump_file_over = true;
        not_finalized = false;

        std::cout << LINUX_PERF_"Dumped ";
        
        if (total_size < 1024) {
            std::cout << total_size << " bytes ";
        } else if (total_size < 1024*1024) {
            std::cout << total_size/1024 << " KB ";
        } else {
            std::cout << total_size/(1024 * 1024) << " MB ";
        }
        std::cout << " to " << dump_file_name << std::endl;
    }

    int register_manager(IPerfEventDumper* pthis) {
        std::lock_guard<std::mutex> guard(g_mutex);
        std::ostringstream ss;
        auto serial_id = totalProfilerManagers.fetch_add(1);
        ss << LINUX_PERF_"#" << serial_id << "(" << pthis << ") : is registered." << std::endl;
        log_message(ss.str());
        all_dumpers.emplace(pthis);
        return serial_id;
    }

    // local C++ static-based singleton fails when this header-only tool is being
    // used by two separate shared libs (or one by shared lib, another by final application exe)
    //
    template<class T>
    struct singleton_over_so {
        static constexpr const char * shm_name = "/linuxperf_shm01";
        struct shm_data {
            T * pobj;
            int64_t ref_cnt;
            pid_t pid_spinlock0;
            pid_t pid_spinlock1;
        };
        shm_data* _data;
        T * pobj;
        singleton_over_so() {
            int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
            if (ftruncate(fd, 4096) != 0) {
                abort_with_perror("ftruncate failed!");
            }
            pid_t pid = getpid();
            _data = reinterpret_cast<shm_data*>(mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
            if (__atomic_exchange_n(&_data->pid_spinlock0, pid, __ATOMIC_SEQ_CST) != pid) {
                // we are the first one that set pid_spinlock to pid, do initialization
                _data->pobj = new T();
                __atomic_store_n(&_data->ref_cnt, 0, __ATOMIC_SEQ_CST);
                __atomic_store_n(&_data->pid_spinlock1, pid, __ATOMIC_SEQ_CST);
            } else {
                // spin until the owner has done
                while (__atomic_load_n(&_data->pid_spinlock1, __ATOMIC_SEQ_CST) != pid);
            }
            __atomic_add_fetch(&_data->ref_cnt, 1, __ATOMIC_SEQ_CST);
            close(fd);

            pobj = _data->pobj;
        }
        T& obj() {
            return *(pobj);
        }
        ~singleton_over_so() {
            if (__atomic_sub_fetch(&_data->ref_cnt, 1, __ATOMIC_SEQ_CST) == 0) {
                delete _data->pobj;
                munmap(_data, 4096);
            }
            shm_unlink(shm_name);
        }
    };

    static PerfEventJsonDumper& get() {
        static singleton_over_so<PerfEventJsonDumper> inst;
        return inst.obj();
    }
};

inline std::vector<std::string> str_split(const std::string& s, std::string delimiter) {
    std::vector<std::string> ret;
    if (delimiter.empty()) {
        ret.push_back(s);
        return ret;
    }
    size_t last = 0;
    size_t next = 0;
    while ((next = s.find(delimiter, last)) != std::string::npos) {
        //std::cout << last << "," << next << "=" << s.substr(last, next-last) << "\n";
        ret.push_back(s.substr(last, next-last));
        last = next + delimiter.size();
    }
    ret.push_back(s.substr(last));
    return ret;
}

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

// Small RAII helper to ensure lock release on every return path.
class AtomicSpinGuard {
public:
    explicit AtomicSpinGuard(std::atomic<int>& lock) : lock_(lock), owns_(lock_.exchange(1) == 0) {}
    ~AtomicSpinGuard() {
        if (owns_) {
            lock_.store(0);
        }
    }
    bool owns_lock() const { return owns_; }

private:
    std::atomic<int>& lock_;
    bool owns_;
};

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

struct RingBufferCursor {
    perf_event_mmap_page& meta;
    uint64_t offset;

    RingBufferCursor(perf_event_mmap_page& meta, uint64_t offset) : meta(meta), offset(offset) {}

    template<typename T>
    T read() {
        return read_ring_buffer<T>(meta, offset);
    }
};

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

    static ContextSwitchRecord parse(RingBufferCursor& cursor) {
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

template<int Size>
struct ExtraDataStorage {
    union Value {
        double f;
        int64_t i;
        void* p;
    };
    Value values[Size];
    char types[Size];

    ExtraDataStorage() : types() {}

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

struct PerfRawConfig {
    // --- data members ---
    int64_t dump = 0;
    cpu_set_t cpu_mask;
    bool switch_cpu = false;
    std::vector<std::pair<std::string, uint64_t>> raw_configs;

    uint64_t parse_raw_event_config(const std::string& raw_evt) const {
        // raw config format: 0x10d1 or EventSel-UMask-0
        if (raw_evt.find('-') != std::string::npos) {
            auto evsel_umask_cmask = str_split(raw_evt, "-");
            if (evsel_umask_cmask.size() < 2) {
                return 0;
            }
            auto evsel = strtoul(evsel_umask_cmask[0].c_str(), nullptr, 0);
            auto umask = strtoul(evsel_umask_cmask[1].c_str(), nullptr, 0);
            uint64_t cmask = 0;
            if (evsel_umask_cmask.size() > 2) {
                cmask = strtoul(evsel_umask_cmask[2].c_str(), nullptr, 0);
            }
            return X86_RAW_EVENT(evsel, umask, cmask);
        }
        return strtoul(raw_evt.c_str(), nullptr, 0);
    }

    void parse_cpu_list(const std::string& value) {
        // cpus=56 or cpus=56,57,59
        auto cpus = str_split(value, ",");
        CPU_ZERO(&cpu_mask);
        for (auto& cpu : cpus) {
            CPU_SET(std::atoi(cpu.c_str()), &cpu_mask);
        }
    }

    void enable_switch_cpu_mode() {
        // get cpu_mask as early as possible
        switch_cpu = true;
        CPU_ZERO(&cpu_mask);
        if (sched_getaffinity(getpid(), sizeof(cpu_set_t), &cpu_mask)) {
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
                dump = strtoll(value.c_str(), nullptr, 0);
            } else if (key == "cpus") {
                // thread affinity can be changed by runtime libs; allow explicit cpu list.
                parse_cpu_list(value);
            } else {
                auto config = parse_raw_event_config(value);
                if (config > 0) {
                    raw_configs.emplace_back(key, config);
                }
            }
            return;
        }

        if (items.size() == 1) {
            if (items[0] == "switch-cpu") {
                enable_switch_cpu_mode();
            }
            if (items[0] == "dump") {
                dump = std::numeric_limits<int64_t>::max(); // no limit to number of dumps
            }
        }
    }

    void print_config() const {
        for (auto& cfg : raw_configs) {
            log_printf(LINUX_PERF_" config: %s=0x%lx\n", cfg.first.c_str(), cfg.second);
        }
        if (switch_cpu) {
            log_printf(LINUX_PERF_" config: switch_cpu\n");
        }
        if (dump) {
            log_printf(LINUX_PERF_" config: dump=%ld\n", dump);
        }
        if (CPU_COUNT(&cpu_mask)) {
            std::ostringstream ss;
            ss << LINUX_PERF_ << " config: cpus=";
            for (int cpu = 0; cpu < (int)sizeof(cpu_set_t) * 8; cpu++) {
                if (CPU_ISSET(cpu, &cpu_mask)) {
                    ss << cpu << ",";
                }
            }
            ss << std::endl;
            log_message(ss.str());
        }
    }

    PerfRawConfig() {
        CPU_ZERO(&cpu_mask);
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
        print_config();
    }

    bool dump_on_cpu(int cpu) {
        if (dump == 0) {
            return false;
        }
        if (CPU_COUNT(&cpu_mask)) {
            return CPU_ISSET(cpu, &cpu_mask);
        }
        return true;
    }

    static PerfRawConfig& get() {
        static PerfRawConfig inst;
        return inst;
    }
};


// context switch events
// this will visualize 
struct PerfEventContextSwitch : public IPerfEventDumper {
    bool is_enabled;

    struct CpuEvent {
        int fd;
        perf_event_mmap_page * pmeta;
        int cpu;
        uint64_t ctx_switch_in_time;
        uint64_t ctx_switch_in_tid;
        uint64_t ctx_last_time;

        CpuEvent(int fd, perf_event_mmap_page * pmeta): fd(fd), pmeta(pmeta) {}
    };
    std::vector<CpuEvent> events;

    static perf_event_attr make_ctx_switch_perf_attr() {
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

    static size_t context_switch_mmap_length() {
        return sysconf(_SC_PAGESIZE) * (1024 + 1);
    }

    void add_cpu_event(int cpu) {
        perf_event_attr pea = make_ctx_switch_perf_attr();
        const size_t mmap_length = context_switch_mmap_length();

        // measures all processes/threads on the specified CPU
        const int ctx_switch_fd = perf_event_open(&pea, -1, cpu, -1, 0);
        if (ctx_switch_fd < 0) {
            abort_with_perror(LINUX_PERF_"PerfEventContextSwitch perf_event_open failed (check /proc/sys/kernel/perf_event_paranoid please)");
        }

        auto* ctx_switch_pmeta = map_perf_event_metadata_or_abort(
                ctx_switch_fd, mmap_length, LINUX_PERF_"mmap perf_event_mmap_page failed:");

        log_printf(LINUX_PERF_"perf_event_open CPU_WIDE context_switch on cpu %d, ctx_switch_fd=%d\n", cpu, ctx_switch_fd);
        events.emplace_back(ctx_switch_fd, ctx_switch_pmeta);
        events.back().ctx_switch_in_time = get_time_ns();
        events.back().ctx_last_time = get_time_ns();
        events.back().cpu = cpu;
    }

    bool should_enable(const cpu_set_t& mask) {
        const long number_of_processors = sysconf(_SC_NPROCESSORS_ONLN);
        log_printf(LINUX_PERF_"sizeof(cpu_set_t):%lu: _SC_NPROCESSORS_ONLN=%ld CPU_COUNT=%d\n",
                   sizeof(cpu_set_t), number_of_processors, CPU_COUNT(&mask));
        if (CPU_COUNT(&mask) >= number_of_processors) {
            log_printf(LINUX_PERF_" no affinity is set, will not enable PerfEventContextSwitch\n");
            return false;
        }
        return true;
    }

    void open_events_for_mask(const cpu_set_t& mask) {
        for (int cpu = 0; cpu < (int)sizeof(cpu_set_t) * 8; cpu++) {
            if (!CPU_ISSET(cpu, &mask)) {
                continue;
            }
            add_cpu_event(cpu);
        }
    }

    void append_active_context_switches() {
        for (auto& ev : events) {
            if (ev.ctx_switch_in_time == 0) {
                continue;
            }
            all_dump_data.emplace_back();
            auto* pd = &all_dump_data.back();
            pd->tid = ev.ctx_switch_in_tid;
            pd->cpu = ev.cpu;
            pd->tsc_start = ev.ctx_switch_in_time;
            pd->tsc_end = get_time_ns();
            ev.ctx_switch_in_time = 0;
        }
    }

    bool needs_ring_buffer_update() const {
        for (auto& ev : events) {
            const auto& mmap_meta = *ev.pmeta;
            const auto used_size = (mmap_meta.data_head - mmap_meta.data_tail) % mmap_meta.data_size;
            if (used_size > (mmap_meta.data_size >> 1)) {
                return true;
            }
        }
        return false;
    }

    void dump_ring_buffer_constants() const {
        log_printf("PERF_RECORD_SWITCH = %d\n", PERF_RECORD_SWITCH);
        log_printf("PERF_RECORD_SWITCH_CPU_WIDE = %d\n", PERF_RECORD_SWITCH_CPU_WIDE);
        log_printf("PERF_RECORD_MISC_SWITCH_OUT = %d\n", PERF_RECORD_MISC_SWITCH_OUT);
        log_printf("PERF_RECORD_MISC_SWITCH_OUT_PREEMPT  = %d\n", PERF_RECORD_MISC_SWITCH_OUT_PREEMPT);
    }

    void append_switch_out_event(CpuEvent& ev, uint32_t tid, uint32_t cpu, uint64_t time, __u16 misc) {
        all_dump_data.emplace_back();
        auto* pd = &all_dump_data.back();
        pd->tid = tid;
        pd->cpu = cpu;
        pd->preempt = (misc & PERF_RECORD_MISC_SWITCH_OUT_PREEMPT);
        pd->tsc_start = ev.ctx_switch_in_time;
        pd->tsc_end = time;

        if (ring_buffer_verbose) {
            log_printf("\t  cpu: %u tid: %u  %lu (+%lu)\n", cpu, tid, ev.ctx_switch_in_time, time - ev.ctx_switch_in_time);
        }
        ev.ctx_switch_in_time = 0;
    }

    void process_ring_buffer_record(CpuEvent& ev, perf_event_mmap_page& mmap_meta, uint64_t& head0, uint64_t head1) {
        const auto h0 = head0;
        RingBufferCursor cursor(mmap_meta, head0);
        const ContextSwitchRecord record = ContextSwitchRecord::parse(cursor);
        (void)record.reserved0;
        (void)record.next_prev_pid;
        (void)record.pid;

        if (record.tid > 0 && ring_buffer_verbose) {
            log_printf("event: %lu/%lu\ttype,misc,size=(%u,%u,%u) cpu%u,next_prev_tid=%u,tid=%u  time:(%lu), (+%lu)\n",
                       h0, head1, record.type, record.misc, record.size, record.cpu,
                       record.next_prev_tid, record.tid, record.time, record.time - ev.ctx_last_time);
        }

        if (record.type == PERF_RECORD_SWITCH_CPU_WIDE && record.tid > 0) {
            if (record.misc & PERF_RECORD_MISC_SWITCH_OUT || record.misc & PERF_RECORD_MISC_SWITCH_OUT_PREEMPT) {
                append_switch_out_event(ev, record.tid, record.cpu, record.time, record.misc);
            } else {
                ev.ctx_switch_in_time = record.time;
                ev.ctx_switch_in_tid = record.tid;
            }
        }

        ev.ctx_last_time = record.time;
        head0 = h0 + record.size;
    }

    void process_ring_buffer(CpuEvent& ev) {
        auto& mmap_meta = *ev.pmeta;
        uint64_t head0 = mmap_meta.data_tail;
        const uint64_t head1 = mmap_meta.data_head;

        if (head0 == head1) {
            return;
        }
        if (ring_buffer_verbose) {
            dump_ring_buffer_constants();
        }

        while (head0 < head1) {
            process_ring_buffer_record(ev, mmap_meta, head0, head1);
        }

        if (head0 != head1) {
            log_printf("head0(%lu) != head1(%lu)\n", head0, head1);
            abort_with_perror("ring buffer head mismatch");
        }

        // update tail so kernel can keep generate event records
        mmap_meta.data_tail = head0;
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    PerfEventContextSwitch() {
        is_enabled = PerfRawConfig::get().switch_cpu;
        if (is_enabled) {
            // make sure TSC in PerfEventJsonDumper is the very first thing to initialize
            PerfEventJsonDumper::get().register_manager(this);

            // open fd for each CPU
            const cpu_set_t mask = PerfRawConfig::get().cpu_mask;
            if (!should_enable(mask)) {
                is_enabled = false;
                return;
            }

            open_events_for_mask(mask);
            my_pid = getpid();
            my_tid = gettid();
        }
    }

    ~PerfEventContextSwitch() {
        if (is_enabled) {
            PerfEventJsonDumper::get().finalize();
        }
        release_perf_event_resources(events, context_switch_mmap_length());
    }

    struct ProfileData {
        uint64_t tsc_start;
        uint64_t tsc_end;
        uint32_t tid;
        uint32_t cpu;
        bool preempt;   // preempt means current TID preempts previous thread
    };

    std::deque<ProfileData> all_dump_data;

    void dump_json(std::ofstream& fw, TscCounter& tsc) override {
        if (!is_enabled) {
            return;
        }

        updateRingBuffer();

        auto data_size = all_dump_data.size();
        if (!data_size) {
            return;
        }

        append_active_context_switches();

        auto pid = 9999;    // fake pid for CPU
        auto cat = "TID";
        
        // TID is used for CPU id instead
        for (auto& d : all_dump_data) {
            auto duration = tsc.tsc_to_usec(d.tsc_start, d.tsc_end);
            auto start = tsc.tsc_to_usec(d.tsc_start);
            //auto end = tsc.tsc_to_usec(d.tsc_end);
            auto cpu_id = d.cpu;

            fw << "{\"ph\": \"X\", \"name\": \"" << d.tid << "\", \"cat\":\"" << cat << "\","
                << "\"pid\": " << pid << ", \"tid\": \"CPU" << cpu_id <<  "\","
                << "\"ts\": " << std::setprecision (15) << start << ", \"dur\": " << duration << "},\n";
        }
    }

    bool ring_buffer_verbose = false;
    uint32_t my_pid = 0;
    uint32_t my_tid = 0;
    std::atomic<int> atom_guard{0};

    void updateRingBuffer() {
        // only one thread can update ring-buffer at one time
        AtomicSpinGuard lock_guard(atom_guard);
        if (!lock_guard.owns_lock()) {
            return;
        }

        if (!needs_ring_buffer_update()) {
            return;
        }

        for (auto& ev : events) {
            process_ring_buffer(ev);
        }
    }

    static PerfEventContextSwitch& get() {
        static PerfEventContextSwitch inst;
        return inst;
    }
};

struct PerfEventGroup : public IPerfEventDumper {
    static constexpr size_t kReadBufferU64Count = 512;

    int group_fd = -1;
    uint64_t read_format;

    struct CounterEvent {
        int fd = -1;
        uint64_t id = 0;
        uint64_t pmc_index = 0;
        perf_event_mmap_page* pmeta = nullptr;
        std::string name = "?";
        char format[32];
    };
    std::vector<CounterEvent> events;

    uint64_t read_buf[kReadBufferU64Count]; // 4KB
    uint64_t time_enabled;
    uint64_t time_running;
    uint64_t pmc_width;
    uint64_t pmc_mask;
    uint64_t values[32];
    uint32_t tsc_time_shift;
    uint32_t tsc_time_mult;

    // ref_cpu_cycles even id
    // this event is fixed function counter provided by most x86 CPU
    // and it provides TSC clock which is:
    //    - very high-resolution (<1ns or >1GHz)
    //    - independent of CPU-frequency throttling
    int ref_cpu_cycles_evid = -1;
    int sw_task_clock_evid = -1;
    int hw_cpu_cycles_evid = -1;
    int hw_instructions_evid = -1;

    struct ProfileData {
        uint64_t tsc_start;
        uint64_t tsc_end;
        std::string title;
        std::string cat;
        int32_t id;
        static const int data_size = 16; // 4(fixed) + 8(PMU) + 4(software)
        uint64_t data[data_size] = {0};
        ExtraDataStorage<data_size> extra_data;

        template<typename T>
        void set_extra_data(int& i, T* t) { extra_data.set_data(i, t); }
        void set_extra_data(int& i, float t) { extra_data.set_data(i, t); }
        void set_extra_data(int& i, double t) { extra_data.set_data(i, t); }
        template<typename T>
        void set_extra_data(int& i, T t) { extra_data.set_data(i, t); }
        template<typename T>
        void set_extra_data(int& i, const std::vector<T>& t) { extra_data.set_data(i, t); }
        void set_extra_data(int i) { extra_data.set_terminator(i); }

        template <typename ... Values>
        void set_extra_datas(Values... vals) { extra_data.set_datas(vals...); }

        ProfileData(std::string title, std::string cat = {})
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

    bool enable_dump_json = false;
    int64_t dump_limit = 0;
    std::deque<ProfileData> all_dump_data;
    int serial;

    using EventArgsSerializer = std::function<void(std::ostream& fw, double usec, uint64_t* counters)>;
    EventArgsSerializer event_args_serializer;

    void append_serializer_args(std::stringstream& ss, double duration, const ProfileData& d) const {
        if (event_args_serializer) {
            event_args_serializer(ss, duration, const_cast<uint64_t*>(d.data));
        }
    }

    void append_derived_metrics(std::stringstream& ss, double duration, const ProfileData& d) const {
        if (sw_task_clock_evid >= 0) {
            // PERF_COUNT_SW_TASK_CLOCK is in nano-seconds.
            ss << "\"CPU Usage\":" << (d.data[sw_task_clock_evid] * 1e-3) / duration << ",";
        }
        if (hw_cpu_cycles_evid >= 0) {
            if (sw_task_clock_evid >= 0 && d.data[sw_task_clock_evid] > 0) {
                ss << "\"CPU Freq(GHz)\":" << static_cast<double>(d.data[hw_cpu_cycles_evid]) / d.data[sw_task_clock_evid] << ",";
            } else {
                ss << "\"CPU Freq(GHz)\":" << static_cast<double>(d.data[hw_cpu_cycles_evid]) * 1e-3 / duration << ",";
            }
            if (hw_instructions_evid >= 0 && d.data[hw_instructions_evid] > 0) {
                ss << "\"CPI\":" << static_cast<double>(d.data[hw_cpu_cycles_evid]) / d.data[hw_instructions_evid] << ",";
            }
        }
    }

    void append_counter_values(std::stringstream& ss, const ProfileData& d) const {
        const std::locale prev_locale = ss.imbue(std::locale(""));
        const char * sep = "";
        for (size_t i = 0; i < events.size() && i < d.data_size; i++) {
            ss << sep << "\"" << events[i].name << "\":\"" << d.data[i] << "\"";
            sep = ",";
        }
        ss.imbue(prev_locale);
    }

    void append_extra_data(std::stringstream& ss, const ProfileData& d) const {
        d.extra_data.append_json(ss);
    }

    std::string build_args_json(double duration, const ProfileData& d) const {
        std::stringstream ss;
        append_serializer_args(ss, duration, d);
        append_derived_metrics(ss, duration, d);
        append_counter_values(ss, d);
        append_extra_data(ss, d);
        return ss.str();
    }

    void write_event_prefix(std::ofstream& fw, const ProfileData& d, TscCounter& tsc, double start, double duration) const {
        if (d.id < 0) {
            fw << "{\"ph\": \"b\", \"name\": \"" << d.title << "\", \"cat\":\"" << d.cat << "\"," 
               << "\"pid\": " << my_pid << ", \"id\": " << (-d.id) << ","
               << "\"ts\": " << std::setprecision(15) << start << "},";

            fw << "{\"ph\": \"e\", \"name\": \"" << d.title << "\", \"cat\":\"" << d.cat << "\"," 
               << "\"pid\": " << my_pid << ", \"id\": " << (-d.id) << ","
               << "\"ts\": " << std::setprecision(15) << tsc.tsc_to_usec(d.tsc_end) << ",";
            return;
        }

        fw << "{\"ph\": \"X\", \"name\": \"" << d.title << "_" << d.id << "\", \"cat\":\"" << d.cat << "\"," 
           << "\"pid\": " << my_pid << ", \"tid\": " << my_tid << ","
           << "\"ts\": " << std::setprecision(15) << start << ", \"dur\": " << duration << ",";
    }

    void write_event_json(std::ofstream& fw, const ProfileData& d, TscCounter& tsc) const {
        const double duration = tsc.tsc_to_usec(d.tsc_start, d.tsc_end);
        const double start = tsc.tsc_to_usec(d.tsc_start);
        write_event_prefix(fw, d, tsc, start, duration);
        fw << "\"args\":{" << build_args_json(duration, d) << "}},\n";
    }

    void dump_json(std::ofstream& fw, TscCounter& tsc) override {
        if (!enable_dump_json) {
            return;
        }
        auto data_size = all_dump_data.size();
        if (!data_size) {
            return;
        }

        for (auto& d : all_dump_data) {
            write_event_json(fw, d, tsc);
        }
        all_dump_data.clear();
        std::cout << LINUX_PERF_"#" << serial << "(" << this << ") finalize: dumped " << data_size << std::endl;
    }

    uint64_t operator[](size_t i) {
        if (i < events.size()) {
            return values[i];
        } else {
            log_printf(LINUX_PERF_"PerfEventGroup: operator[] with index %lu oveflow (>%lu)\n", i, events.size());
            abort();
        }
        return 0;
    }
    
    PerfEventGroup() = default;

    size_t captured_event_count() const {
        return std::min(events.size(), static_cast<size_t>(ProfileData::data_size));
    }

    bool can_use_rdpmc_fast_path() const {
        return num_events_no_pmc == 0;
    }

    void snapshot_profile_start(ProfileData& profile_data) {
        const size_t num_counters = captured_event_count();
        if (can_use_rdpmc_fast_path()) {
            for (size_t i = 0; i < num_counters; i++) {
                if (events[i].pmc_index) {
                    profile_data.data[i] = read_pmc(events[i].pmc_index - 1);
                }
            }
            return;
        }

        read_group_values();
        for (size_t i = 0; i < num_counters; i++) {
            profile_data.data[i] = values[i];
        }
    }

    uint64_t* snapshot_profile_finish(ProfileData& profile_data, std::map<std::string, uint64_t>* ext_data = nullptr) {
        const size_t num_counters = captured_event_count();

        profile_data.stop();
        if (can_use_rdpmc_fast_path()) {
            for (size_t i = 0; i < num_counters; i++) {
                if (events[i].pmc_index) {
                    profile_data.data[i] = (read_pmc(events[i].pmc_index - 1) - profile_data.data[i]) & pmc_mask;
                } else {
                    profile_data.data[i] = 0;
                }
            }
        } else {
            read_group_values();
            for (size_t i = 0; i < num_counters; i++) {
                profile_data.data[i] = values[i] - profile_data.data[i];
            }
        }

        if (ext_data) {
            (*ext_data)["ns"] = profile_data.tsc_end - profile_data.tsc_start;
            for (size_t i = 0; i < num_counters; i++) {
                (*ext_data)[events[i].name] = profile_data.data[i];
            }
        }

        return profile_data.data;
    }

    void initialize_dump_state() {
        dump_limit = PerfRawConfig::get().dump;
        enable_dump_json = PerfRawConfig::get().dump_on_cpu(sched_getcpu());
        serial = 0;
        if (enable_dump_json) {
            serial = PerfEventJsonDumper::get().register_manager(this);
        }
    }

    void initialize_thread_identity() {
        my_pid = getpid();
        my_tid = gettid();
    }

    struct Config {
        Config(std::string str) {
            parse_from_string(str);
        }
        Config(uint32_t type, uint64_t config, const char * name = "?") : type(type), config(config), name(name) {}

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

        static uint64_t parse_raw_config_values(const std::vector<std::string>& values, const std::string& str) {
            if (values.size() == 1) {
                return std::strtoull(values[0].c_str(), nullptr, 0);
            }
            if (values.size() == 2) {
                return X86_RAW_EVENT(
                        std::strtoull(values[0].c_str(), nullptr, 0),
                        std::strtoull(values[1].c_str(), nullptr, 0), 0);
            }
            if (values.size() == 3) {
                return X86_RAW_EVENT(
                        std::strtoull(values[0].c_str(), nullptr, 0),
                        std::strtoull(values[1].c_str(), nullptr, 0),
                        std::strtoull(values[2].c_str(), nullptr, 0));
            }
            throw std::runtime_error(std::string("Unknown Perf config (too many values): ") + str);
        }

        void parse_raw_config(const std::string& str) {
            type = PERF_TYPE_RAW;
            auto items = str_split(str, "=");
            if (items.size() != 2) {
                throw std::runtime_error(std::string("Unknown Perf config: ") + str);
            }
            name = items[0];
            auto values = str_split(items[1], ",");
            config = parse_raw_config_values(values, str);
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

    void add_configured_event(const Config& config) {
        if (config.type == PERF_TYPE_SOFTWARE) {
            add_sw(config.config);
        } else if (config.type == PERF_TYPE_HARDWARE) {
            add_hw(config.config);
        } else if (config.type == PERF_TYPE_RAW) {
            add_raw(config.config);
        }
        events.back().name = config.name;
        snprintf(events.back().format, sizeof(events.back().format), "%%%lulu, ", config.name.size());
    }

    uint32_t my_pid = 0;
    uint32_t my_tid = 0;

    PerfEventGroup(const std::vector<Config> type_configs, EventArgsSerializer fn = {}) : event_args_serializer(fn) {
        for(auto& tc : type_configs) {
            add_configured_event(tc);
        }

        // env var defined raw events
        for (auto raw_cfg : PerfRawConfig::get().raw_configs) {
            add_configured_event({PERF_TYPE_RAW, raw_cfg.second, raw_cfg.first.c_str()});
        }

        initialize_dump_state();
        initialize_thread_identity();

        enable();
    }

    ~PerfEventGroup() {
        if (enable_dump_json) {
            PerfEventJsonDumper::get().finalize();
        }
        disable();
        release_perf_event_resources(events, sysconf(_SC_PAGESIZE));
    }

    void show_header() {
        std::ostringstream ss;
        ss << "\e[33m";
        ss << "#" << serial << ":";
        for(auto& ev : events) {
            ss << ev.name << ", ";
        }
        ss << "\e[0m\n";
        log_message(ss.str());
    }

    void update_pmc_metadata(CounterEvent& ev) {
        if (ev.pmc_index != 0 || !ev.pmeta->cap_user_rdpmc) {
            return;
        }

        uint32_t seqlock;
        do {
            seqlock = ev.pmeta->lock;
            std::atomic_thread_fence(std::memory_order_seq_cst);
            ev.pmc_index = ev.pmeta->index;
            pmc_width = ev.pmeta->pmc_width;
            pmc_mask = 1;
            pmc_mask = (pmc_mask << pmc_width) - 1;
            if (ev.pmeta->cap_user_time) {
                tsc_time_shift = ev.pmeta->time_shift;
                tsc_time_mult = ev.pmeta->time_mult;
            }
            std::atomic_thread_fence(std::memory_order_seq_cst);
        } while (ev.pmeta->lock != seqlock || (seqlock & 1));
    }

    void reset_and_enable_group() {
        ioctl(group_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(group_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }

    void initialize_rdpmc_state() {
        // PMC index is only valid when being enabled.
        num_events_no_pmc = 0;
        for (auto& ev : events) {
            update_pmc_metadata(ev);
            if (ev.pmc_index == 0) {
                num_events_no_pmc ++;
            }
        }
    }

    void add_raw(uint64_t config, bool pinned=false) {
        // pinned only applies to hardware events and group leaders; keep behavior unchanged.
        perf_event_attr pea = make_perf_event_attr(PERF_TYPE_RAW,
                                                   config,
                                                   true,
                                                   group_fd == -1,
                                                   pinned && group_fd == -1);
        add(&pea);
    }

    void add_hw(uint64_t config, bool pinned=false) {
        perf_event_attr pea = make_perf_event_attr(PERF_TYPE_HARDWARE,
                                                   config,
                                                   true,
                                                   group_fd == -1,
                                                   pinned && group_fd == -1);
        add(&pea);
    }

    void add_sw(uint64_t config) {
        // some SW events are counted in kernel, so keep exclude_kernel=false.
        perf_event_attr pea = make_perf_event_attr(PERF_TYPE_SOFTWARE,
                                                   config,
                                                   false,
                                                   false,
                                                   false);
        add(&pea);
    }

    void prepare_event_attr(perf_event_attr* pev_attr) {
        // clockid must be consistent within group.
        pev_attr->use_clockid = 1;
        // synchronize with clock_gettime(CLOCK_MONOTONIC_RAW)
        pev_attr->clockid = CLOCK_MONOTONIC_RAW;
    }

    int open_event_fd(perf_event_attr* pev_attr, pid_t pid, int cpu) {
        bool has_retried_with_exclude_kernel = false;
        while (true) {
            const int fd = perf_event_open(pev_attr, pid, cpu, group_fd, 0);
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

    perf_event_mmap_page* map_event_metadata(int fd, size_t mmap_length) {
        return map_perf_event_metadata_or_abort(fd, mmap_length, LINUX_PERF_"mmap perf_event_mmap_page failed:");
    }

    void update_group_leader(CounterEvent& ev, const perf_event_attr* pev_attr) {
        if (group_fd == -1) {
            group_fd = ev.fd;
            read_format = pev_attr->read_format;
        }
    }

    void register_event_indices(const perf_event_attr* pev_attr, size_t event_index) {
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_REF_CPU_CYCLES) {
            ref_cpu_cycles_evid = event_index;
        }
        if (pev_attr->type == PERF_TYPE_SOFTWARE && pev_attr->config == PERF_COUNT_SW_TASK_CLOCK) {
            sw_task_clock_evid = event_index;
        }
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_CPU_CYCLES) {
            hw_cpu_cycles_evid = event_index;
        }
        if (pev_attr->type == PERF_TYPE_HARDWARE && pev_attr->config == PERF_COUNT_HW_INSTRUCTIONS) {
            hw_instructions_evid = event_index;
        }
    }

    void finalize_added_event(CounterEvent& ev, const perf_event_attr* pev_attr) {
        update_group_leader(ev, pev_attr);
        register_event_indices(pev_attr, events.size());
        events.push_back(ev);
    }

    void add(perf_event_attr* pev_attr, pid_t pid = 0, int cpu = -1) {
        CounterEvent ev;

        const size_t mmap_length = sysconf(_SC_PAGESIZE) * 1;
        prepare_event_attr(pev_attr);

        ev.fd = open_event_fd(pev_attr, pid, cpu);
        ioctl(ev.fd, PERF_EVENT_IOC_ID, &ev.id);
        ev.pmeta = map_event_metadata(ev.fd, mmap_length);
        finalize_added_event(ev, pev_attr);
    }

    bool event_group_enabled = false;
    uint32_t num_events_no_pmc;

    void enable() {
        if (event_group_enabled) {
            return;
        }
        reset_and_enable_group();
        initialize_rdpmc_state();
        event_group_enabled = true;
    }

    uint64_t tsc2nano(uint64_t cyc) {
        uint64_t quot, rem;
        quot  = cyc >> tsc_time_shift;
        rem   = cyc & (((uint64_t)1 << tsc_time_shift) - 1);
        return quot * tsc_time_mult + ((rem * tsc_time_mult) >> tsc_time_shift);
    }

    void disable() {
        if (!event_group_enabled) {
            return;
        }

        ioctl(group_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

        for(auto& ev : events) {
            ev.pmc_index = 0;
        }
        event_group_enabled = false;
    }

    template<class FN>
    std::vector<uint64_t> rdpmc(FN fn, std::string name = {}, int64_t loop_cnt = 0, std::function<void(uint64_t, uint64_t*, char*&)> addinfo = {}) {
        int cnt = events.size();
        std::vector<uint64_t> pmc(cnt, 0);

        bool use_pmc = (num_events_no_pmc == 0);
        if (use_pmc) {
            for(int i = 0; i < cnt; i++) {
                if (events[i].pmc_index) {
                    pmc[i] = read_pmc(events[i].pmc_index - 1);
                } else {
                    pmc[i] = 0;
                }
            }
        } else {
            read_group_values();
            for(int i = 0; i < cnt; i++) {
                pmc[i] = values[i];
            }
        }

        auto tsc0 = read_tsc();
        fn();
        auto tsc1 = read_tsc();

        if (use_pmc) {
            for(int i = 0; i < cnt; i++) {
                if (events[i].pmc_index) {
                    pmc[i] = (read_pmc(events[i].pmc_index - 1) - pmc[i]) & pmc_mask;
                } else {
                    pmc[i] = 0;
                }
            }
        } else {
            read_group_values();
            for(int i = 0; i < cnt; i++) {
                pmc[i] -= values[i];
            }
        }

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
            for(int i = 0; i < cnt; i++) {
                safe_snprintf(events[i].format, pmc[i]);
            }
            auto duration_ns = tsc2nano(tsc1 - tsc0);
            
            safe_snprintf("\e[0m [%16s] %.3f us", name.c_str(), duration_ns/1e3);
            if (hw_cpu_cycles_evid >= 0) {
                safe_snprintf(" CPU:%.2f(GHz)", 1.0 * pmc[hw_cpu_cycles_evid] / duration_ns);
                if (hw_instructions_evid >= 0) {
                    safe_snprintf(" CPI:%.2f", 1.0 * pmc[hw_cpu_cycles_evid] / pmc[hw_instructions_evid]);
                }
                if (loop_cnt > 0) {
                    safe_snprintf(" CPK:%.1fx%ld", 1.0 * pmc[hw_cpu_cycles_evid] / loop_cnt, loop_cnt);
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

    void read_group_values(bool verbose = false) {
        for(size_t i = 0; i < events.size(); i++) values[i] = 0;

        if (::read(group_fd, read_buf, sizeof(read_buf)) == -1) {
            abort_with_perror(LINUX_PERF_"read perf event failed:");
        }

        uint64_t * readv = read_buf;
        auto nr = *readv++;
        if (verbose) {
            log_printf("number of counters:\t%lu\n", nr);
        }
        time_enabled = 0;
        time_running = 0;
        if (read_format & PERF_FORMAT_TOTAL_TIME_ENABLED) {
            time_enabled = *readv++;
            if (verbose) {
                log_printf("time_enabled:\t%lu\n", time_enabled);
            }
        }
        if (read_format & PERF_FORMAT_TOTAL_TIME_RUNNING) {
            time_running = *readv++;
            if (verbose) {
                log_printf("time_running:\t%lu\n", time_running);
            }
        }

        for (size_t i = 0; i < nr; i++) {
            auto value = *readv++;
            auto id = *readv++;
            for (size_t k = 0; k < events.size(); k++) {
                if (id == events[k].id) {
                    values[k] = value;
                }
            }
        }

        if (verbose) {
            for (size_t k = 0; k < events.size(); k++) {
                log_printf("\t[%lu]: %lu\n", k, values[k]);
            }
        }
    }

    //================================================================================
    // profiler API with json_dump capability
    struct ProfileScope {
        PerfEventGroup* pevg = nullptr;
        ProfileData* pd = nullptr;
        bool do_unlock = false;
        size_t num_events = 0;
        ProfileScope() = default;
        ProfileScope(PerfEventGroup* pevg, ProfileData* pd, bool do_unlock = false) : pevg(pevg), pd(pd), do_unlock(do_unlock) {}

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
                PerfEventGroup::get_sampling_lock() --;
            }
            num_events = 0;
            if (!pevg || !pd) {
                return nullptr;
            }

            num_events = pevg->captured_event_count();
            uint64_t* result = pevg->snapshot_profile_finish(*pd, ext_data);

            pevg = nullptr;
            return result;
        }

        ~ProfileScope() {
            finish();
        }
    };

    ProfileData* begin_profile(const std::string& title, int id = 0, const std::string& cat = "") {
        if (get_sampling_lock().load() != 0) {
            return nullptr;
        }
        if (dump_limit == 0) {
            return nullptr;
        }
        dump_limit --;

        PerfEventContextSwitch::get().updateRingBuffer();

        all_dump_data.emplace_back(title, cat);
        auto* pd = &all_dump_data.back();
        pd->id = id;

        snapshot_profile_start(*pd);

        return pd;
    }

    static PerfEventGroup& get() {
        thread_local PerfEventGroup pevg({
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

    // this lock is global, affect all threads
    static std::atomic_int& get_sampling_lock() {
        static std::atomic_int sampling_lock{0};
        return sampling_lock;
    }
};

using ProfileScope = PerfEventGroup::ProfileScope;

inline bool should_disable_profile(float sampling_probability) {
    return (std::rand() % 1000) * 0.001f >= sampling_probability;
}

inline void lock_sampling_if_needed(bool disable_profile) {
    if (disable_profile) {
        PerfEventGroup::get_sampling_lock() ++;
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
    auto& pevg = PerfEventGroup::get();
    auto* pd = category ? pevg.begin_profile(title, id, *category) : pevg.begin_profile(title, id);
    if (pd) {
        pd->set_extra_datas(std::forward<Args>(args)...);
    }

    bool disable_profile = false;
    if (use_sampling_probability) {
        disable_profile = should_disable_profile(sampling_probability);
        lock_sampling_if_needed(disable_profile);
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
    PerfEventContextSwitch::get();

    // this is for making main threads the first process
    auto dummy = Profile("start");
    return 0;
}

} // namespace LinuxPerf


#ifdef LINUX_PERF_C_API
#include <cstdarg>
inline void fill_profile_data_from_va_list(LinuxPerf::PerfEventGroup::ProfileData* pd, int count, va_list ap) {
    if (!pd) {
        return;
    }
    for (int j = 0; j < count; j++) {
        pd->set_extra_data(j, va_arg(ap, int));
    }
    pd->set_extra_data(count);
}

inline void* start_c_api_profile(const char * title,
                                 const char * category,
                                 float sampling_probability,
                                 bool use_sampling_probability,
                                 int count,
                                 va_list ap) {
    auto& pevg = LinuxPerf::PerfEventGroup::get();
    auto* pd = pevg.begin_profile(title, 0, category);
    fill_profile_data_from_va_list(pd, count, ap);

    bool disable_profile = false;
    if (use_sampling_probability) {
        disable_profile = LinuxPerf::should_disable_profile(sampling_probability);
        LinuxPerf::lock_sampling_if_needed(disable_profile);
    }
    return reinterpret_cast<void*>(new LinuxPerf::ProfileScope{&pevg, pd, disable_profile});
}

extern "C" void* linux_perf_profile_start(const char * title, const char * category, int count, ...) {
    va_list ap;
    va_start(ap, count);
    void* profile = start_c_api_profile(title, category, 0.0f, false, count, ap);
    va_end(ap);
    return profile;
}
extern "C" void* linux_perf_profile_start_prob(const char * title, const char * category, float sampling_probability, int count, ...) {
    va_list ap;
    va_start(ap, count);
    void* profile = start_c_api_profile(title, category, sampling_probability, true, count, ap);
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

