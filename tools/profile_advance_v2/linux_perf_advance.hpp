#pragma once

#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#if defined(__x86_64__)
#include <x86intrin.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace LinuxPerf {

using CounterValue = uint64_t;
using CounterMap = std::map<std::string, CounterValue>;

namespace detail {

// =============================================================================
// 1. Diagnostics and platform primitives
// =============================================================================

[[noreturn]] inline void abort_with_errno(const char* message) {
    const int error = errno;
    std::fprintf(stderr, "[LinuxPerf] %s: %s\n", message, std::strerror(error));
    std::abort();
}

inline int open_perf_event(perf_event_attr* attributes, pid_t pid, int cpu, int group_fd, unsigned long flags = 0) {
    return static_cast<int>(::syscall(__NR_perf_event_open, attributes, pid, cpu, group_fd, flags));
}

inline uint64_t monotonic_time_ns() {
    timespec value{};
    if (::clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
        abort_with_errno("clock_gettime failed");
    }
    return static_cast<uint64_t>(value.tv_sec) * 1000000000ULL + static_cast<uint64_t>(value.tv_nsec);
}

inline uint64_t read_hardware_counter(uint32_t index) {
#if defined(__x86_64__)
    return __rdpmc(index);
#elif defined(__aarch64__)
    (void)index;
    uint64_t value;
    asm volatile("mrs %0, PMCCNTR_EL0" : "=r"(value));
    return value;
#else
    (void)index;
    return 0;
#endif
}

class TimestampConverter {
public:
    TimestampConverter() : base_ns_(monotonic_time_ns()) {}

    double to_microseconds(uint64_t timestamp_ns) const {
        if (timestamp_ns < base_ns_) {
            return 0.0;
        }
        return static_cast<double>(timestamp_ns - base_ns_) / 1000.0;
    }

    static double duration_microseconds(uint64_t begin_ns, uint64_t end_ns) {
        if (end_ns < begin_ns) {
            return 0.0;
        }
        return static_cast<double>(end_ns - begin_ns) / 1000.0;
    }

private:
    uint64_t base_ns_;
};

// =============================================================================
// 2. Small reusable utilities
// =============================================================================

inline std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::string::size_type begin = 0;
    while (true) {
        const std::string::size_type end = text.find(delimiter, begin);
        parts.push_back(text.substr(begin, end - begin));
        if (end == std::string::npos) {
            return parts;
        }
        begin = end + 1;
    }
}

// Reference: manually compose an x86 raw perf config from its fields.
// The active parser accepts the already encoded config directly instead.
// inline uint64_t encode_x86_raw_event(uint64_t event_select,
//                                      uint64_t unit_mask,
//                                      uint64_t counter_mask) {
//     return (counter_mask << 24U) | (unit_mask << 8U) | event_select;
// }

inline uint64_t parse_unsigned(const std::string& text) {
    char* end = nullptr;
    errno = 0;
    const auto value = std::strtoull(text.c_str(), &end, 0);
    if (errno != 0 || end == text.c_str() || *end != '\0') {
        throw std::invalid_argument("invalid unsigned integer: " + text);
    }
    return static_cast<uint64_t>(value);
}

inline uint64_t parse_raw_event(const std::string& text) {
    return parse_unsigned(text);
}

struct NamedEvent {
    const char* name;
    uint32_t type;
    uint64_t config;
};

inline const NamedEvent* find_named_event(const std::string& name) {
    static const NamedEvent events[] = {
        {"CPU_CYCLES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {"HW_CPU_CYCLES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {"cycles", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
        {"INSTRUCTIONS", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {"HW_INSTRUCTIONS", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
        {"CACHE_REFERENCES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
        {"HW_CACHE_REFERENCES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_REFERENCES},
        {"CACHE_MISSES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
        {"HW_CACHE_MISSES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES},
        {"BRANCH_INSTRUCTIONS", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
        {"HW_BRANCH_INSTRUCTIONS", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
        {"BRANCH_MISSES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        {"HW_BRANCH_MISSES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BRANCH_MISSES},
        {"BUS_CYCLES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES},
        {"HW_BUS_CYCLES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_BUS_CYCLES},
        {"STALLED_CYCLES_FRONTEND", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
        {"HW_STALLED_CYCLES_FRONTEND", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_FRONTEND},
        {"STALLED_CYCLES_BACKEND", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
        {"HW_STALLED_CYCLES_BACKEND", PERF_TYPE_HARDWARE, PERF_COUNT_HW_STALLED_CYCLES_BACKEND},
        {"REF_CPU_CYCLES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES},
        {"HW_REF_CPU_CYCLES", PERF_TYPE_HARDWARE, PERF_COUNT_HW_REF_CPU_CYCLES},

        {"CPU_CLOCK", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK},
        {"SW_CPU_CLOCK", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_CLOCK},
        {"TASK_CLOCK", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
        {"SW_TASK_CLOCK", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK},
        {"PAGE_FAULTS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS},
        {"SW_PAGE_FAULTS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS},
        {"pagefaults", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS},
        {"CONTEXT_SWITCHES", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES},
        {"SW_CONTEXT_SWITCHES", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES},
        {"CPU_MIGRATIONS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS},
        {"SW_CPU_MIGRATIONS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CPU_MIGRATIONS},
        {"PAGE_FAULTS_MIN", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN},
        {"SW_PAGE_FAULTS_MIN", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MIN},
        {"PAGE_FAULTS_MAJ", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ},
        {"SW_PAGE_FAULTS_MAJ", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS_MAJ},
        {"ALIGNMENT_FAULTS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_ALIGNMENT_FAULTS},
        {"SW_ALIGNMENT_FAULTS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_ALIGNMENT_FAULTS},
        {"EMULATION_FAULTS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_EMULATION_FAULTS},
        {"SW_EMULATION_FAULTS", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_EMULATION_FAULTS},
        {"DUMMY", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_DUMMY},
        {"SW_DUMMY", PERF_TYPE_SOFTWARE, PERF_COUNT_SW_DUMMY},
    };

    for (const NamedEvent& event : events) {
        if (name == event.name) {
            return &event;
        }
    }
    return nullptr;
}

inline void write_json_string(std::ostream& output, const std::string& value) {
    output << '"';
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\b': output << "\\b"; break;
        case '\f': output << "\\f"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20U) {
                const char hex[] = "0123456789abcdef";
                output << "\\u00" << hex[character >> 4U] << hex[character & 0x0fU];
            } else {
                output << static_cast<char>(character);
            }
        }
    }
    output << '"';
}

inline perf_event_attr make_event_attributes(uint32_t type,
                                             uint64_t config,
                                             bool exclude_kernel,
                                             bool include_running_times,
                                             bool pinned) {
    perf_event_attr pea{};
    pea.type = type;
    pea.size = sizeof(pea);
    pea.config = config;
    pea.disabled = 1;
    pea.exclude_kernel = exclude_kernel ? 1U : 0U;
    pea.exclude_hv = 1;
    pea.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
    if (include_running_times) {
        pea.read_format |= PERF_FORMAT_TOTAL_TIME_ENABLED
                        | PERF_FORMAT_TOTAL_TIME_RUNNING;
    }
    pea.pinned = pinned ? 1U : 0U;
    return pea;
}

inline size_t system_page_size() {
    const long value = ::sysconf(_SC_PAGESIZE);
    if (value <= 0) {
        abort_with_errno("[LINUX_PERF] sysconf(_SC_PAGESIZE) failed");
    }
    return static_cast<size_t>(value);
}

class MappedPerfPage {
public:
    MappedPerfPage() = default;

    MappedPerfPage(int fd, size_t length) { map(fd, length); }

    MappedPerfPage(const MappedPerfPage&) = delete;
    MappedPerfPage& operator=(const MappedPerfPage&) = delete;

    MappedPerfPage(MappedPerfPage&& other) noexcept
        : page_(other.page_), length_(other.length_) {
        other.page_ = nullptr;
        other.length_ = 0;
    }

    MappedPerfPage& operator=(MappedPerfPage&& other) noexcept {
        if (this != &other) {
            reset();
            page_ = other.page_;
            length_ = other.length_;
            other.page_ = nullptr;
            other.length_ = 0;
        }
        return *this;
    }

    ~MappedPerfPage() { reset(); }

    perf_event_mmap_page* get() const { return page_; }

private:
    void map(int fd, size_t length) {
        void* address = ::mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (address == MAP_FAILED) {
            abort_with_errno("mmap perf page failed");
        }
        page_ = static_cast<perf_event_mmap_page*>(address);
        length_ = length;
    }

    void reset() {
        if (page_ != nullptr) {
            ::munmap(page_, length_);
            page_ = nullptr;
            length_ = 0;
        }
    }

    perf_event_mmap_page* page_ = nullptr;
    size_t length_ = 0;
};

class AtomicTryLock {
public:
    explicit AtomicTryLock(std::atomic_flag& flag)
        : flag_(flag), owns_lock_(!flag_.test_and_set(std::memory_order_acquire)) {}

    ~AtomicTryLock() {
        if (owns_lock_) {
            flag_.clear(std::memory_order_release);
        }
    }

    bool owns_lock() const { return owns_lock_; }

private:
    std::atomic_flag& flag_;
    bool owns_lock_;
};

// =============================================================================
// 3. Configuration and event descriptions
// =============================================================================

struct EventSpec {
    uint32_t type;
    uint64_t config;
    std::string name;

    EventSpec(uint32_t event_type, uint64_t event_config, std::string event_name)
        : type(event_type), config(event_config), name(std::move(event_name)) {}
};

inline EventSpec parse_event_spec(const std::string& description) {
    const NamedEvent* event = find_named_event(description);
    if (event == nullptr) {
        return EventSpec(PERF_TYPE_RAW, parse_raw_event(description), description);
    }

    switch (event->type) {
    case PERF_TYPE_HARDWARE:
        return EventSpec(PERF_TYPE_HARDWARE, event->config, description);
    case PERF_TYPE_SOFTWARE:
        return EventSpec(PERF_TYPE_SOFTWARE, event->config, description);
    default:
        throw std::logic_error("unknown named event type");
    }
}

class Configuration {
public:
    static const Configuration& instance() {
        static const Configuration configuration;
        return configuration;
    }

    bool trace_enabled() const { return dump_limit_ != 0; }
    bool context_switch_enabled() const { return context_switch_enabled_; }
    int64_t dump_limit() const { return dump_limit_; }
    const cpu_set_t& cpu_mask() const { return cpu_mask_; }
    const std::vector<EventSpec>& custom_events() const { return custom_events_; }

    bool trace_allowed_on_cpu(int cpu) const {
        return !CPU_COUNT(&cpu_mask_) || CPU_ISSET(cpu, &cpu_mask_);
    }

private:
    Configuration() {
        CPU_ZERO(&cpu_mask_);
        const char* environment = std::getenv("LINUX_PERF");
        if (environment == nullptr || *environment == '\0') {
            return;
        }
        for (const std::string& option : split(environment, ':')) {
            parse_option(option);
        }
    }

    void parse_option(const std::string& option) {
        const std::string::size_type equals = option.find('=');
        std::string key = option;
        std::string value = option;
        if (equals != std::string::npos) {
            key = option.substr(0, equals);
            value = option.substr(equals + 1);
        }

        // parse dump
        if (key == "dump") {
            if (equals == std::string::npos) {
                dump_limit_ = std::numeric_limits<int64_t>::max();
            } else {
                dump_limit_ = static_cast<int64_t>(parse_unsigned(value));
            }
            return;
        }

        // parse switch-cpu
        if (key == "switch-cpu") {
            context_switch_enabled_ = true;
            if (::sched_getaffinity(0, sizeof(cpu_mask_), &cpu_mask_) != 0) {
                abort_with_errno("sched_getaffinity failed");
            }
            return;
        }

        // parse cpus
        if (key == "cpus") {
            CPU_ZERO(&cpu_mask_);
            for (const std::string& cpu : split(value, ',')) {
                CPU_SET(static_cast<int>(parse_unsigned(cpu)), &cpu_mask_);
            }
            return;
        }

        if (equals != std::string::npos) {
            throw std::invalid_argument(
                "custom event aliases are not supported: " + option);
        }

        // parse custom event
        const EventSpec event = parse_event_spec(option);
        custom_events_.push_back(event);
    }

    int64_t dump_limit_ = 0;
    bool context_switch_enabled_ = false;
    cpu_set_t cpu_mask_{};
    std::vector<EventSpec> custom_events_;
};

// =============================================================================
// 4. Trace model and writer
// =============================================================================

class TraceSink {
public:
    virtual ~TraceSink() = default;
    virtual void write_trace_events(std::ostream& output,
                                    const TimestampConverter& timestamps,
                                    bool& first_event) = 0;
};

class TraceWriter {
public:
    static TraceWriter& instance() {
        static TraceWriter writer;
        return writer;
    }

    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;

    void register_sink(TraceSink* sink) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (std::find(sinks_.begin(), sinks_.end(), sink) == sinks_.end()) {
            sinks_.push_back(sink);
        }
    }

    void unregister_sink(TraceSink* sink) {
        std::lock_guard<std::mutex> lock(mutex_);
        sinks_.erase(std::remove(sinks_.begin(), sinks_.end(), sink), sinks_.end());
    }

    void flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (sinks_.empty()) {
            return;
        }

        std::ofstream output("perf_dump.json", std::ios::out | std::ios::trunc);
        if (!output) {
            abort_with_errno("open perf_dump.json failed");
        }

        output << "{\n\"schemaVersion\":1,\n\"traceEvents\":[\n";
        bool first_event = true;
        for (TraceSink* sink : sinks_) {
            sink->write_trace_events(output, timestamps_, first_event);
        }
        output << "\n]}\n";
    }

private:
    TraceWriter() = default;
    ~TraceWriter() { flush(); }

    std::mutex mutex_;
    std::vector<TraceSink*> sinks_;
    TimestampConverter timestamps_;
};

template <size_t Capacity>
class ExtraArguments {
public:
    ExtraArguments() = default;

    template <typename... Values>
    void assign(Values&&... values) {
        size_ = 0;
        const int unpack[] = {0, (append(std::forward<Values>(values)), 0)...};
        (void)unpack;
    }

    void write_json(std::ostream& output) const {
        if (size_ == 0) {
            return;
        }
        output << ",\"Extra Data\":[";
        for (size_t index = 0; index < size_; ++index) {
            if (index != 0) {
                output << ',';
            }
            output << values_[index];
        }
        output << ']';
    }

private:
    template <typename T>
    typename std::enable_if<std::is_integral<T>::value>::type append(T value) {
        append_number(static_cast<long double>(value));
    }

    template <typename T>
    typename std::enable_if<std::is_floating_point<T>::value>::type append(T value) {
        append_number(static_cast<long double>(value));
    }

    template <typename T>
    void append(T* pointer) {
        std::ostringstream text;
        text << '"' << static_cast<const void*>(pointer) << '"';
        append_text(text.str());
    }

    template <typename T>
    void append(const std::vector<T>& values) {
        std::ostringstream text;
        text << '"' << '(';
        for (size_t index = 0; index < values.size(); ++index) {
            if (index != 0) {
                text << ',';
            }
            text << values[index];
        }
        text << ')' << '"';
        append_text(text.str());
    }

    void append_number(long double value) {
        std::ostringstream text;
        text << value;
        append_text(text.str());
    }

    void append_text(const std::string& value) {
        if (size_ >= Capacity) {
            throw std::length_error("too many profile extra arguments");
        }
        values_[size_++] = value;
    }

    std::array<std::string, Capacity> values_{};
    size_t size_ = 0;
};

struct Snapshot {
    static constexpr size_t kMaxCounters = 16;

    uint64_t begin_ns = 0;
    uint64_t end_ns = 0;
    std::string title;
    std::string category;
    int32_t id = 0;
    std::array<CounterValue, kMaxCounters> counters{};
    ExtraArguments<16> extra_arguments;
};

// =============================================================================
// 5. Optional CPU-wide context-switch timeline
// =============================================================================

class RingBufferReader {
public:
    RingBufferReader(const perf_event_mmap_page& metadata, uint64_t offset)
        : metadata_(metadata), offset_(offset) {}

    template <typename T>
    T read() {
        T value{};
        copy_bytes(&value, sizeof(value));
        return value;
    }

    uint64_t offset() const { return offset_; }

private:
    void copy_bytes(void* destination, size_t byte_count) {
        const uint8_t* data = reinterpret_cast<const uint8_t*>(&metadata_)
                            + metadata_.data_offset;
        const size_t begin = static_cast<size_t>(offset_ % metadata_.data_size);
        const size_t first = std::min(byte_count,
                                      static_cast<size_t>(metadata_.data_size) - begin);
        std::memcpy(destination, data + begin, first);
        if (first < byte_count) {
            std::memcpy(static_cast<uint8_t*>(destination) + first,
                        data,
                        byte_count - first);
        }
        offset_ += byte_count;
    }

    const perf_event_mmap_page& metadata_;
    uint64_t offset_;
};

struct ContextSwitchRecord {
    uint32_t type = 0;
    uint16_t misc = 0;
    uint16_t size = 0;
    uint32_t pid = 0;
    uint32_t tid = 0;
    uint64_t time_ns = 0;
    uint32_t cpu = 0;

    static ContextSwitchRecord read_from(RingBufferReader& reader) {
        ContextSwitchRecord record;
        record.type = reader.read<uint32_t>();
        record.misc = reader.read<uint16_t>();
        record.size = reader.read<uint16_t>();
        if (record.type == PERF_RECORD_SWITCH_CPU_WIDE) {
            (void)reader.read<uint32_t>();
            (void)reader.read<uint32_t>();
        }
        record.pid = reader.read<uint32_t>();
        record.tid = reader.read<uint32_t>();
        record.time_ns = reader.read<uint64_t>();
        record.cpu = reader.read<uint32_t>();
        (void)reader.read<uint32_t>();
        return record;
    }
};

class ContextSwitchTracker : public TraceSink {
public:
    static ContextSwitchTracker& instance() {
        static ContextSwitchTracker tracker;
        return tracker;
    }

    void drain_if_needed() {
        if (!enabled_) {
            return;
        }
        AtomicTryLock lock(drain_lock_);
        if (!lock.owns_lock()) {
            return;
        }
        for (CpuMonitor& monitor : monitors_) {
            drain(monitor, false);
        }
    }

    void write_trace_events(std::ostream& output,
                            const TimestampConverter& timestamps,
                            bool& first_event) override {
        if (!enabled_) {
            return;
        }
        for (CpuMonitor& monitor : monitors_) {
            drain(monitor, true);
        }
        for (const TimeSlice& slice : slices_) {
            if (!first_event) {
                output << ",\n";
            }
            first_event = false;
                        output << "{\"ph\":\"X\",\"name\":";
                        write_json_string(output, std::to_string(slice.tid));
                        output << ",\"cat\":\"TID\",\"pid\":9999,\"tid\":";
                        write_json_string(output, "CPU" + std::to_string(slice.cpu));
                        output << ",\"ts\":"
                   << std::setprecision(15) << timestamps.to_microseconds(slice.begin_ns)
                   << ",\"dur\":"
                   << TimestampConverter::duration_microseconds(slice.begin_ns, slice.end_ns)
                   << '}';
        }
        slices_.clear();
    }

private:
    struct CpuMonitor {
        int fd = -1;
        int cpu = -1;
        MappedPerfPage mapping;
        uint64_t running_since_ns = 0;
        uint32_t running_tid = 0;

        CpuMonitor(int event_fd, int cpu_id, size_t mapping_size)
            : fd(event_fd), cpu(cpu_id), mapping(event_fd, mapping_size) {}

        CpuMonitor(CpuMonitor&&) = default;
        CpuMonitor& operator=(CpuMonitor&&) = default;
        CpuMonitor(const CpuMonitor&) = delete;
        CpuMonitor& operator=(const CpuMonitor&) = delete;

        ~CpuMonitor() {
            if (fd >= 0) {
                ::close(fd);
            }
        }
    };

    struct TimeSlice {
        uint64_t begin_ns;
        uint64_t end_ns;
        uint32_t tid;
        uint32_t cpu;
    };

    ContextSwitchTracker() {
        const Configuration& config = Configuration::instance();
        enabled_ = config.context_switch_enabled();
        if (!enabled_) {
            return;
        }

        const cpu_set_t& mask = config.cpu_mask();
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (CPU_ISSET(cpu, &mask)) {
                open_monitor(cpu);
            }
        }
        TraceWriter::instance().register_sink(this);
    }

    ~ContextSwitchTracker() {
        if (enabled_) {
            TraceWriter::instance().flush();
            TraceWriter::instance().unregister_sink(this);
        }
    }

    void open_monitor(int cpu) {
        perf_event_attr attributes = make_event_attributes(PERF_TYPE_SOFTWARE,
                                                           PERF_COUNT_SW_DUMMY,
                                                           true,
                                                           false,
                                                           false);
        attributes.disabled = 0;
        attributes.context_switch = 1;
        attributes.sample_id_all = 1;
        attributes.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_TID | PERF_SAMPLE_CPU;
        attributes.use_clockid = 1;
        attributes.clockid = CLOCK_MONOTONIC_RAW;

        const int fd = open_perf_event(&attributes, -1, cpu, -1);
        if (fd < 0) {
            abort_with_errno("open CPU-wide context switch event failed");
        }
        monitors_.emplace_back(fd, cpu, ring_mapping_size());
    }

    static size_t ring_mapping_size() {
        return system_page_size() * 1025U;
    }

    void drain(CpuMonitor& monitor, bool flush_active) {
        perf_event_mmap_page& metadata = *monitor.mapping.get();
        const uint64_t head = __atomic_load_n(&metadata.data_head, __ATOMIC_ACQUIRE);
        uint64_t tail = metadata.data_tail;

        while (tail < head) {
            RingBufferReader reader(metadata, tail);
            const ContextSwitchRecord record = ContextSwitchRecord::read_from(reader);
            if (record.size == 0 || tail + record.size > head) {
                break;
            }
            process_record(monitor, record);
            tail += record.size;
        }

        __atomic_store_n(&metadata.data_tail, tail, __ATOMIC_RELEASE);
        if (flush_active && monitor.running_since_ns != 0) {
            slices_.push_back({monitor.running_since_ns,
                               monotonic_time_ns(),
                               monitor.running_tid,
                               static_cast<uint32_t>(monitor.cpu)});
            monitor.running_since_ns = 0;
        }
    }

    void process_record(CpuMonitor& monitor, const ContextSwitchRecord& record) {
        if (record.type != PERF_RECORD_SWITCH_CPU_WIDE || record.tid == 0) {
            return;
        }
        const bool switch_out = (record.misc & PERF_RECORD_MISC_SWITCH_OUT) != 0;
        if (switch_out) {
            if (monitor.running_since_ns != 0) {
                slices_.push_back({monitor.running_since_ns,
                                   record.time_ns,
                                   record.tid,
                                   record.cpu});
            }
            monitor.running_since_ns = 0;
        } else {
            monitor.running_since_ns = record.time_ns;
            monitor.running_tid = record.tid;
        }
    }

    bool enabled_ = false;
    std::vector<CpuMonitor> monitors_;
    std::deque<TimeSlice> slices_;
    std::atomic_flag drain_lock_ = ATOMIC_FLAG_INIT;
};

// =============================================================================
// 6. Per-thread counter group
// =============================================================================

class CounterGroup : public TraceSink {
public:
    explicit CounterGroup(std::vector<EventSpec> events)
        : remaining_trace_quota_(Configuration::instance().dump_limit()) {
        const Configuration& config = Configuration::instance();
        events.insert(events.end(), config.custom_events().begin(), config.custom_events().end());
        for (const EventSpec& event : events) {
            open_counter(event);
        }
        enable();

        trace_enabled_ = config.trace_enabled()
                      && config.trace_allowed_on_cpu(::sched_getcpu());
        if (trace_enabled_) {
            TraceWriter::instance().register_sink(this);
        }
    }

    CounterGroup(const CounterGroup&) = delete;
    CounterGroup& operator=(const CounterGroup&) = delete;

    ~CounterGroup() {
        if (trace_enabled_) {
            TraceWriter::instance().flush();
            TraceWriter::instance().unregister_sink(this);
        }
        disable();
        for (Counter& counter : counters_) {
            if (counter.fd >= 0) {
                ::close(counter.fd);
            }
        }
    }

    static CounterGroup& thread_instance() {
        thread_local CounterGroup group({
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES, "HW_CPU_CYCLES"},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS, "HW_INSTRUCTIONS"},
            {PERF_TYPE_HARDWARE, PERF_COUNT_HW_CACHE_MISSES, "HW_CACHE_MISSES"},
            {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_CONTEXT_SWITCHES, "SW_CONTEXT_SWITCHES"},
            {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_TASK_CLOCK, "SW_TASK_CLOCK"},
            {PERF_TYPE_SOFTWARE, PERF_COUNT_SW_PAGE_FAULTS, "SW_PAGE_FAULTS"},
        });
        return group;
    }

    size_t counter_count() const {
        return std::min(counters_.size(), Snapshot::kMaxCounters);
    }

    void begin_snapshot(Snapshot& snapshot) {
        ContextSwitchTracker::instance().drain_if_needed();
        snapshot.begin_ns = monotonic_time_ns();
        read_current_values(snapshot.counters);
    }

    CounterValue* finish_snapshot(Snapshot& snapshot, CounterMap* output_values) {
        std::array<CounterValue, Snapshot::kMaxCounters> end_values{};
        read_current_values(end_values);
        snapshot.end_ns = monotonic_time_ns();

        const size_t count = counter_count();
        for (size_t index = 0; index < count; ++index) {
            snapshot.counters[index] = end_values[index] - snapshot.counters[index];
            if (output_values != nullptr) {
                (*output_values)[counters_[index].spec.name] = snapshot.counters[index];
            }
        }
        if (output_values != nullptr) {
            (*output_values)["ns"] = snapshot.end_ns - snapshot.begin_ns;
        }

        if (should_record_trace()) {
            completed_snapshots_.push_back(snapshot);
        }
        return snapshot.counters.data();
    }

    void write_trace_events(std::ostream& output,
                            const TimestampConverter& timestamps,
                            bool& first_event) override {
        for (const Snapshot& snapshot : completed_snapshots_) {
            if (!first_event) {
                output << ",\n";
            }
            first_event = false;
            write_snapshot(output, timestamps, snapshot);
        }
        completed_snapshots_.clear();
    }

private:
    struct Counter {
        EventSpec spec;
        int fd = -1;
        uint64_t id = 0;
        MappedPerfPage metadata;

        explicit Counter(EventSpec value) : spec(std::move(value)) {}
        Counter(Counter&&) = default;
        Counter& operator=(Counter&&) = default;
        Counter(const Counter&) = delete;
        Counter& operator=(const Counter&) = delete;
    };

    void open_counter(const EventSpec& spec) {
        const bool first = counters_.empty();
        perf_event_attr attributes = make_event_attributes(spec.type,
                                                           spec.config,
                                                           spec.type != PERF_TYPE_SOFTWARE,
                                                           first,
                                                           false);
        attributes.use_clockid = 1;
        attributes.clockid = CLOCK_MONOTONIC_RAW;

        int fd = open_perf_event(&attributes, 0, -1, leader_fd_);
        if (fd < 0 && !attributes.exclude_kernel) {
            attributes.exclude_kernel = 1;
            fd = open_perf_event(&attributes, 0, -1, leader_fd_);
        }
        if (fd < 0) {
            abort_with_errno("open perf counter failed");
        }

        Counter counter(spec);
        counter.fd = fd;
        if (::ioctl(fd, PERF_EVENT_IOC_ID, &counter.id) != 0) {
            ::close(fd);
            abort_with_errno("PERF_EVENT_IOC_ID failed");
        }
        counter.metadata = MappedPerfPage(fd, system_page_size());

        if (first) {
            leader_fd_ = fd;
            read_format_ = attributes.read_format;
        }
        counters_.push_back(std::move(counter));
    }

    void enable() {
        if (leader_fd_ < 0) {
            return;
        }
        if (::ioctl(leader_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0
            || ::ioctl(leader_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
            abort_with_errno("enable perf group failed");
        }
        refresh_rdpmc_metadata();
        enabled_ = true;
    }

    void disable() {
        if (enabled_) {
            ::ioctl(leader_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
            enabled_ = false;
        }
    }

    void refresh_rdpmc_metadata() {
        all_counters_support_rdpmc_ = !counters_.empty();
        for (Counter& counter : counters_) {
            perf_event_mmap_page* metadata = counter.metadata.get();
            if (!metadata->cap_user_rdpmc) {
                all_counters_support_rdpmc_ = false;
            }
        }
    }

    void read_current_values(std::array<CounterValue, Snapshot::kMaxCounters>& values) {
        values.fill(0);
        if (all_counters_support_rdpmc_ && try_read_with_rdpmc(values)) {
            return;
        }
        read_with_syscall(values);
    }

    static bool try_read_one_with_rdpmc(const perf_event_mmap_page& metadata,
                                        CounterValue& value) {
        uint32_t sequence;
        uint32_t index;
        uint16_t width;
        int64_t offset;
        uint64_t enabled;
        uint64_t running;
        do {
            sequence = __atomic_load_n(&metadata.lock, __ATOMIC_ACQUIRE);
            if ((sequence & 1U) != 0) {
                continue;
            }
            index = metadata.index;
            width = metadata.pmc_width;
            offset = metadata.offset;
            enabled = metadata.time_enabled;
            running = metadata.time_running;
            if (index == 0 || enabled != running) {
                return false;
            }

            uint64_t raw = read_hardware_counter(index - 1U);
            if (width != 0 && width < 64) {
                const uint32_t shift = 64U - width;
                raw = static_cast<uint64_t>(
                    static_cast<int64_t>(raw << shift) >> shift);
            }
            value = static_cast<uint64_t>(offset) + raw;
            std::atomic_thread_fence(std::memory_order_acquire);
        } while (__atomic_load_n(&metadata.lock, __ATOMIC_RELAXED) != sequence);
        return true;
    }

    bool try_read_with_rdpmc(
        std::array<CounterValue, Snapshot::kMaxCounters>& values) {
        const size_t count = counter_count();
        for (size_t index = 0; index < count; ++index) {
            if (!try_read_one_with_rdpmc(*counters_[index].metadata.get(), values[index])) {
                return false;
            }
        }
        return true;
    }

    void read_with_syscall(std::array<CounterValue, Snapshot::kMaxCounters>& values) {
        const size_t count = counters_.size();
        const size_t header_words = 1
                                  + ((read_format_ & PERF_FORMAT_TOTAL_TIME_ENABLED) ? 1 : 0)
                                  + ((read_format_ & PERF_FORMAT_TOTAL_TIME_RUNNING) ? 1 : 0);
        const size_t words = header_words + count * 2U;
        std::vector<uint64_t> buffer(words);

        const size_t expected_bytes = words * sizeof(uint64_t);
        const ssize_t bytes_read = ::read(leader_fd_, buffer.data(), expected_bytes);
        if (bytes_read < 0) {
            abort_with_errno("read perf group failed");
        }
        if (static_cast<size_t>(bytes_read) < sizeof(uint64_t)) {
            throw std::runtime_error("perf group returned a truncated header");
        }

        const uint64_t event_count = buffer[0];
        const size_t returned_words = static_cast<size_t>(bytes_read) / sizeof(uint64_t);
        if (event_count > count || header_words + event_count * 2U > returned_words) {
            throw std::runtime_error("perf group returned malformed data");
        }

        size_t cursor = header_words;
        for (uint64_t item = 0; item < event_count; ++item) {
            const uint64_t value = buffer[cursor++];
            const uint64_t id = buffer[cursor++];
            for (size_t index = 0; index < counter_count(); ++index) {
                if (counters_[index].id == id) {
                    values[index] = value;
                    break;
                }
            }
        }
    }

    bool should_record_trace() {
        if (!trace_enabled_ || remaining_trace_quota_ == 0) {
            return false;
        }
        --remaining_trace_quota_;
        return true;
    }

    void write_snapshot(std::ostream& output,
                        const TimestampConverter& timestamps,
                        const Snapshot& snapshot) const {
        const double duration = TimestampConverter::duration_microseconds(
            snapshot.begin_ns, snapshot.end_ns);
        output << "{\"ph\":\"X\",\"name\":";
        write_json_string(output, snapshot.title + '_' + std::to_string(snapshot.id));
        output << ",\"cat\":";
        write_json_string(output, snapshot.category);
        output << ",\"pid\":" << ::getpid() << ",\"tid\":"
               << static_cast<uint64_t>(::syscall(SYS_gettid)) << ",\"ts\":"
               << std::setprecision(15) << timestamps.to_microseconds(snapshot.begin_ns)
               << ",\"dur\":" << duration << ",\"args\":{";

        for (size_t index = 0; index < counter_count(); ++index) {
            if (index != 0) {
                output << ',';
            }
              write_json_string(output, counters_[index].spec.name);
              output << ":\"" << snapshot.counters[index] << '"';
        }
        snapshot.extra_arguments.write_json(output);
        output << "}}";
    }

    int leader_fd_ = -1;
    uint64_t read_format_ = 0;
    bool enabled_ = false;
    bool all_counters_support_rdpmc_ = false;
    bool trace_enabled_ = false;
    int64_t remaining_trace_quota_ = 0;
    std::vector<Counter> counters_;
    std::deque<Snapshot> completed_snapshots_;
};

}  // namespace detail

// =============================================================================
// 7. Public C++ API
// =============================================================================

class ProfileScope {
public:
    ProfileScope() = default;

    ProfileScope(detail::CounterGroup* group,
                 std::string title,
                 std::string category,
                 int id)
        : group_(group) {
        snapshot_.title = std::move(title);
        snapshot_.category = std::move(category);
        snapshot_.id = id;
        if (group_ != nullptr) {
            group_->begin_snapshot(snapshot_);
        }
    }

    ProfileScope(const ProfileScope&) = delete;
    ProfileScope& operator=(const ProfileScope&) = delete;

    ProfileScope(ProfileScope&& other) noexcept
        : group_(other.group_),
          snapshot_(std::move(other.snapshot_)),
          counter_count_(other.counter_count_) {
        other.group_ = nullptr;
        other.counter_count_ = 0;
    }

    ProfileScope& operator=(ProfileScope&& other) noexcept {
        if (this != &other) {
            finish();
            group_ = other.group_;
            snapshot_ = std::move(other.snapshot_);
            counter_count_ = other.counter_count_;
            other.group_ = nullptr;
            other.counter_count_ = 0;
        }
        return *this;
    }

    ~ProfileScope() { finish(); }

    CounterValue* finish(CounterMap* output_values = nullptr) {
        if (group_ == nullptr) {
            return nullptr;
        }
        counter_count_ = group_->counter_count();
        CounterValue* result = group_->finish_snapshot(snapshot_, output_values);
        group_ = nullptr;
        return result;
    }

    size_t counter_count() const { return counter_count_; }

    template <typename... Values>
    void set_extra_arguments(Values&&... values) {
        snapshot_.extra_arguments.assign(std::forward<Values>(values)...);
    }

private:
    detail::CounterGroup* group_ = nullptr;
    detail::Snapshot snapshot_;
    size_t counter_count_ = 0;
};

template <typename... Values>
ProfileScope make_profile_scope(const std::string& title,
                                const std::string& category,
                                int id,
                                Values&&... values) {
    detail::CounterGroup& group = detail::CounterGroup::thread_instance();
    ProfileScope scope(&group, title, category, id);
    scope.set_extra_arguments(std::forward<Values>(values)...);
    return scope;
}

template <typename... Values>
ProfileScope Profile(const std::string& title, int id = 0, Values&&... values) {
    return make_profile_scope(title,
                              std::string(),
                              id,
                              std::forward<Values>(values)...);
}

template <typename... Values>
ProfileScope Profile(const std::string& title,
                     const std::string& category,
                     int id = 0,
                     Values&&... values) {
    return make_profile_scope(title,
                              category,
                              id,
                              std::forward<Values>(values)...);
}

inline int Init() {
    (void)detail::Configuration::instance();
    (void)detail::ContextSwitchTracker::instance();
    (void)detail::CounterGroup::thread_instance();
    return 0;
}

}  // namespace LinuxPerf

// Use once at the beginning of a C++ scope. Destruction ends profiling.
#define LINUX_PERF_PROFILE(title) \
    const auto linux_perf_scope = ::LinuxPerf::Profile((title))

