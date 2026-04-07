// simple_perf.h — Standalone, single-header Linux perf_event wrapper.
// No external dependencies. C++11 or later.
//
// Usage:
//   #include "simple_perf.h"
//   SimplePerf perf;                       // default: cycles + instructions
//   perf.start();
//   /* ... work ... */
//   auto& evs = perf.stop();               // returns map<string, uint64_t>
//   printf("cycles = %lu\n", evs.at("CPU_CYCLES"));
//
// See README.md for full documentation.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

// ---- internal assertion (prints location, triggers debugger trap) ----------
namespace simple_perf_detail {

[[noreturn]] inline void assert_fail(const char* file, int line, const char* expr, const char* msg = nullptr) {
    fprintf(stderr, "\033[31m%s:%d  %s failed", file, line, expr);
    if (msg) fprintf(stderr, ": %s", msg);
    fprintf(stderr, "\033[0m\n");
#if defined(__x86_64__)
    __asm__ volatile("int3");
#elif defined(__aarch64__)
    __asm__ volatile("brk #0x1");
#endif
    abort();
}

} // namespace simple_perf_detail

#define SPERF_ASSERT(cond, ...) \
    do { if (!(cond)) simple_perf_detail::assert_fail(__FILE__, __LINE__, #cond, ##__VA_ARGS__); } while (0)

// ---------------------------------------------------------------------------
// SimplePerf — lightweight perf_event group reader
// ---------------------------------------------------------------------------
struct SimplePerf {

    // ---------- event descriptor ----------------------------------------
    struct Event {
        std::string name;
        uint32_t    type   = 0;
        uint64_t    config = 0;
        int         fd     = -1;
        uint64_t    id     = 0;
        uint64_t    value  = 0;
        perf_event_attr attr{};

        bool is_cycles()       const { return type == PERF_TYPE_HARDWARE && config == PERF_COUNT_HW_CPU_CYCLES; }
        bool is_instructions() const { return type == PERF_TYPE_HARDWARE && config == PERF_COUNT_HW_INSTRUCTIONS; }

        // Construct from a mnemonic name or a raw hex config.
        //   Mnemonics: "CPU_CYCLES"/"C", "INSTRUCTIONS"/"I",
        //              "STALLED_CYCLES_FRONTEND", "STALLED_CYCLES_BACKEND",
        //              "CACHE_MISSES", "BRANCH_MISSES"
        //   Otherwise: treated as PERF_TYPE_RAW with the supplied raw_config.
        Event(const std::string& _name, uint64_t raw_config = 0)
            : name(_name), config(raw_config)
        {
            resolve_type_config();
            init_attr();
        }

    private:
        void resolve_type_config() {
            if (name == "CPU_CYCLES"  || name == "C") {
                type = PERF_TYPE_HARDWARE; config = PERF_COUNT_HW_CPU_CYCLES;
            } else if (name == "INSTRUCTIONS" || name == "I") {
                type = PERF_TYPE_HARDWARE; config = PERF_COUNT_HW_INSTRUCTIONS;
            } else if (name == "STALLED_CYCLES_FRONTEND") {
                type = PERF_TYPE_HARDWARE; config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
            } else if (name == "STALLED_CYCLES_BACKEND") {
                type = PERF_TYPE_HARDWARE; config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
            } else if (name == "CACHE_MISSES") {
                type = PERF_TYPE_HARDWARE; config = PERF_COUNT_HW_CACHE_MISSES;
            } else if (name == "BRANCH_MISSES") {
                type = PERF_TYPE_HARDWARE; config = PERF_COUNT_HW_BRANCH_MISSES;
            } else {
                type = PERF_TYPE_RAW;
                // config already set from raw_config
            }
        }

        void init_attr() {
            memset(&attr, 0, sizeof(attr));
            attr.type           = type;
            attr.size           = sizeof(attr);
            attr.config         = config;
            attr.read_format    = PERF_FORMAT_GROUP | PERF_FORMAT_ID;
            attr.disabled       = 1;
            attr.exclude_kernel = 1;
            attr.exclude_hv     = 1;
        }
    };

    // ---------- construction --------------------------------------------

    // Default: cycles + instructions.
    SimplePerf() : SimplePerf({{"C", 0}, {"I", 0}}) {}

    explicit SimplePerf(const std::vector<Event>& events) : m_events(events) {
        SPERF_ASSERT(!m_events.empty(), "at least one event required");
        open_events();
    }

    SimplePerf(std::initializer_list<Event> events) : SimplePerf(std::vector<Event>(events)) {}

    ~SimplePerf() {
        for (auto& e : m_events)
            if (e.fd >= 0) close(e.fd);
    }

    // non-copyable
    SimplePerf(const SimplePerf&) = delete;
    SimplePerf& operator=(const SimplePerf&) = delete;

    // ---------- measurement API -----------------------------------------

    void start() {
        ioctl(m_events[0].fd, PERF_EVENT_IOC_RESET,  PERF_IOC_FLAG_GROUP);
        ioctl(m_events[0].fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
        m_start_ns = clock_ns();
    }

    // Returns a reference to the internal results map (includes "ns" key).
    const std::map<std::string, uint64_t>& stop() {
        uint64_t end_ns = clock_ns();
        ioctl(m_events[0].fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

        m_results.clear();
        m_results["ns"] = end_ns - m_start_ns;

        ReadBuf buf{};
        ssize_t rc = ::read(m_events[0].fd, &buf, sizeof(buf));
        SPERF_ASSERT(rc > 0, "read perf counters failed");
        SPERF_ASSERT(buf.nr == m_events.size(), "event count mismatch");

        for (uint64_t i = 0; i < buf.nr; i++) {
            for (auto& e : m_events) {
                if (buf.values[i].id == e.id) {
                    e.value = buf.values[i].value;
                    m_results[e.name] = e.value;
                }
            }
        }
        return m_results;
    }

    // ---------- convenience: profile a callable -------------------------
    // Prints a one-line summary per repeat.
    //   name       — label for the output line
    //   n_repeats  — how many times to call func and print
    //   loop_count — iterations inside func (for per-iter stats)
    //   ops/bytes  — total FLOPs / bytes moved inside func (0 = skip)
    template <typename F>
    void profile(F func,
                 const std::string& name,
                 int n_repeats  = 1,
                 int loop_count = 1,
                 double ops     = 0,
                 double bytes   = 0)
    {
        for (int r = 0; r < n_repeats; r++) {
            start();
            func();
            auto& evs = stop();

            printf("\033[0;36m[%8s] %d/%d %lu(ns) ", name.c_str(), r + 1, n_repeats, (unsigned long)evs.at("ns"));

            // Print counter names header
            for (auto const& kv : evs)
                if (kv.first != "ns") printf("%s/", kv.first.c_str());

            printf("(per-iter): ");

            // Print per-iteration values
            for (auto const& kv : evs)
                if (kv.first != "ns") printf("%.1f ", static_cast<double>(kv.second) / loop_count);

            // CPI
            if (has_cycles() && has_instructions()) {
                double cpi = static_cast<double>(evs.at(m_cycles_name)) / evs.at(m_instr_name);
                printf("CPI: %.2f ", cpi);
            }

            // GHz
            if (has_cycles())
                printf("%.2f(GHz) ", static_cast<double>(evs.at(m_cycles_name)) / evs.at("ns"));

            // Throughput
            if (ops > 0)
                printf("%.1f(GOP/s) ", ops / evs.at("ns"));
            if (bytes > 0)
                printf("%.1f(GB/s) ", bytes / evs.at("ns"));
            if (ops > 0 && bytes > 0)
                printf("%.1f(OP/B) ", ops / bytes);

            printf("\033[0m\n");
        }
    }

    // ---------- accessors -----------------------------------------------
    const std::vector<Event>&              events()  const { return m_events;  }
    const std::map<std::string, uint64_t>& results() const { return m_results; }

private:
    // ---- read buffer matching PERF_FORMAT_GROUP | PERF_FORMAT_ID -------
    static constexpr int MAX_EVENTS = 8;
    struct ReadBuf {
        uint64_t nr;
        struct { uint64_t value; uint64_t id; } values[MAX_EVENTS];
    };

    std::vector<Event>              m_events;
    std::map<std::string, uint64_t> m_results;
    uint64_t                        m_start_ns = 0;
    std::string                     m_cycles_name;
    std::string                     m_instr_name;

    bool has_cycles()       const { return !m_cycles_name.empty(); }
    bool has_instructions() const { return !m_instr_name.empty(); }

    static uint64_t clock_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
    }

    void open_events() {
        // Identify cycles / instructions names
        for (auto& e : m_events) {
            if (e.is_cycles())       m_cycles_name = e.name;
            if (e.is_instructions()) m_instr_name  = e.name;
        }

        // Group leader
        m_events[0].fd = syscall(SYS_perf_event_open, &m_events[0].attr, 0, -1, -1, 0);
        SPERF_ASSERT(m_events[0].fd >= 0, "perf_event_open failed for group leader (try: echo 0 > /proc/sys/kernel/perf_event_paranoid)");
        ioctl(m_events[0].fd, PERF_EVENT_IOC_ID, &m_events[0].id);

        // Remaining events join the group
        for (size_t i = 1; i < m_events.size(); i++) {
            m_events[i].fd = syscall(SYS_perf_event_open, &m_events[i].attr, 0, -1, m_events[0].fd, 0);
            SPERF_ASSERT(m_events[i].fd >= 0, "perf_event_open failed");
            ioctl(m_events[i].fd, PERF_EVENT_IOC_ID, &m_events[i].id);
        }
    }
};
