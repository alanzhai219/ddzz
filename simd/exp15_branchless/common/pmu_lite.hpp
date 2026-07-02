#ifndef SIMD_EXP15_COMMON_PMU_LITE_HPP
#define SIMD_EXP15_COMMON_PMU_LITE_HPP

#include <errno.h>
#include <linux/perf_event.h>
#include <stdint.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <string>
#include <vector>

namespace simd_exp15 {
namespace pmu {

struct Sample {
    uint64_t ns = 0;
    uint64_t cycles = 0;
    uint64_t instructions = 0;
    uint64_t branch_instructions = 0;
    uint64_t branch_misses = 0;
    uint64_t cache_misses = 0;
    uint64_t stalled_frontend = 0;
    uint64_t stalled_backend = 0;

    bool has_cycles = false;
    bool has_instructions = false;
    bool has_branch_instructions = false;
    bool has_branch_misses = false;
    bool has_cache_misses = false;
    bool has_stalled_frontend = false;
    bool has_stalled_backend = false;

    double ipc() const {
        if (!has_cycles || !has_instructions || cycles == 0) {
            return 0.0;
        }
        return (double)instructions / (double)cycles;
    }

    double cpi() const {
        if (!has_cycles || !has_instructions || instructions == 0) {
            return 0.0;
        }
        return (double)cycles / (double)instructions;
    }

    double branch_miss_rate() const {
        if (!has_branch_misses || !has_branch_instructions || branch_instructions == 0) {
            return 0.0;
        }
        return (double)branch_misses / (double)branch_instructions;
    }

    double cache_miss_per_kinst() const {
        if (!has_cache_misses || !has_instructions || instructions == 0) {
            return 0.0;
        }
        return (double)cache_misses * 1000.0 / (double)instructions;
    }
};

enum class EventKind {
    Cycles,
    Instructions,
    BranchInstructions,
    BranchMisses,
    CacheMisses,
    StalledFrontend,
    StalledBackend,
};

class CounterGroup {
public:
    CounterGroup() {
        const bool ok_cycles = open_event(EventKind::Cycles, true, -1);
        const bool ok_instructions = open_event(EventKind::Instructions, true, leader_fd_);
        open_event(EventKind::BranchInstructions, false, leader_fd_);
        open_event(EventKind::BranchMisses, false, leader_fd_);
        open_event(EventKind::CacheMisses, false, leader_fd_);
        open_event(EventKind::StalledFrontend, false, leader_fd_);
        open_event(EventKind::StalledBackend, false, leader_fd_);

        if (!ok_cycles || !ok_instructions) {
            valid_ = false;
            if (error_.empty()) {
                error_ = "failed to open required PMU events";
            }
        } else {
            valid_ = true;
        }
    }

    ~CounterGroup() {
        for (size_t i = 0; i < events_.size(); i++) {
            if (events_[i].fd >= 0) {
                close(events_[i].fd);
            }
        }
    }

    CounterGroup(const CounterGroup &) = delete;
    CounterGroup &operator=(const CounterGroup &) = delete;

    bool valid() const { return valid_; }
    const std::string &error() const { return error_; }

    bool start() {
        if (!valid_ || leader_fd_ < 0) {
            return false;
        }

        if (ioctl(leader_fd_, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) != 0) {
            return false;
        }
        if (ioctl(leader_fd_, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP) != 0) {
            return false;
        }

        start_ns_ = clock_ns();
        return true;
    }

    bool stop(Sample *out) {
        if (out == nullptr || !valid_ || leader_fd_ < 0) {
            return false;
        }

        const uint64_t end_ns = clock_ns();
        if (ioctl(leader_fd_, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP) != 0) {
            return false;
        }

        out->ns = end_ns - start_ns_;
        out->cycles = 0;
        out->instructions = 0;
        out->branch_instructions = 0;
        out->branch_misses = 0;
        out->cache_misses = 0;
        out->stalled_frontend = 0;
        out->stalled_backend = 0;
        out->has_cycles = false;
        out->has_instructions = false;
        out->has_branch_instructions = false;
        out->has_branch_misses = false;
        out->has_cache_misses = false;
        out->has_stalled_frontend = false;
        out->has_stalled_backend = false;

        struct ReadValue {
            uint64_t value;
            uint64_t id;
        };
        struct ReadData {
            uint64_t nr;
            ReadValue values[16];
        };

        ReadData data;
        memset(&data, 0, sizeof(data));
        const ssize_t n = read(leader_fd_, &data, sizeof(data));
        if (n <= 0) {
            return false;
        }

        for (uint64_t i = 0; i < data.nr && i < 16; i++) {
            for (size_t j = 0; j < events_.size(); j++) {
                if (events_[j].id == data.values[i].id) {
                    assign_value(out, events_[j].kind, data.values[i].value);
                    break;
                }
            }
        }

        return true;
    }

private:
    struct EventFd {
        EventKind kind;
        int fd = -1;
        uint64_t id = 0;
    };

    std::vector<EventFd> events_;
    int leader_fd_ = -1;
    bool valid_ = false;
    std::string error_;
    uint64_t start_ns_ = 0;

    static uint64_t clock_ns() {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }

    static void map_kind(EventKind kind, uint32_t *type, uint64_t *config) {
        switch (kind) {
            case EventKind::Cycles:
                *type = PERF_TYPE_HARDWARE;
                *config = PERF_COUNT_HW_CPU_CYCLES;
                return;
            case EventKind::Instructions:
                *type = PERF_TYPE_HARDWARE;
                *config = PERF_COUNT_HW_INSTRUCTIONS;
                return;
            case EventKind::BranchInstructions:
                *type = PERF_TYPE_HARDWARE;
                *config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
                return;
            case EventKind::BranchMisses:
                *type = PERF_TYPE_HARDWARE;
                *config = PERF_COUNT_HW_BRANCH_MISSES;
                return;
            case EventKind::CacheMisses:
                *type = PERF_TYPE_HARDWARE;
                *config = PERF_COUNT_HW_CACHE_MISSES;
                return;
            case EventKind::StalledFrontend:
                *type = PERF_TYPE_HARDWARE;
                *config = PERF_COUNT_HW_STALLED_CYCLES_FRONTEND;
                return;
            case EventKind::StalledBackend:
                *type = PERF_TYPE_HARDWARE;
                *config = PERF_COUNT_HW_STALLED_CYCLES_BACKEND;
                return;
        }
    }

    bool open_event(EventKind kind, bool required, int group_fd) {
        uint32_t type = 0;
        uint64_t config = 0;
        map_kind(kind, &type, &config);

        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.type = type;
        attr.size = sizeof(attr);
        attr.config = config;
        attr.disabled = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv = 1;
        attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID;

        const int fd = (int)syscall(SYS_perf_event_open, &attr, 0, -1, group_fd, 0);
        if (fd < 0) {
            if (required) {
                error_ = std::string("perf_event_open failed: ") + strerror(errno) +
                         ". try: sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'";
            }
            return !required;
        }

        uint64_t id = 0;
        if (ioctl(fd, PERF_EVENT_IOC_ID, &id) != 0) {
            if (required) {
                error_ = "PERF_EVENT_IOC_ID failed";
            }
            close(fd);
            return !required;
        }

        EventFd e;
        e.kind = kind;
        e.fd = fd;
        e.id = id;
        events_.push_back(e);

        if (leader_fd_ < 0) {
            leader_fd_ = fd;
        }
        return true;
    }

    static void assign_value(Sample *out, EventKind kind, uint64_t value) {
        switch (kind) {
            case EventKind::Cycles:
                out->cycles = value;
                out->has_cycles = true;
                return;
            case EventKind::Instructions:
                out->instructions = value;
                out->has_instructions = true;
                return;
            case EventKind::BranchInstructions:
                out->branch_instructions = value;
                out->has_branch_instructions = true;
                return;
            case EventKind::BranchMisses:
                out->branch_misses = value;
                out->has_branch_misses = true;
                return;
            case EventKind::CacheMisses:
                out->cache_misses = value;
                out->has_cache_misses = true;
                return;
            case EventKind::StalledFrontend:
                out->stalled_frontend = value;
                out->has_stalled_frontend = true;
                return;
            case EventKind::StalledBackend:
                out->stalled_backend = value;
                out->has_stalled_backend = true;
                return;
        }
    }
};

template <typename Fn>
bool measure(int rounds, Fn fn, Sample *out, std::string *error_message = nullptr) {
    CounterGroup group;
    if (!group.valid()) {
        if (error_message != nullptr) {
            *error_message = group.error();
        }
        return false;
    }

    if (!group.start()) {
        if (error_message != nullptr) {
            *error_message = "failed to start PMU counters";
        }
        return false;
    }

    for (int i = 0; i < rounds; i++) {
        fn();
    }

    if (!group.stop(out)) {
        if (error_message != nullptr) {
            *error_message = "failed to stop/read PMU counters";
        }
        return false;
    }
    return true;
}

}  // namespace pmu
}  // namespace simd_exp15

#endif