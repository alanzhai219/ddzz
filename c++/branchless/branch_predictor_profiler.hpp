#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <stdint.h>

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
                            int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

struct perf_counter {
    int fd;
};

struct perf_counter create_counter(uint32_t type, uint64_t config) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    struct perf_counter c;
    c.fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (c.fd == -1) {
        perror("perf_event_open");
        exit(1);
    }
    return c;
}

void start_counter(struct perf_counter *c) {
    ioctl(c->fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(c->fd, PERF_EVENT_IOC_ENABLE, 0);
}

long long stop_counter(struct perf_counter *c) {
    ioctl(c->fd, PERF_EVENT_IOC_DISABLE, 0);
    long long count = 0;
    ssize_t bytes_read = read(c->fd, &count, sizeof(count));
    if (bytes_read != (ssize_t)sizeof(count)) {
        perror("read");
        exit(1);
    }
    return count;
}
