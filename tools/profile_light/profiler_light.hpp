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
    uint64_t read_format;
    uint64_t id;
};

static struct perf_counter create_counter_ex(uint32_t type,
                                             uint64_t config,
                                             int group_fd,
                                             uint64_t read_format,
                                             int disabled) {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = type;
    pe.size = sizeof(pe);
    pe.config = config;
    pe.disabled = disabled;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    pe.read_format = read_format;

    struct perf_counter c;
    c.read_format = read_format;
    c.fd = perf_event_open(&pe, 0, -1, group_fd, 0);
    if (c.fd == -1) {
        perror("perf_event_open");
        exit(1);
    }
    if (ioctl(c.fd, PERF_EVENT_IOC_ID, &c.id) == -1) {
        perror("ioctl(PERF_EVENT_IOC_ID)");
        close(c.fd);
        exit(1);
    }
    return c;
}

struct perf_counter create_counter(uint32_t type, uint64_t config) {
    return create_counter_ex(type, config, -1, 0, 1);
}

static uint64_t make_hw_cache_config(uint64_t cache_id,
                                     uint64_t op_id,
                                     uint64_t result_id) {
    return cache_id | (op_id << 8) | (result_id << 16);
}

struct perf_counter create_group_counter_leader(uint32_t type, uint64_t config) {
    return create_counter_ex(type, config, -1, PERF_FORMAT_GROUP | PERF_FORMAT_ID, 1);
}

struct perf_counter create_group_counter_member(uint32_t type,
                                                uint64_t config,
                                                const struct perf_counter *leader) {
    return create_counter_ex(type,
                             config,
                             leader->fd,
                             PERF_FORMAT_GROUP | PERF_FORMAT_ID,
                             0);
}

void start_counter(struct perf_counter *c) {
    ioctl(c->fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(c->fd, PERF_EVENT_IOC_ENABLE, 0);
}

void start_counter_group(struct perf_counter *leader) {
    ioctl(leader->fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(leader->fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
}

void close_counter(struct perf_counter *c) {
    if (c->fd >= 0) {
        close(c->fd);
        c->fd = -1;
    }
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

void stop_counter_group(struct perf_counter *leader,
                        const struct perf_counter *counters,
                        size_t value_count,
                        long long *values) {
    ioctl(leader->fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);

    size_t words_per_counter = (leader->read_format & PERF_FORMAT_ID) ? 2 : 1;
    size_t word_count = 1 + value_count * words_per_counter;
    uint64_t *buffer = (uint64_t *)malloc(word_count * sizeof(uint64_t));
    if (buffer == NULL) {
        perror("malloc");
        exit(1);
    }

    ssize_t bytes_read = read(leader->fd, buffer, word_count * sizeof(uint64_t));
    if (bytes_read != (ssize_t)(word_count * sizeof(uint64_t))) {
        perror("read");
        free(buffer);
        exit(1);
    }
    if (buffer[0] != value_count) {
        fprintf(stderr,
                "group read count mismatch: expected=%zu actual=%llu\n",
                value_count,
                (unsigned long long)buffer[0]);
        free(buffer);
        exit(1);
    }

    for (size_t i = 0; i < value_count; i++) {
        values[i] = 0;
    }

    if (leader->read_format & PERF_FORMAT_ID) {
        unsigned char *matched = (unsigned char *)calloc(value_count, sizeof(unsigned char));
        if (matched == NULL) {
            perror("calloc");
            free(buffer);
            exit(1);
        }

        for (size_t i = 0; i < value_count; i++) {
            uint64_t value = buffer[1 + i * 2];
            uint64_t id = buffer[1 + i * 2 + 1];
            size_t found = value_count;

            for (size_t j = 0; j < value_count; j++) {
                if (counters[j].id == id) {
                    found = j;
                    break;
                }
            }

            if (found == value_count) {
                fprintf(stderr, "group read returned unknown event id=%llu\n",
                        (unsigned long long)id);
                free(matched);
                free(buffer);
                exit(1);
            }

            values[found] = (long long)value;
            matched[found] = 1;
        }

        for (size_t i = 0; i < value_count; i++) {
            if (!matched[i]) {
                fprintf(stderr,
                        "group read missing event id=%llu at index=%zu\n",
                        (unsigned long long)counters[i].id,
                        i);
                free(matched);
                free(buffer);
                exit(1);
            }
        }
        free(matched);
    } else {
        for (size_t i = 0; i < value_count; i++) {
            values[i] = (long long)buffer[i + 1];
        }
    }
    free(buffer);
}
