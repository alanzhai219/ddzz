#ifndef SIMD_EXP15_COMMON_BENCHMARK_HPP
#define SIMD_EXP15_COMMON_BENCHMARK_HPP

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

namespace simd_exp15 {

template <typename T>
T *checked_malloc(size_t count) {
    T *ptr = (T *)malloc(count * sizeof(T));
    if (ptr == NULL) {
        fprintf(stderr, "malloc failed\n");
        exit(1);
    }
    return ptr;
}

template <typename Fn>
double benchmark_seconds(int rounds, Fn body) {
    clock_t start = clock();
    for (int round = 0; round < rounds; round++) {
        body();
    }
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

template <typename T>
void validate_equal_buffers(const T *expected, const T *actual, size_t length) {
    for (size_t i = 0; i < length; i++) {
        if (expected[i] != actual[i]) {
            fprintf(stderr,
                    "validation failed at %zu: expected=%lld actual=%lld\n",
                    i,
                    (long long)expected[i],
                    (long long)actual[i]);
            exit(1);
        }
    }
}

inline void validate_equal_size(size_t expected, size_t actual) {
    if (expected != actual) {
        fprintf(stderr,
                "validation failed: expected=%zu actual=%zu\n",
                expected,
                actual);
        exit(1);
    }
}

}  // namespace simd_exp15

#endif