#ifndef CASE3_UNROLLED_SUM_HPP
#define CASE3_UNROLLED_SUM_HPP

#include <stddef.h>

#include "case1_hex.hpp"

inline long long sum_hex_simple(const char *input, size_t length) {
    long long total = 0;
    for (size_t i = 0; i < length; i++) {
        total += decode_hex_branchless((unsigned char)input[i]);
    }
    return total;
}

inline long long sum_hex_unrolled(const char *input, size_t length) {
    long long total = 0;
    size_t i = 0;
    for (; i + 3 < length; i += 4) {
        total += decode_hex_branchless((unsigned char)input[i]);
        total += decode_hex_branchless((unsigned char)input[i + 1]);
        total += decode_hex_branchless((unsigned char)input[i + 2]);
        total += decode_hex_branchless((unsigned char)input[i + 3]);
    }
    for (; i < length; i++) {
        total += decode_hex_branchless((unsigned char)input[i]);
    }
    return total;
}

#endif