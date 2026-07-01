#ifndef CASE2_FILTER_POSITIVE_HPP
#define CASE2_FILTER_POSITIVE_HPP

#include <stddef.h>

inline size_t filter_positive_branchy(int *values, size_t length) {
    size_t pos = 0;
    for (size_t i = 0; i < length; i++) {
        if (values[i] >= 0) {
            values[pos++] = values[i];
        }
    }
    return pos;
}

inline size_t filter_positive_branchless(int *values, size_t length) {
    size_t pos = 0;
    for (size_t i = 0; i < length; i++) {
        values[pos] = values[i];
        pos += (values[i] >= 0 ? 1 : 0);
    }
    return pos;
}

inline size_t filter_positive_branchless_bit(int *values, size_t length) {
    size_t pos = 0;
    for (size_t i = 0; i < length; i++) {
        values[pos] = values[i];
        pos += (size_t)(~((unsigned int)values[i] >> 31) & 1);
    }
    return pos;
}

#endif