#ifndef CASE8_SENTINEL_SEARCH_HPP
#define CASE8_SENTINEL_SEARCH_HPP

#include <stddef.h>

inline size_t find_byte_branchy(const unsigned char *data,
                                size_t length,
                                unsigned char target) {
    for (size_t i = 0; i < length; i++) {
        if (data[i] == target) {
            return i;
        }
    }
    return length;
}

inline size_t find_byte_sentinel(unsigned char *data,
                                 size_t length,
                                 unsigned char target) {
    data[length] = target;

    size_t i = 0;
    while (data[i] != target) {
        i++;
    }

    if (i == length) {
        return length;
    }
    return i;
}

#endif