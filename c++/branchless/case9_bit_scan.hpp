#ifndef CASE9_BIT_SCAN_HPP
#define CASE9_BIT_SCAN_HPP

#include <stdint.h>

inline int first_set_bit_branchy(uint32_t mask) {
    for (int bit = 0; bit < 32; bit++) {
        if (mask & (1u << bit)) {
            return bit;
        }
    }
    return 32;
}

inline int first_set_bit_branchless(uint32_t mask) {
    if (mask == 0) {
        return 32;
    }
    return __builtin_ctz(mask);
}

#endif