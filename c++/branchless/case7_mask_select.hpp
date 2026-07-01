#ifndef CASE7_MASK_SELECT_HPP
#define CASE7_MASK_SELECT_HPP

#include <stdint.h>

inline int select_branchy(int cond, int when_true, int when_false) {
    if (cond) return when_true;
    return when_false;
}

inline int select_branchless_mask(int cond, int when_true, int when_false) {
    uint32_t mask = (uint32_t)-(cond != 0);
    uint32_t true_bits = (uint32_t)when_true;
    uint32_t false_bits = (uint32_t)when_false;
    return (int)((true_bits & mask) | (false_bits & ~mask));
    // return (cond ? when_true : when_false);
}

#endif