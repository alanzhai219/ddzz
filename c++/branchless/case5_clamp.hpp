#ifndef CASE5_CLAMP_HPP
#define CASE5_CLAMP_HPP

inline int clamp_branchy(int val, int lo, int hi) {
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

inline int clamp_branchless(int val, int lo, int hi) {
    int t = val < lo ? lo : val;
    return t > hi ? hi : t;
}

static inline int branchless_max(int a, int b) {
    int diff = a - b;
    return a - (diff & (diff >> 31));
}

static inline int branchless_min(int a, int b) {
    int diff = a - b;
    return b + (diff & (diff >> 31));
}

inline int clamp_branchless_bit(int val, int lo, int hi) {
    return branchless_min(branchless_max(val, lo), hi);
}

#endif