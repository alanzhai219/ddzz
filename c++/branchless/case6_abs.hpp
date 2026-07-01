#ifndef CASE6_ABS_HPP
#define CASE6_ABS_HPP

inline int abs_branchy(int v) {
    if (v < 0) return -v;
    return v;
}

inline int abs_branchless(int v) {
    int mask = v >> 31;
    return (v ^ mask) - mask;
}

#endif