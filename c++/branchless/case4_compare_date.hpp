#ifndef CASE4_COMPARE_DATE_HPP
#define CASE4_COMPARE_DATE_HPP

#include <stdint.h>

inline int compare_date_branchy(int y1, int m1, int d1, int y2, int m2, int d2) {
    if (y1 != y2) return y1 - y2;
    if (m1 != m2) return m1 - m2;
    return d1 - d2;
}

static inline int32_t encode_date(int y, int m, int d) {
    return y * 10000 + m * 100 + d;
}

inline int compare_date_branchless(int y1, int m1, int d1, int y2, int m2, int d2) {
    int32_t a = encode_date(y1, m1, d1);
    int32_t b = encode_date(y2, m2, d2);
    return (a > b) - (a < b);
}

#endif