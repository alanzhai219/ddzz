#include "fp32_transpose.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <immintrin.h>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace exp17 {
namespace {

constexpr size_t kTile = 8;
constexpr size_t kCacheTile = 64;

void transpose_8x8_avx2(const float* src, float* dst, size_t src_stride, size_t dst_stride) {
    __m256 r0 = _mm256_loadu_ps(src + 0 * src_stride);
    __m256 r1 = _mm256_loadu_ps(src + 1 * src_stride);
    __m256 r2 = _mm256_loadu_ps(src + 2 * src_stride);
    __m256 r3 = _mm256_loadu_ps(src + 3 * src_stride);
    __m256 r4 = _mm256_loadu_ps(src + 4 * src_stride);
    __m256 r5 = _mm256_loadu_ps(src + 5 * src_stride);
    __m256 r6 = _mm256_loadu_ps(src + 6 * src_stride);
    __m256 r7 = _mm256_loadu_ps(src + 7 * src_stride);

    const __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    const __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    const __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    const __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    const __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    const __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    const __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    const __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    const __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44);
    const __m256 u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44);
    const __m256 u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44);
    const __m256 u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44);
    const __m256 u7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    _mm256_storeu_ps(dst + 0 * dst_stride, _mm256_permute2f128_ps(u0, u4, 0x20));
    _mm256_storeu_ps(dst + 1 * dst_stride, _mm256_permute2f128_ps(u1, u5, 0x20));
    _mm256_storeu_ps(dst + 2 * dst_stride, _mm256_permute2f128_ps(u2, u6, 0x20));
    _mm256_storeu_ps(dst + 3 * dst_stride, _mm256_permute2f128_ps(u3, u7, 0x20));
    _mm256_storeu_ps(dst + 4 * dst_stride, _mm256_permute2f128_ps(u0, u4, 0x31));
    _mm256_storeu_ps(dst + 5 * dst_stride, _mm256_permute2f128_ps(u1, u5, 0x31));
    _mm256_storeu_ps(dst + 6 * dst_stride, _mm256_permute2f128_ps(u2, u6, 0x31));
    _mm256_storeu_ps(dst + 7 * dst_stride, _mm256_permute2f128_ps(u3, u7, 0x31));
}

void transpose_tile(const float* src, float* dst, size_t rows, size_t cols, size_t row_begin, size_t row_end,
                    size_t col_begin, size_t col_end) {
    size_t row = row_begin;
    for (; row + kTile <= row_end; row += kTile) {
        size_t col = col_begin;
        for (; col + kTile <= col_end; col += kTile) {
            transpose_8x8_avx2(src + row * cols + col, dst + col * rows + row, cols, rows);
        }
        for (; col < col_end; ++col) {
            for (size_t r = 0; r < kTile; ++r)
                dst[col * rows + row + r] = src[(row + r) * cols + col];
        }
    }
    for (; row < row_end; ++row) {
        for (size_t col = col_begin; col < col_end; ++col)
            dst[col * rows + row] = src[row * cols + col];
    }
}

void transpose_2d_serial(const float* src, float* dst, size_t rows, size_t cols) {
    for (size_t row_begin = 0; row_begin < rows; row_begin += kCacheTile) {
        for (size_t col_begin = 0; col_begin < cols; col_begin += kCacheTile) {
            transpose_tile(src, dst, rows, cols, row_begin, std::min(row_begin + kCacheTile, rows), col_begin,
                           std::min(col_begin + kCacheTile, cols));
        }
    }
}

template <size_t Rank>
void validate_permutation(const std::array<size_t, Rank>& order) {
    std::array<bool, Rank> used{};
    for (const size_t axis : order) {
        if (axis >= Rank || used[axis])
            throw std::invalid_argument("order must be a permutation of input axes");
        used[axis] = true;
    }
}

template <size_t Rank>
StridedTransposeProblem make_contiguous_problem(const std::array<size_t, Rank>& input_dims,
                                                const std::array<size_t, Rank>& order) {
    validate_permutation(order);

    StridedTransposeProblem problem;
    problem.dims.resize(Rank);
    problem.src_strides.resize(Rank);
    problem.dst_strides.resize(Rank);

    std::array<size_t, Rank> input_strides{};
    input_strides[Rank - 1] = 1;
    for (size_t axis = Rank - 1; axis > 0; --axis)
        input_strides[axis - 1] = input_strides[axis] * input_dims[axis];

    size_t dst_stride = 1;
    for (size_t axis = Rank; axis-- > 0;) {
        problem.dims[axis] = input_dims[order[axis]];
        problem.src_strides[axis] = input_strides[order[axis]];
        problem.dst_strides[axis] = dst_stride;
        dst_stride *= problem.dims[axis];
    }
    return problem;
}

}  // namespace

void transpose_2d_fp32(const float* src, float* dst, size_t rows, size_t cols) {
    if (!src || !dst)
        throw std::invalid_argument("source and destination must not be null");
    if (src == dst)
        throw std::invalid_argument("in-place transpose is not supported");

    const size_t row_blocks = (rows + kCacheTile - 1) / kCacheTile;
    const size_t col_blocks = (cols + kCacheTile - 1) / kCacheTile;

    // Cache blocking keeps an input/output working set bounded, while each
    // independent tile can be scheduled without synchronization.
#pragma omp parallel for collapse(2) schedule(static) if (row_blocks * col_blocks > 1)
    for (size_t rb = 0; rb < row_blocks; ++rb) {
        for (size_t cb = 0; cb < col_blocks; ++cb) {
            const size_t row_begin = rb * kCacheTile;
            const size_t col_begin = cb * kCacheTile;
            transpose_tile(src, dst, rows, cols, row_begin, std::min(row_begin + kCacheTile, rows), col_begin,
                           std::min(col_begin + kCacheTile, cols));
        }
    }
}

void transpose_fp32(const float* src, float* dst, const StridedTransposeProblem& problem) {
    if (!src || !dst)
        throw std::invalid_argument("source and destination must not be null");
    if (src == dst)
        throw std::invalid_argument("in-place transpose is not supported");
    if (problem.dims.empty() || problem.dims.size() != problem.src_strides.size() ||
        problem.dims.size() != problem.dst_strides.size()) {
        throw std::invalid_argument("dimensions and stride vectors must be non-empty and have equal lengths");
    }

    size_t work_amount = 1;
    for (const size_t dim : problem.dims) {
        if (dim == 0)
            return;
        work_amount *= dim;
    }

    // The generic path retains the reorder model: a flattened logical index
    // is decoded once and mapped through independent input/output strides.
#pragma omp parallel for schedule(static) if (work_amount >= 4096)
    for (size_t linear = 0; linear < work_amount; ++linear) {
        size_t remainder = linear;
        size_t src_offset = 0;
        size_t dst_offset = 0;
        for (size_t axis = problem.dims.size(); axis-- > 0;) {
            const size_t coordinate = remainder % problem.dims[axis];
            remainder /= problem.dims[axis];
            src_offset += coordinate * problem.src_strides[axis];
            dst_offset += coordinate * problem.dst_strides[axis];
        }
        dst[dst_offset] = src[src_offset];
    }
}

void transpose_3d_fp32(const float* src,
                       float* dst,
                       const std::array<size_t, 3>& input_dims,
                       const std::array<size_t, 3>& order) {
    if (!src || !dst)
        throw std::invalid_argument("source and destination must not be null");
    if (src == dst)
        throw std::invalid_argument("in-place transpose is not supported");
    validate_permutation(order);

    // [D0, rows, cols] -> [D0, cols, rows]: one independent matrix per D0.
    if (order == std::array<size_t, 3>{0, 2, 1}) {
        const size_t matrices = input_dims[0];
        const size_t rows = input_dims[1];
        const size_t cols = input_dims[2];
#pragma omp parallel for schedule(static) if (matrices > 1)
        for (size_t matrix = 0; matrix < matrices; ++matrix)
            transpose_2d_serial(src + matrix * rows * cols, dst + matrix * rows * cols, rows, cols);
        return;
    }

    transpose_fp32(src, dst, make_contiguous_problem(input_dims, order));
}

void transpose_4d_fp32(const float* src,
                       float* dst,
                       const std::array<size_t, 4>& input_dims,
                       const std::array<size_t, 4>& order) {
    if (!src || !dst)
        throw std::invalid_argument("source and destination must not be null");
    if (src == dst)
        throw std::invalid_argument("in-place transpose is not supported");
    validate_permutation(order);

    // [D0, D1, rows, cols] -> [D0, D1, cols, rows]: each outer index owns a
    // separate matrix, retaining AVX2 tiles without nested OpenMP regions.
    if (order == std::array<size_t, 4>{0, 1, 3, 2}) {
        const size_t matrices = input_dims[0] * input_dims[1];
        const size_t rows = input_dims[2];
        const size_t cols = input_dims[3];
#pragma omp parallel for schedule(static) if (matrices > 1)
        for (size_t matrix = 0; matrix < matrices; ++matrix)
            transpose_2d_serial(src + matrix * rows * cols, dst + matrix * rows * cols, rows, cols);
        return;
    }

    transpose_fp32(src, dst, make_contiguous_problem(input_dims, order));
}

}  // namespace exp17
