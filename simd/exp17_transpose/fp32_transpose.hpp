#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace exp17 {

struct StridedTransposeProblem {
    std::vector<size_t> dims;
    std::vector<size_t> src_strides;
    std::vector<size_t> dst_strides;
};

// Generic FP32 reorder: each logical coordinate is read through src_strides
// and written through dst_strides.
void transpose_fp32(const float* src, float* dst, const StridedTransposeProblem& problem);

// Optimized matrix transpose. Uses AVX2 8x8 register transposes for aligned
// tiles, then scalar tails. Source is rows x cols and destination is cols x rows.
void transpose_2d_fp32(const float* src, float* dst, size_t rows, size_t cols);

// Transpose contiguous row-major 3D and 4D FP32 tensors. `order` maps an
// output axis to its input axis.
// For example, {0, 2, 1} maps [D0, D1, D2] to [D0, D2, D1].
void transpose_3d_fp32(const float* src,
                       float* dst,
                       const std::array<size_t, 3>& input_dims,
                       const std::array<size_t, 3>& order);
void transpose_4d_fp32(const float* src,
                       float* dst,
                       const std::array<size_t, 4>& input_dims,
                       const std::array<size_t, 4>& order);

}  // namespace exp17
