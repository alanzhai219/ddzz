#include "fp32_transpose.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_2d(size_t rows, size_t cols) {
    std::vector<float> src(rows * cols);
    std::iota(src.begin(), src.end(), 0.0F);
    std::vector<float> dst(cols * rows, -1.0F);

    exp17::transpose_2d_fp32(src.data(), dst.data(), rows, cols);

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col)
            require(dst[col * rows + row] == src[row * cols + col], "2D transpose result is incorrect");
    }
}

void test_generic_nchw_to_nwch() {
    constexpr size_t n = 2;
    constexpr size_t c = 3;
    constexpr size_t h = 5;
    constexpr size_t w = 7;
    std::vector<float> src(n * c * h * w);
    std::iota(src.begin(), src.end(), 0.0F);
    std::vector<float> dst(src.size(), -1.0F);

    // Logical output coordinate is [n, w, c, h]. It reads source bytes as
    // NCHW and writes a contiguous NWCH output, mirroring acdb -> abcd.
    const exp17::StridedTransposeProblem problem{
        {n, w, c, h},
        {c * h * w, 1, h * w, w},
        {w * c * h, c * h, h, 1},
    };
    exp17::transpose_fp32(src.data(), dst.data(), problem);

    for (size_t ni = 0; ni < n; ++ni) {
        for (size_t ci = 0; ci < c; ++ci) {
            for (size_t hi = 0; hi < h; ++hi) {
                for (size_t wi = 0; wi < w; ++wi) {
                    const size_t src_offset = ((ni * c + ci) * h + hi) * w + wi;
                    const size_t dst_offset = ((ni * w + wi) * c + ci) * h + hi;
                    require(dst[dst_offset] == src[src_offset], "generic strided transpose result is incorrect");
                }
            }
        }
    }
}

template <size_t Rank>
void verify_contiguous_transpose(const std::vector<float>& src,
                                 const std::vector<float>& dst,
                                 const std::array<size_t, Rank>& input_dims,
                                 const std::array<size_t, Rank>& order) {
    std::array<size_t, Rank> input_strides{};
    std::array<size_t, Rank> output_dims{};
    std::array<size_t, Rank> output_strides{};
    input_strides[Rank - 1] = 1;
    output_strides[Rank - 1] = 1;
    for (size_t axis = Rank - 1; axis > 0; --axis) {
        input_strides[axis - 1] = input_strides[axis] * input_dims[axis];
        output_dims[axis] = input_dims[order[axis]];
        output_strides[axis - 1] = output_strides[axis] * output_dims[axis];
    }
    output_dims[0] = input_dims[order[0]];

    for (size_t output_offset = 0; output_offset < dst.size(); ++output_offset) {
        size_t remainder = output_offset;
        size_t input_offset = 0;
        for (size_t axis = 0; axis < Rank; ++axis) {
            const size_t coordinate = remainder / output_strides[axis];
            remainder %= output_strides[axis];
            input_offset += coordinate * input_strides[order[axis]];
        }
        require(dst[output_offset] == src[input_offset], "multidimensional transpose result is incorrect");
    }
}

void test_3d() {
    constexpr std::array<size_t, 3> dims{3, 65, 67};
    std::vector<float> src(dims[0] * dims[1] * dims[2]);
    std::iota(src.begin(), src.end(), 0.0F);
    std::vector<float> dst(src.size());
    const std::array<size_t, 3> order{0, 2, 1};
    exp17::transpose_3d_fp32(src.data(), dst.data(), dims, order);
    verify_contiguous_transpose(src, dst, dims, order);

    const std::array<size_t, 3> generic_order{2, 0, 1};
    exp17::transpose_3d_fp32(src.data(), dst.data(), dims, generic_order);
    verify_contiguous_transpose(src, dst, dims, generic_order);
}

void test_4d() {
    constexpr std::array<size_t, 4> dims{2, 3, 65, 67};
    std::vector<float> src(dims[0] * dims[1] * dims[2] * dims[3]);
    std::iota(src.begin(), src.end(), 0.0F);
    std::vector<float> dst(src.size());
    const std::array<size_t, 4> order{0, 1, 3, 2};
    exp17::transpose_4d_fp32(src.data(), dst.data(), dims, order);
    verify_contiguous_transpose(src, dst, dims, order);

    const std::array<size_t, 4> generic_order{0, 3, 1, 2};
    exp17::transpose_4d_fp32(src.data(), dst.data(), dims, generic_order);
    verify_contiguous_transpose(src, dst, dims, generic_order);
}

void benchmark() {
    constexpr size_t rows = 2048;
    constexpr size_t cols = 2048;
    constexpr int iterations = 20;
    std::vector<float> src(rows * cols, 1.0F);
    std::vector<float> dst(cols * rows);

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i)
        exp17::transpose_2d_fp32(src.data(), dst.data(), rows, cols);
    const auto stop = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(stop - start).count();
    const double bytes = static_cast<double>(iterations) * rows * cols * sizeof(float) * 2.0;
    std::cout << "Benchmark: " << rows << 'x' << cols << ", " << iterations << " iterations, " << std::fixed
              << std::setprecision(2) << bytes / seconds / 1e9 << " GB/s\n";
}

}  // namespace

int main() {
    try {
        test_2d(8, 8);
        test_2d(65, 67);
        test_2d(256, 192);
        test_3d();
        test_4d();
        test_generic_nchw_to_nwch();
        std::cout << "All correctness tests passed.\n";
        benchmark();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Failure: " << error.what() << '\n';
        return 1;
    }
}
