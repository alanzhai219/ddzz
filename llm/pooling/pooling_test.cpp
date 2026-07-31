#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace pooling {

enum class pooling_kind { max, avg_include_padding, avg_exclude_padding };
enum class post_op_kind { none, relu, clamp, scale_bias };

struct pooling2d_desc {
    int n;
    int c;
    int ih;
    int iw;
    int oh;
    int ow;
    int kh;
    int kw;
    int stride_h;
    int stride_w;
    int pad_t;
    int pad_l;
    pooling_kind kind;
    post_op_kind post_op {post_op_kind::none};
    float post_op_scale {1.0F};
    float post_op_bias {0.0F};
    float post_op_min {0.0F};
    float post_op_max {0.0F};
    int threads {0};
};

void pooling2d_nchw_reference(
        const float* src, float* dst, const pooling2d_desc& d);
void pooling2d_nchw_jit_style(
        const float* src, float* dst, const pooling2d_desc& d);

}  // namespace pooling

int main() {
    using namespace pooling;
    const pooling2d_desc base {
            2, 13, 5, 7, 3, 4, 3, 3, 2, 2, 1, 1,
            pooling_kind::max, post_op_kind::clamp, 1.0F, 0.0F, -0.25F, 0.75F, 3};
    const std::size_t src_size = static_cast<std::size_t>(base.n) * base.c * base.ih * base.iw;
    const std::size_t dst_size = static_cast<std::size_t>(base.n) * base.c * base.oh * base.ow;
    std::vector<float> src(src_size);
    for (std::size_t index = 0; index < src.size(); ++index)
        src[index] = static_cast<float>(static_cast<int>(index % 23) - 11) / 10.0F;

    for (const auto kind : {pooling_kind::max, pooling_kind::avg_include_padding,
                 pooling_kind::avg_exclude_padding}) {
        auto d = base;
        d.kind = kind;
        std::vector<float> reference(dst_size);
        std::vector<float> optimized(dst_size);
        pooling2d_nchw_reference(src.data(), reference.data(), d);
        pooling2d_nchw_jit_style(src.data(), optimized.data(), d);

        for (std::size_t index = 0; index < dst_size; ++index) {
            if (std::fabs(reference[index] - optimized[index]) > 1e-6F) {
                std::cerr << "mismatch at index " << index << '\n';
                return 1;
            }
        }
    }

    std::cout << "reference and AVX2 paths match\n";
    return 0;
}
