// Reference NCHW 2D pooling implementation.
// Deliberately scalar and layout-generic within NCHW.

#include <algorithm>
#include <cstddef>
#include <limits>

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

static float apply_post_op(float value, const pooling2d_desc& d) {
    switch (d.post_op) {
        case post_op_kind::none: return value;
        case post_op_kind::relu: return std::max(value, 0.0F);
        case post_op_kind::clamp:
            return std::min(std::max(value, d.post_op_min), d.post_op_max);
        case post_op_kind::scale_bias:
            return value * d.post_op_scale + d.post_op_bias;
    }
    return value;
}

static std::size_t nchw_offset(
        int n, int c, int h, int w, const pooling2d_desc& d, bool output) {
    const int height = output ? d.oh : d.ih;
    const int width = output ? d.ow : d.iw;
    return static_cast<std::size_t>(((n * d.c + c) * height + h) * width + w);
}

void pooling2d_nchw_reference(
        const float* src, float* dst, const pooling2d_desc& d) {
    for (int n = 0; n < d.n; ++n) {
        for (int c = 0; c < d.c; ++c) {
            for (int oh = 0; oh < d.oh; ++oh) {
                for (int ow = 0; ow < d.ow; ++ow) {
                    float value = d.kind == pooling_kind::max
                            ? std::numeric_limits<float>::lowest()
                            : 0.0F;
                    int valid_elements = 0;

                    for (int kh = 0; kh < d.kh; ++kh) {
                        const int ih = oh * d.stride_h - d.pad_t + kh;
                        if (ih < 0 || ih >= d.ih)
                            continue;

                        for (int kw = 0; kw < d.kw; ++kw) {
                            const int iw = ow * d.stride_w - d.pad_l + kw;
                            if (iw < 0 || iw >= d.iw)
                                continue;

                            const float input = src[nchw_offset(n, c, ih, iw, d, false)];
                            if (d.kind == pooling_kind::max)
                                value = std::max(value, input);
                            else {
                                value += input;
                                ++valid_elements;
                            }
                        }
                    }

                    if (d.kind == pooling_kind::avg_include_padding)
                        value /= static_cast<float>(d.kh * d.kw);
                    else if (d.kind == pooling_kind::avg_exclude_padding)
                        value /= static_cast<float>(valid_elements);

                    dst[nchw_offset(n, c, oh, ow, d, true)]
                            = apply_post_op(value, d);
                }
            }
        }
    }
}

}  // namespace pooling
