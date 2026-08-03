// Optimized NCHW 2D forward pooling example.
// It mirrors the high-level ncsp path used by OpenVINO JIT pooling:
// NCHW slice -> private NHWC8 scratchpad -> vector pooling -> NCHW slice.

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <limits>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace pooling {

enum class pooling_kind { max, avg_include_padding, avg_exclude_padding };

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
    int threads {0};
};

// [OPT-1] Each AVX2 instruction processes eight adjacent channels.
constexpr int channel_block = 8;
// [OPT-3] A tile maintains up to four independent output accumulators.
constexpr int output_width_unroll = 4;

// [N, C, H, W]
static std::size_t nchw_offset(int n, int c, int h, int w, const pooling2d_desc& d, bool output) {
    const int height = output ? d.oh : d.ih;
    const int width = output ? d.ow : d.iw;
    return static_cast<std::size_t>(((n * d.c + c) * height + h) * width + w);
}

static void pooling2d_nchw_scalar(
        const float* src, float* dst, const pooling2d_desc& d) {
    for (int n = 0; n < d.n; ++n)
        for (int c = 0; c < d.c; ++c)
            for (int oh = 0; oh < d.oh; ++oh)
                for (int ow = 0; ow < d.ow; ++ow) {
                    float value = d.kind == pooling_kind::max
                            ? std::numeric_limits<float>::lowest()
                            : 0.0F;
                    int valid_elements = 0;
                    for (int kh = 0; kh < d.kh; ++kh) {
                        const int ih = oh * d.stride_h - d.pad_t + kh;
                        if (ih < 0 || ih >= d.ih) {
                            continue;
                        }
                        for (int kw = 0; kw < d.kw; ++kw) {
                            const int iw = ow * d.stride_w - d.pad_l + kw;
                            if (iw < 0 || iw >= d.iw) {
                                continue;
                            }
                            const float input = src[nchw_offset(n, c, ih, iw, d, false)];
                            if (d.kind == pooling_kind::max)
                                value = std::max(value, input);
                            else {
                                value += input;
                                ++valid_elements;
                            }
                        }
                    }
                    if (d.kind == pooling_kind::avg_include_padding) {
                        value /= static_cast<float>(d.kh * d.kw);
                    } else if (d.kind == pooling_kind::avg_exclude_padding) {
                        value /= static_cast<float>(valid_elements);
                    }
                    dst[nchw_offset(n, c, oh, ow, d, true)] = value;
                }
}

// [OPT-5] Convert one NCHW channel block into contiguous [H][W][8] storage.
// Zero padding makes the final partial channel block safe for vector loads.
static void nchw_to_nhwc8(const float* src, float* blocked, int n, int cb, const pooling2d_desc& d) {
    const int first_channel = cb * channel_block;
    for (int h = 0; h < d.ih; ++h)
        for (int w = 0; w < d.iw; ++w)
            for (int lane = 0; lane < channel_block; ++lane) {
                const int c = first_channel + lane;
                blocked[(h * d.iw + w) * channel_block + lane] = c < d.c ? src[nchw_offset(n, c, h, w, d, false)] : 0.0F;
            }
}

// [OPT-9] Store only valid lanes of an incomplete [OH][OW][8] tail block.
static void nhwc8_to_nchw(const float* blocked, float* dst, int n, int cb, const pooling2d_desc& d) {
    const int first_channel = cb * channel_block;
    const int valid_channels = std::min(channel_block, d.c - first_channel);
    for (int oh = 0; oh < d.oh; ++oh)
        for (int ow = 0; ow < d.ow; ++ow)
            for (int lane = 0; lane < valid_channels; ++lane) {
                dst[nchw_offset(n, first_channel + lane, oh, ow, d, true)] = blocked[(oh * d.ow + ow) * channel_block + lane];
            }
}

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
#define POOLING_AVX2 __attribute__((target("avx2")))
#else
#define POOLING_AVX2
#endif

// [OPT-4] The interior path omits padding checks from its KH/KW hot loop.
static POOLING_AVX2 void pooling_tile_interior(const float* src, float* dst,
        int oh, int ow_begin, int tile_width, const pooling2d_desc& d) {
    __m256 acc[output_width_unroll];
    const __m256 initial = d.kind == pooling_kind::max
            ? _mm256_set1_ps(std::numeric_limits<float>::lowest())
            : _mm256_setzero_ps();
    for (int u = 0; u < tile_width; ++u) {
        acc[u] = initial;
    }

    const int input_h = oh * d.stride_h - d.pad_t;
    for (int kh = 0; kh < d.kh; ++kh) {
        const int ih = input_h + kh;
        for (int kw = 0; kw < d.kw; ++kw) {
            for (int u = 0; u < tile_width; ++u) {
                const int iw = (ow_begin + u) * d.stride_w - d.pad_l + kw;
                const __m256 input = _mm256_loadu_ps(src + (ih * d.iw + iw) * channel_block);
                acc[u] = d.kind == pooling_kind::max
                    ? _mm256_max_ps(acc[u], input)
                    : _mm256_add_ps(acc[u], input);
            }
        }
    }

    if (d.kind != pooling_kind::max) {
        const float divisor = static_cast<float>(d.kh * d.kw);
        // [OPT-8] Broadcast one reciprocal and use vector multiplication.
        const __m256 reciprocal = _mm256_set1_ps(1.0F / divisor);
        for (int u = 0; u < tile_width; ++u) {
            acc[u] = _mm256_mul_ps(acc[u], reciprocal);
        }
    }

    for (int u = 0; u < tile_width; ++u) {
    _mm256_storeu_ps(
        dst + (oh * d.ow + ow_begin + u) * channel_block, acc[u]);
    }
}

// [OPT-4] Only boundary positions pay the cost of bounds checks.
static POOLING_AVX2 void pooling_point_boundary(const float* src, float* dst, int oh, int ow, const pooling2d_desc& d) {
    __m256 acc = d.kind == pooling_kind::max
            ? _mm256_set1_ps(std::numeric_limits<float>::lowest())
            : _mm256_setzero_ps();
    int valid_elements = 0;

    for (int kh = 0; kh < d.kh; ++kh) {
        const int ih = oh * d.stride_h - d.pad_t + kh;
        if (ih < 0 || ih >= d.ih) {
            continue;
        }
        for (int kw = 0; kw < d.kw; ++kw) {
            const int iw = ow * d.stride_w - d.pad_l + kw;
            if (iw < 0 || iw >= d.iw) {
                continue;
            }
            const __m256 input = _mm256_loadu_ps(src + (ih * d.iw + iw) * channel_block);
            acc = d.kind == pooling_kind::max ? _mm256_max_ps(acc, input)
                                              : _mm256_add_ps(acc, input);
            ++valid_elements;
        }
    }

    if (d.kind == pooling_kind::avg_include_padding) {
        acc = _mm256_mul_ps(acc, _mm256_set1_ps(1.0F / (d.kh * d.kw)));
    }
    else if (d.kind == pooling_kind::avg_exclude_padding) {
        acc = _mm256_mul_ps(acc, _mm256_set1_ps(1.0F / valid_elements));
    }

    _mm256_storeu_ps(dst + (oh * d.ow + ow) * channel_block, acc);
}

// [OPT-3] Calls the interior kernel with tiles up to output_width_unroll wide.
static POOLING_AVX2 void pooling_block_avx2(const float* src, float* dst, const pooling2d_desc& d) {
    for (int oh = 0; oh < d.oh; ++oh) {
        int ow = 0;
        while (ow < d.ow) {
            const int tile_width = std::min(output_width_unroll, d.ow - ow);
            const int first_ih = oh * d.stride_h - d.pad_t;
            const int last_ih = first_ih + d.kh - 1;
            const int first_iw = ow * d.stride_w - d.pad_l;
            const int last_iw = (ow + tile_width - 1) * d.stride_w - d.pad_l + d.kw - 1;
            const bool interior = first_ih >= 0 && last_ih < d.ih && first_iw >= 0 && last_iw < d.iw;

            if (interior) {
                pooling_tile_interior(src, dst, oh, ow, tile_width, d);
            } else {
                for (int u = 0; u < tile_width; ++u) {
                    pooling_point_boundary(src, dst, oh, ow + u, d);
                }
            }
            ow += tile_width;
        }
    }
}

static bool has_avx2() {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}
#undef POOLING_AVX2
#endif

// [OPT-2] Runtime dispatch selects AVX2 or the scalar fallback.
void pooling2d_nchw_jit_style(
        const float* src, float* dst, const pooling2d_desc& d) {
#if defined(__x86_64__) || defined(_M_X64)
    if (!has_avx2()) {
        pooling2d_nchw_scalar(src, dst, d);
        return;
    }

    // hyper params
    const int channel_block_num = (d.c + channel_block - 1) / channel_block;
    const int task_count = d.n * channel_block_num;
    const unsigned int detected_threads = std::thread::hardware_concurrency();
    const int requested_threads = d.threads > 0 ? d.threads : static_cast<int>(detected_threads);
    const int worker_count = std::max(1, std::min(task_count, requested_threads > 0 ? requested_threads : 1));
    std::atomic<int> next_task {0};

    auto worker = [&] {
        // [OPT-6] Scratchpads are private to this worker, preventing sharing.
        std::vector<float> src_scratch(static_cast<std::size_t>(d.ih) * d.iw * channel_block);
        std::vector<float> dst_scratch(static_cast<std::size_t>(d.oh) * d.ow * channel_block);

        for (int task = next_task.fetch_add(1); task < task_count; task = next_task.fetch_add(1)) {
            const int n = task / channel_block_num;
            const int cb = task % channel_block_num;
            // [OPT-7] Complete all output rows while the converted (n, cb)
            // source slice remains cache-resident.
            nchw_to_nhwc8(src, src_scratch.data(), n, cb, d);
            pooling_block_avx2(src_scratch.data(), dst_scratch.data(), d);
            nhwc8_to_nchw(dst_scratch.data(), dst, n, cb, d);
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(worker_count - 1);
    for (int worker_index = 1; worker_index < worker_count; ++worker_index) {
        workers.emplace_back(worker);
    }
    worker();
    for (auto& thread : workers) {
        thread.join();
    }
#else
    pooling2d_nchw_scalar(src, dst, d);
#endif
}

}  // namespace pooling
