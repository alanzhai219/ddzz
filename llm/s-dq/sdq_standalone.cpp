#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

enum class WeightFormat {
    U8,
    I8,
    U4,
    I4,
};

struct ProblemConfig {
    int m = 4;
    int k = 128;
    int n = 32;
    int group_size = 16;
    WeightFormat weight_format = WeightFormat::U8;
};

struct Problem {
    ProblemConfig cfg;
    std::vector<float> src;
    std::vector<int32_t> qweight;
    std::vector<float> weight_scales;
    std::vector<float> weight_zero_points;
    std::vector<float> bias;
};

static int index2d(int row, int col, int cols) {
    return row * cols + col;
}

struct WeightRange {
    int qmin;
    int qmax;
    bool is_unsigned;
};

WeightRange get_weight_range(WeightFormat weight_format) {
    switch (weight_format) {
    case WeightFormat::U8:
        return {0, 255, true};
    case WeightFormat::I8:
        return {-127, 127, false};
    case WeightFormat::U4:
        return {0, 15, true};
    case WeightFormat::I4:
        return {-8, 7, false};
    }
    throw std::runtime_error("Unsupported weight format");
}

std::string to_string(WeightFormat weight_format) {
    switch (weight_format) {
    case WeightFormat::U8:
        return "u8";
    case WeightFormat::I8:
        return "i8";
    case WeightFormat::U4:
        return "u4";
    case WeightFormat::I4:
        return "i4";
    }
    return "unknown";
}

bool is_nibble_format(WeightFormat weight_format) {
    return weight_format == WeightFormat::U4 || weight_format == WeightFormat::I4;
}

Problem make_problem(const ProblemConfig& cfg, std::mt19937& rng) {
    Problem problem;
    problem.cfg = cfg;

    const auto qrange = get_weight_range(cfg.weight_format);

    const int group_count = cfg.k / cfg.group_size;
    problem.src.resize(cfg.m * cfg.k);
    problem.qweight.resize(cfg.k * cfg.n);
    problem.weight_scales.resize(group_count * cfg.n);
    problem.weight_zero_points.resize(group_count * cfg.n);
    problem.bias.resize(cfg.n);

    std::normal_distribution<float> src_dist(0.0f, 1.0f);
    std::normal_distribution<float> bias_dist(0.0f, 0.5f);
    std::uniform_real_distribution<float> scale_dist(0.005f, 0.055f);
    std::uniform_int_distribution<int> qwei_dist(qrange.qmin, qrange.qmax);
    std::uniform_int_distribution<int> zp_dist(0, 15);

    for (float& value : problem.src) {
        value = src_dist(rng);
    }
    for (float& value : problem.bias) {
        value = bias_dist(rng);
    }

    for (int group = 0; group < group_count; ++group) {
        for (int col = 0; col < cfg.n; ++col) {
            problem.weight_scales[index2d(group, col, cfg.n)] = scale_dist(rng);
            if (qrange.is_unsigned) {
                const int zp_upper = std::min(qrange.qmax, 15);
                std::uniform_int_distribution<int> zp_local(0, zp_upper);
                problem.weight_zero_points[index2d(group, col, cfg.n)] = static_cast<float>(zp_local(rng));
            } else {
                problem.weight_zero_points[index2d(group, col, cfg.n)] = 0.0f;
            }
        }
    }

    for (int k_idx = 0; k_idx < cfg.k; ++k_idx) {
        for (int col = 0; col < cfg.n; ++col) {
            problem.qweight[index2d(k_idx, col, cfg.n)] = qwei_dist(rng);
        }
    }

    return problem;
}

// =========================
// Weight Side
// =========================
std::vector<uint8_t> pack_nibbles(const std::vector<int32_t>& logical_qweight) {
    if (logical_qweight.size() % 2 != 0) {
        throw std::runtime_error("Nibble packing requires an even number of logical values");
    }

    std::vector<uint8_t> packed(logical_qweight.size() / 2, 0);
    for (size_t i = 0; i < packed.size(); ++i) {
        const uint8_t low = static_cast<uint8_t>(logical_qweight[2 * i] & 0x0F);
        const uint8_t high = static_cast<uint8_t>(logical_qweight[2 * i + 1] & 0x0F);
        packed[i] = static_cast<uint8_t>(low | (high << 4));
    }
    return packed;
}

std::vector<int32_t> unpack_nibbles(const std::vector<uint8_t>& packed, bool is_signed) {
    std::vector<int32_t> unpacked(packed.size() * 2, 0);
    for (size_t i = 0; i < packed.size(); ++i) {
        int32_t low = static_cast<int32_t>(packed[i] & 0x0F);
        int32_t high = static_cast<int32_t>((packed[i] >> 4) & 0x0F);
        if (is_signed) {
            low = low >= 8 ? low - 16 : low;
            high = high >= 8 ? high - 16 : high;
        }
        unpacked[2 * i] = low;
        unpacked[2 * i + 1] = high;
    }
    return unpacked;
}

std::vector<int32_t> materialize_logical_qweight(const Problem& problem) {
    if (!is_nibble_format(problem.cfg.weight_format)) {
        return problem.qweight;
    }

    const bool is_signed = problem.cfg.weight_format == WeightFormat::I4;
    std::vector<int32_t> nibble_source(problem.qweight.size(), 0);
    for (size_t i = 0; i < problem.qweight.size(); ++i) {
        nibble_source[i] = problem.qweight[i] & 0x0F;
    }
    const auto packed = pack_nibbles(nibble_source);
    return unpack_nibbles(packed, is_signed);
}

std::vector<float> dequantize_weights(const Problem& problem) {
    const auto& cfg = problem.cfg;
    const auto logical_qweight = materialize_logical_qweight(problem);
    std::vector<float> weights(cfg.k * cfg.n, 0.0f);

    for (int group = 0; group < cfg.k / cfg.group_size; ++group) {
        const int begin = group * cfg.group_size;
        const int end = begin + cfg.group_size;
        for (int k_idx = begin; k_idx < end; ++k_idx) {
            for (int col = 0; col < cfg.n; ++col) {
                const float scale = problem.weight_scales[index2d(group, col, cfg.n)];
                const float zp = problem.weight_zero_points[index2d(group, col, cfg.n)];
                const float qwei = static_cast<float>(logical_qweight[index2d(k_idx, col, cfg.n)]);
                weights[index2d(k_idx, col, cfg.n)] = (qwei - zp) * scale;
            }
        }
    }

    return weights;
}

std::vector<float> reference_matmul(const Problem& problem) {
    const auto& cfg = problem.cfg;
    const auto weights = dequantize_weights(problem);
    std::vector<float> output(cfg.m * cfg.n, 0.0f);

    for (int row = 0; row < cfg.m; ++row) {
        for (int col = 0; col < cfg.n; ++col) {
            float acc = problem.bias[col];
            for (int k_idx = 0; k_idx < cfg.k; ++k_idx) {
                acc += problem.src[index2d(row, k_idx, cfg.k)] * weights[index2d(k_idx, col, cfg.n)];
            }
            output[index2d(row, col, cfg.n)] = acc;
        }
    }

    return output;
}

struct SourceDQData {
    std::vector<int32_t> qsrc;
    std::vector<float> src_scales;
    std::vector<int32_t> grouped_sums;
};

// =========================
// Source Side
// =========================
SourceDQData quantize_source_dynamic(const Problem& problem) {
    const auto& cfg = problem.cfg;
    const int group_count = cfg.k / cfg.group_size;

    SourceDQData data;
    data.qsrc.resize(cfg.m * cfg.k, 0);
    data.src_scales.resize(cfg.m * group_count, 0.0f);
    data.grouped_sums.resize(cfg.m * group_count, 0);

    for (int row = 0; row < cfg.m; ++row) {
        for (int group = 0; group < group_count; ++group) {
            const int begin = group * cfg.group_size;
            const int end = begin + cfg.group_size;

            float amax = 0.0f;
            for (int k_idx = begin; k_idx < end; ++k_idx) {
                amax = std::max(amax, std::abs(problem.src[index2d(row, k_idx, cfg.k)]));
            }

            const float dscale = amax == 0.0f ? 0.0f : (amax / 127.0f);
            data.src_scales[index2d(row, group, group_count)] = dscale;

            int32_t sum = 0;
            for (int k_idx = begin; k_idx < end; ++k_idx) {
                int32_t qvalue = 0;
                if (dscale != 0.0f) {
                    const float scaled = problem.src[index2d(row, k_idx, cfg.k)] / dscale;
                    const float rounded = std::nearbyint(scaled);
                    const float clamped = std::max(-127.0f, std::min(127.0f, rounded));
                    qvalue = static_cast<int32_t>(clamped);
                }
                data.qsrc[index2d(row, k_idx, cfg.k)] = qvalue;
                sum += qvalue;
            }
            data.grouped_sums[index2d(row, group, group_count)] = sum;
        }
    }

    return data;
}

// =========================
// Integer Core + Finalize Side
// =========================
std::vector<float> sdq_matmul(const Problem& problem, const SourceDQData& qdata) {
    const auto& cfg = problem.cfg;
    const int group_count = cfg.k / cfg.group_size;
    const auto logical_qweight = materialize_logical_qweight(problem);
    std::vector<float> output(cfg.m * cfg.n, 0.0f);

    for (int row = 0; row < cfg.m; ++row) {
        for (int group = 0; group < group_count; ++group) {
            const int begin = group * cfg.group_size;
            const int end = begin + cfg.group_size;
            const float src_scale = qdata.src_scales[index2d(row, group, group_count)];
            const int32_t src_sum = qdata.grouped_sums[index2d(row, group, group_count)];

            for (int col = 0; col < cfg.n; ++col) {
                int32_t dot = 0;
                for (int k_idx = begin; k_idx < end; ++k_idx) {
                    dot += qdata.qsrc[index2d(row, k_idx, cfg.k)] * logical_qweight[index2d(k_idx, col, cfg.n)];
                }

                const float wei_scale = problem.weight_scales[index2d(group, col, cfg.n)];
                const float wei_zp = problem.weight_zero_points[index2d(group, col, cfg.n)];
                const float scale = src_scale * wei_scale;
                const float compensation = static_cast<float>(src_sum) * wei_zp * scale;
                output[index2d(row, col, cfg.n)] += static_cast<float>(dot) * scale - compensation;
            }
        }
    }

    for (int row = 0; row < cfg.m; ++row) {
        for (int col = 0; col < cfg.n; ++col) {
            output[index2d(row, col, cfg.n)] += problem.bias[col];
        }
    }

    return output;
}

void print_error_stats(const std::string& name, const std::vector<float>& reference, const std::vector<float>& candidate) {
    double max_abs_diff = 0.0;
    double mean_abs_diff = 0.0;
    double mse = 0.0;

    for (size_t i = 0; i < reference.size(); ++i) {
        const double diff = static_cast<double>(candidate[i]) - static_cast<double>(reference[i]);
        const double abs_diff = std::abs(diff);
        max_abs_diff = std::max(max_abs_diff, abs_diff);
        mean_abs_diff += abs_diff;
        mse += diff * diff;
    }

    mean_abs_diff /= static_cast<double>(reference.size());
    mse /= static_cast<double>(reference.size());

    std::cout << "[" << name << "] max abs diff: " << std::fixed << std::setprecision(8) << max_abs_diff << "\n";
    std::cout << "[" << name << "] mean abs diff: " << std::fixed << std::setprecision(8) << mean_abs_diff << "\n";
    std::cout << "[" << name << "] rmse: " << std::fixed << std::setprecision(8) << std::sqrt(mse) << "\n";
}

void run_case(const ProblemConfig& cfg, std::mt19937& rng) {
    const Problem problem = make_problem(cfg, rng);
    const auto reference = reference_matmul(problem);
    const auto qdata = quantize_source_dynamic(problem);
    const auto candidate = sdq_matmul(problem, qdata);

    std::cout << "============================================================\n";
    std::cout << "Case: weight_format=" << to_string(cfg.weight_format) << "\n";
    std::cout << "shape: M=" << cfg.m << ", K=" << cfg.k << ", N=" << cfg.n << ", group_size=" << cfg.group_size << "\n";
    if (is_nibble_format(cfg.weight_format)) {
        std::vector<int32_t> nibble_source(problem.qweight.size(), 0);
        for (size_t i = 0; i < problem.qweight.size(); ++i) {
            nibble_source[i] = problem.qweight[i] & 0x0F;
        }
        const auto packed = pack_nibbles(nibble_source);
        std::cout << "packed nibble bytes count=" << packed.size() << ", logical values count=" << problem.qweight.size() << "\n";
    }
    print_error_stats(to_string(cfg.weight_format), reference, candidate);

    std::cout << "reference[0, 0..7]: ";
    for (int col = 0; col < std::min(8, cfg.n); ++col) {
        std::cout << std::fixed << std::setprecision(5) << reference[index2d(0, col, cfg.n)] << ' ';
    }
    std::cout << "\n";

    std::cout << "candidate[0, 0..7]: ";
    for (int col = 0; col < std::min(8, cfg.n); ++col) {
        std::cout << std::fixed << std::setprecision(5) << candidate[index2d(0, col, cfg.n)] << ' ';
    }
    std::cout << "\n";
}

int main() {
    std::mt19937 rng(7);

    ProblemConfig u8_cfg;
    u8_cfg.weight_format = WeightFormat::U8;
    run_case(u8_cfg, rng);

    ProblemConfig i8_cfg;
    i8_cfg.weight_format = WeightFormat::I8;
    run_case(i8_cfg, rng);

    ProblemConfig u4_cfg;
    u4_cfg.weight_format = WeightFormat::U4;
    run_case(u4_cfg, rng);

    ProblemConfig i4_cfg;
    i4_cfg.weight_format = WeightFormat::I4;
    run_case(i4_cfg, rng);

    return 0;
}
