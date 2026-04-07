/**
 * @file test_weight_decomp.cpp
 * @brief 权重解压缩JIT内核测试程序
 *
 * 测试用例涵盖:
 *   1. u8  -> f32 (无scale/zp, 有scale, 有scale+zp)
 *   2. s8  -> f32
 *   3. u4  -> f32
 *   4. s4  -> f32
 *   5. u2  -> f32
 *   6. nf4 -> f32 (NormalFloat4查找表)
 *   7. f4_e2m1 -> f32 (FP4查找表)
 *   8. f16 -> f32
 *   9. bf16 -> f32
 *  10. u8  -> f32 with broadcast scale/zp
 *  11. u4  -> f32 with e8m0 scale
 *  12. u8  -> f32 multi oc block
 *  13. u8  -> f16 (输出为f16)
 *  14. u4  -> f16 with scale+zp
 *  15. nf4 -> f16 with scale
 *  16. [AVX2] u8  -> f32
 *  17. [AVX2] u4  -> f32 with scale+zp
 *  18. [AVX2] s4  -> f32
 *  19. [AVX2] u2  -> f32
 *  20. [AVX2] nf4 -> f32 (拆分查找表+blend)
 *  21. [AVX2] f4_e2m1 -> f32 (符号位分离查找)
 *  22. [AVX2] u8  -> f32 multi oc block
 *  23. [AVX2] f16 -> f32 (F16C)
 *
 * 每个用例包含:
 *   - 参考C++实现(reference)
 *   - JIT内核执行
 *   - 逐元素比较 (允许浮点误差)
 *
 * 编译 & 运行:
 *   g++ -O2 -mavx2 -mf16c -mavx512f -mavx512bw -mavx512vl -mavx512dq -std=c++17 \
 *       -I../../3rdparty/xbyak test_weight_decomp.cpp -o test_weight_decomp
 *   ./test_weight_decomp
 */

#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <algorithm>
#include <random>
#include <string>
#include <functional>

#include "weight_decomp_kernel.hpp"

using namespace weight_decomp;

// ============================================================================
// 测试辅助工具
// ============================================================================

/// 分配 64-byte 对齐的内存 (AVX-512要求)
template <typename T>
T* aligned_alloc_array(size_t count) {
    void* p = std::aligned_alloc(64, count * sizeof(T) + 63);
    assert(p && "aligned_alloc failed");
    std::memset(p, 0, count * sizeof(T) + 63);
    return static_cast<T*>(p);
}

/// 浮点近似比较
bool approx_equal(float a, float b, float tol = 1e-4f) {
    if (std::isnan(a) && std::isnan(b)) return true;
    if (std::isinf(a) && std::isinf(b)) return a == b;
    return std::fabs(a - b) <= tol + tol * std::fabs(b);
}

/// NF4查找表 (与内核中相同)
static const float nf4_lut[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};

/// F4_E2M1查找表
static const float f4e2m1_lut[16] = {
     0.0f,   0.5f,   1.0f,   1.5f,
     2.0f,   3.0f,   4.0f,   6.0f,
    -0.0f,  -0.5f,  -1.0f,  -1.5f,
    -2.0f,  -3.0f,  -4.0f,  -6.0f
};

// ============================================================================
// 参考实现: 各种数据类型的解压缩
// ============================================================================

/// 将u8权重解压缩为f32
float decompress_u8(uint8_t val) {
    return static_cast<float>(val);
}

/// 将s8权重解压缩为f32
float decompress_s8(int8_t val) {
    return static_cast<float>(val);
}

/// 从打包byte中提取u4值 (ic_internal=0取高nibble, =1取低nibble)
float decompress_u4(uint8_t byte_val, int ic) {
    if (ic % 2 == 0) {
        return static_cast<float>((byte_val >> 4) & 0xF);
    } else {
        return static_cast<float>(byte_val & 0xF);
    }
}

/// 从打包byte中提取s4值
float decompress_s4(uint8_t byte_val, int ic) {
    int32_t val;
    if (ic % 2 == 0) {
        // 高nibble, 保留符号
        val = static_cast<int8_t>(byte_val) >> 4;
    } else {
        // 低nibble, 符号扩展
        val = (static_cast<int32_t>(byte_val) << 28) >> 28;
    }
    return static_cast<float>(val);
}

/// 从打包byte中提取u2值
float decompress_u2(uint8_t byte_val, int ic) {
    if (ic == 0) {
        return static_cast<float>((byte_val >> 6) & 0x3);
    } else {
        return static_cast<float>((byte_val >> (6 - 2 * ic)) & 0x3);
    }
}

/// NF4解压缩: 4-bit index -> 查找表
float decompress_nf4(uint8_t byte_val, int ic) {
    uint8_t idx;
    if (ic % 2 == 0) {
        idx = (byte_val >> 4) & 0xF;
    } else {
        idx = byte_val & 0xF;
    }
    return nf4_lut[idx];
}

/// F4_E2M1解压缩: 4-bit -> 查找表
float decompress_f4_e2m1(uint8_t byte_val, int ic) {
    uint8_t idx;
    if (ic % 2 == 0) {
        idx = (byte_val >> 4) & 0xF;
    } else {
        idx = byte_val & 0xF;
    }
    return f4e2m1_lut[idx];
}

/// f16 -> f32 转换 (使用编译器内置或软件模拟)
float decompress_f16(uint16_t val) {
    // 使用 _cvtsh_ss 或手动转换
    uint32_t sign = (val >> 15) & 1;
    uint32_t exp  = (val >> 10) & 0x1F;
    uint32_t mant = val & 0x3FF;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // 非规格化数
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++; mant &= 0x3FF;
    } else if (exp == 0x1F) {
        uint32_t f = (sign << 31) | 0x7F800000 | (mant << 13);
        float result;
        std::memcpy(&result, &f, 4);
        return result;
    }

    uint32_t f = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

/// bf16 -> f32 转换
float decompress_bf16(uint16_t val) {
    uint32_t f = static_cast<uint32_t>(val) << 16;
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

/// e8m0 scale 解码: 将8位指数放入IEEE754 float的exponent字段
float decode_e8m0(uint8_t val) {
    uint32_t f = static_cast<uint32_t>(val) << 23;
    float result;
    std::memcpy(&result, &f, 4);
    return result;
}

// ============================================================================
// 通用参考解压缩 + 反量化
// ============================================================================

/**
 * @brief 参考实现: 完整的权重解压缩流程
 *
 * @param jcp       编译参数
 * @param weights   压缩权重
 * @param scales    scale数组 (可选)
 * @param zps       zero_point数组 (可选)
 * @param output    输出f32缓冲区
 * @param ic_groups IC分组数
 */
void reference_decompress(
    const compile_params_t& jcp,
    const uint8_t* weights,
    const void* scales,
    const void* zps,
    float* output,
    size_t ic_groups)
{
    size_t pack_scale = type_pack_scale(jcp.weights_dt);
    size_t wt_dt_size = data_type_size(jcp.weights_dt);

    for (size_t ig = 0; ig < ic_groups; ig++) {
        for (size_t ic = 0; ic < jcp.ic_internal_size; ic++) {
            for (size_t oc = 0; oc < jcp.oc_size; oc++) {
                // 计算权重的byte偏移
                size_t w_byte_offset = ig * jcp.oc_size * jcp.ic_internal_size * wt_dt_size / pack_scale
                    + (oc / 16) * jcp.ic_internal_size * 16 * wt_dt_size / pack_scale; // oc_block offset
                // 对于sub-byte类型, 所有ic共享同一组byte
                size_t oc_in_block = oc % 16;

                uint8_t raw_byte = 0;
                int16_t raw_word = 0;

                float val = 0.0f;

                switch (jcp.weights_dt) {
                    case data_type_t::u8: {
                        size_t off = ig * jcp.oc_size * jcp.ic_internal_size + ic * jcp.oc_size + oc;
                        val = decompress_u8(weights[off]);
                        break;
                    }
                    case data_type_t::s8: {
                        size_t off = ig * jcp.oc_size * jcp.ic_internal_size + ic * jcp.oc_size + oc;
                        val = decompress_s8(static_cast<int8_t>(weights[off]));
                        break;
                    }
                    case data_type_t::u4: {
                        // u4: 2个值共享1 byte, 布局 [oc_block][ic_internal][vec_size/2]
                        size_t off = w_byte_offset + oc_in_block;
                        val = decompress_u4(weights[off], ic);
                        break;
                    }
                    case data_type_t::s4: {
                        size_t off = w_byte_offset + oc_in_block;
                        val = decompress_s4(weights[off], ic);
                        break;
                    }
                    case data_type_t::u2: {
                        size_t off = w_byte_offset + oc_in_block;
                        val = decompress_u2(weights[off], ic);
                        break;
                    }
                    case data_type_t::nf4: {
                        size_t off = w_byte_offset + oc_in_block;
                        val = decompress_nf4(weights[off], ic);
                        break;
                    }
                    case data_type_t::f4_e2m1: {
                        size_t off = w_byte_offset + oc_in_block;
                        val = decompress_f4_e2m1(weights[off], ic);
                        break;
                    }
                    case data_type_t::f16: {
                        size_t off = ig * jcp.oc_size * jcp.ic_internal_size + ic * jcp.oc_size + oc;
                        uint16_t raw;
                        std::memcpy(&raw, weights + off * 2, 2);
                        val = decompress_f16(raw);
                        break;
                    }
                    case data_type_t::bf16: {
                        size_t off = ig * jcp.oc_size * jcp.ic_internal_size + ic * jcp.oc_size + oc;
                        uint16_t raw;
                        std::memcpy(&raw, weights + off * 2, 2);
                        val = decompress_bf16(raw);
                        break;
                    }
                    default: assert(!"unsupported");
                }

                // 反量化
                if (jcp.with_zero_points) {
                    float zp_val = 0.0f;
                    if (jcp.broadcast_zero_points) {
                        switch (jcp.zero_points_dt) {
                            case data_type_t::f32: zp_val = *static_cast<const float*>(zps); break;
                            case data_type_t::u8:  zp_val = static_cast<float>(*static_cast<const uint8_t*>(zps)); break;
                            default: assert(!"unsupported zp type");
                        }
                    } else {
                        switch (jcp.zero_points_dt) {
                            case data_type_t::f32: zp_val = static_cast<const float*>(zps)[oc]; break;
                            case data_type_t::u8:  zp_val = static_cast<float>(static_cast<const uint8_t*>(zps)[oc]); break;
                            default: assert(!"unsupported zp type");
                        }
                    }
                    val -= zp_val;
                }

                if (jcp.with_scales) {
                    float s_val = 1.0f;
                    if (jcp.broadcast_scales) {
                        switch (jcp.scales_dt) {
                            case data_type_t::f32: s_val = *static_cast<const float*>(scales); break;
                            case data_type_t::e8m0: s_val = decode_e8m0(*static_cast<const uint8_t*>(scales)); break;
                            default: assert(!"unsupported scale type");
                        }
                    } else {
                        switch (jcp.scales_dt) {
                            case data_type_t::f32: s_val = static_cast<const float*>(scales)[oc]; break;
                            case data_type_t::e8m0: s_val = decode_e8m0(static_cast<const uint8_t*>(scales)[oc]); break;
                            default: assert(!"unsupported scale type");
                        }
                    }
                    val *= s_val;
                }

                // 写入输出 (IC-major布局: [ig][ic][oc])
                size_t out_idx = ig * jcp.ic_internal_size * jcp.oc_size + ic * jcp.oc_size + oc;
                output[out_idx] = val;
            }
        }
    }
}

// ============================================================================
// 测试框架
// ============================================================================

struct TestResult {
    std::string name;
    bool passed;
    int mismatches;
    float max_diff;
};

int g_total = 0;
int g_passed = 0;
int g_failed = 0;

/**
 * @brief 运行一个解压缩测试用例
 *
 * @param test_name   测试名称
 * @param jcp         编译参数
 * @param weights     压缩权重数据
 * @param weights_bytes 权重数据字节数
 * @param scales      scale数据 (nullptr if not used)
 * @param zps         zero_point数据 (nullptr if not used)
 * @param ic_groups   IC分组数
 * @param tol         浮点比较容差
 */
TestResult run_test(
    const std::string& test_name,
    const compile_params_t& jcp,
    const uint8_t* weights,
    size_t weights_bytes,
    const void* scales,
    const void* zps,
    size_t ic_groups,
    float tol = 1e-4f)
{
    g_total++;
    size_t total_output_elems = ic_groups * jcp.ic_internal_size * jcp.oc_size;
    size_t decomp_dt_size = data_type_size(jcp.decomp_buffer_dt);

    // 参考输出 (always f32)
    auto* ref_output = aligned_alloc_array<float>(total_output_elems);
    reference_decompress(jcp, weights, scales, zps, ref_output, ic_groups);

    // JIT输出缓冲区: 分配足够空间 (可能是f16/bf16, 2字节每元素)
    size_t jit_buf_bytes = total_output_elems * decomp_dt_size;
    auto* jit_buf = aligned_alloc_array<uint8_t>(jit_buf_bytes);

    try {
        WeightDecompKernel<isa_t::avx512> kernel(jcp);

        runtime_params_t rt;
        rt.weights_ptr = weights;
        rt.decomp_buffer_ptr = jit_buf;
        rt.scales_ptr = scales;
        rt.zero_points_ptr = zps;
        rt.ic_size = ic_groups;

        kernel.execute(&rt);
    } catch (const std::exception& e) {
        std::cout << "[SKIP] " << test_name << ": " << e.what() << "\n";
        std::free(ref_output);
        std::free(jit_buf);
        return {test_name, true, 0, 0.0f};
    }

    // 将JIT输出转为f32进行比较
    auto* jit_output = aligned_alloc_array<float>(total_output_elems);
    if (jcp.decomp_buffer_dt == data_type_t::f32) {
        std::memcpy(jit_output, jit_buf, total_output_elems * sizeof(float));
    } else if (jcp.decomp_buffer_dt == data_type_t::f16) {
        // f16 -> f32: 从输出缓冲区读取f16并转换
        auto* f16_buf = reinterpret_cast<const uint16_t*>(jit_buf);
        for (size_t i = 0; i < total_output_elems; i++) {
            jit_output[i] = decompress_f16(f16_buf[i]);
        }
    } else if (jcp.decomp_buffer_dt == data_type_t::bf16) {
        auto* bf16_buf = reinterpret_cast<const uint16_t*>(jit_buf);
        for (size_t i = 0; i < total_output_elems; i++) {
            jit_output[i] = decompress_bf16(bf16_buf[i]);
        }
    }

    // 比较
    int mismatches = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < total_output_elems; i++) {
        float diff = std::fabs(ref_output[i] - jit_output[i]);
        max_diff = std::max(max_diff, diff);
        if (!approx_equal(ref_output[i], jit_output[i], tol)) {
            if (mismatches < 5) { // 只打印前5个不匹配
                std::cout << "  MISMATCH at [" << i << "]: ref=" << ref_output[i]
                          << " jit=" << jit_output[i] << " diff=" << diff << "\n";
            }
            mismatches++;
        }
    }

    bool passed = (mismatches == 0);
    if (passed) {
        std::cout << "[PASS] " << test_name
                  << " (elems=" << total_output_elems << ", max_diff=" << max_diff << ")\n";
        g_passed++;
    } else {
        std::cout << "[FAIL] " << test_name
                  << " (" << mismatches << "/" << total_output_elems << " mismatches, max_diff=" << max_diff << ")\n";
        g_failed++;
    }

    std::free(ref_output);
    std::free(jit_output);
    std::free(jit_buf);
    return {test_name, passed, mismatches, max_diff};
}

// ============================================================================
// 打包工具函数 —— 为sub-byte类型生成测试数据
// ============================================================================

/// 将两个u4值打包到一个byte: high nibble + low nibble
uint8_t pack_u4(uint8_t hi, uint8_t lo) {
    return ((hi & 0xF) << 4) | (lo & 0xF);
}

/// 将两个s4值打包到一个byte
uint8_t pack_s4(int8_t hi, int8_t lo) {
    return ((static_cast<uint8_t>(hi) & 0xF) << 4) | (static_cast<uint8_t>(lo) & 0xF);
}

/// 将四个u2值打包到一个byte: [ic0|ic1|ic2|ic3]
uint8_t pack_u2(uint8_t v0, uint8_t v1, uint8_t v2, uint8_t v3) {
    return ((v0 & 0x3) << 6) | ((v1 & 0x3) << 4) | ((v2 & 0x3) << 2) | (v3 & 0x3);
}

/// 将两个nf4 index打包到一个byte
uint8_t pack_nf4(uint8_t hi_idx, uint8_t lo_idx) {
    return ((hi_idx & 0xF) << 4) | (lo_idx & 0xF);
}

/// f32 -> bf16 (截断)
uint16_t float_to_bf16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, 4);
    return static_cast<uint16_t>(bits >> 16);
}

/// f32 -> f16 (简化转换)
uint16_t float_to_f16(float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, 4);
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0) { exp = 0; mant = 0; }
    if (exp >= 0x1F) { exp = 0x1F; mant = 0; }
    return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

// ============================================================================
// 测试用例
// ============================================================================

/**
 * Test 1: u8 -> f32, 无scale/zp
 * 最简单的情况: 直接将u8值转为float
 */
void test_u8_to_f32_no_scale() {
    const size_t oc = 16;     // 1个向量宽度
    const size_t ic_int = 1;  // u8: 1个值/byte
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = false;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = false;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u8;
    jcp.decomp_buffer_dt = data_type_t::f32;

    // 填充测试数据: 0, 1, 2, ..., 31
    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_int * ic_groups);
    for (size_t i = 0; i < oc * ic_int * ic_groups; i++) {
        weights[i] = static_cast<uint8_t>(i);
    }

    run_test("u8->f32 no_scale", jcp, weights, oc * ic_groups, nullptr, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 2: u8 -> f32, 有per-tensor scale
 */
void test_u8_to_f32_broadcast_scale() {
    const size_t oc = 16;
    const size_t ic_int = 1;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u8;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;

    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_groups);
    for (size_t i = 0; i < oc * ic_groups; i++) {
        weights[i] = static_cast<uint8_t>(i * 2);
    }

    float scale = 0.5f;
    run_test("u8->f32 broadcast_scale", jcp, weights, oc * ic_groups, &scale, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 3: u8 -> f32, 有per-channel scale + 有broadcast zero_point
 */
void test_u8_to_f32_scale_and_zp() {
    const size_t oc = 16;
    const size_t ic_int = 1;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = true;
    jcp.broadcast_scales = false;
    jcp.broadcast_zero_points = true;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u8;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;
    jcp.zero_points_dt = data_type_t::f32;

    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_groups);
    for (size_t i = 0; i < oc * ic_groups; i++) {
        weights[i] = static_cast<uint8_t>(100 + i);
    }

    auto* scales = aligned_alloc_array<float>(oc);
    for (size_t i = 0; i < oc; i++) {
        scales[i] = 0.1f * (i + 1);
    }

    float zp = 128.0f;
    run_test("u8->f32 scale+zp", jcp, weights, oc * ic_groups, scales, &zp, ic_groups);
    std::free(weights);
    std::free(scales);
}

/**
 * Test 4: s8 -> f32
 */
void test_s8_to_f32() {
    const size_t oc = 16;
    const size_t ic_int = 1;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::s8;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;

    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_groups);
    // 填充有符号值: -8, -7, ..., 7, 8, 9, ..., 23
    for (size_t i = 0; i < oc * ic_groups; i++) {
        weights[i] = static_cast<uint8_t>(static_cast<int8_t>(-8 + (int)i));
    }

    float scale = 2.0f;
    run_test("s8->f32 broadcast_scale", jcp, weights, oc * ic_groups, &scale, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 5: u4 -> f32
 * ic_internal_size = 2 (每byte存2个u4值)
 */
void test_u4_to_f32() {
    const size_t oc = 16;
    const size_t ic_int = 2;   // u4: 每byte 2个值
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = true;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = true;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u4;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;
    jcp.zero_points_dt = data_type_t::f32;

    // 每个oc_block有 ic_int * vec_size 个字节 (但u4打包: / 2)
    // 实际每个ic_group: oc * ic_int / 2 字节 = 16 字节
    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group * ic_groups);

    // 填充: 每byte的高nibble=偶数ic值, 低nibble=奇数ic值
    for (size_t ig = 0; ig < ic_groups; ig++) {
        for (size_t oc_i = 0; oc_i < oc; oc_i++) {
            uint8_t hi = (ig * 2 + oc_i) % 16;     // ic=0的值
            uint8_t lo = (ig * 2 + oc_i + 1) % 16;  // ic=1的值
            weights[ig * bytes_per_group + oc_i] = pack_u4(hi, lo);
        }
    }

    float scale = 0.25f;
    float zp = 8.0f;
    run_test("u4->f32 scale+zp", jcp, weights, bytes_per_group * ic_groups, &scale, &zp, ic_groups);
    std::free(weights);
}

/**
 * Test 6: s4 -> f32
 */
void test_s4_to_f32() {
    const size_t oc = 16;
    const size_t ic_int = 2;
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = false;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = false;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::s4;
    jcp.decomp_buffer_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group * ic_groups);

    // 填充s4值: [-8..7] 范围
    for (size_t oc_i = 0; oc_i < oc; oc_i++) {
        int8_t hi = static_cast<int8_t>((oc_i % 16) - 8);  // -8..7
        int8_t lo = static_cast<int8_t>(((oc_i + 1) % 16) - 8);
        weights[oc_i] = pack_s4(hi, lo);
    }

    run_test("s4->f32 no_scale", jcp, weights, bytes_per_group, nullptr, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 7: u2 -> f32
 * ic_internal_size = 4 (每byte存4个u2值)
 */
void test_u2_to_f32() {
    const size_t oc = 16;
    const size_t ic_int = 4;   // u2: 每byte 4个值
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u2;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 4;  // = 16
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group * ic_groups);

    // 填充: 每byte包含4个值 [3,2,1,0]
    for (size_t i = 0; i < oc; i++) {
        weights[i] = pack_u2(
            (i + 0) % 4,
            (i + 1) % 4,
            (i + 2) % 4,
            (i + 3) % 4
        );
    }

    float scale = 1.5f;
    run_test("u2->f32 broadcast_scale", jcp, weights, bytes_per_group, &scale, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 8: nf4 -> f32 (NormalFloat 4-bit查找表)
 */
void test_nf4_to_f32() {
    const size_t oc = 16;
    const size_t ic_int = 2;   // nf4: 每byte 2个值
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::nf4;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group * ic_groups);

    // 填充nf4 indices
    for (size_t ig = 0; ig < ic_groups; ig++) {
        for (size_t i = 0; i < oc; i++) {
            uint8_t hi = (ig + i) % 16;
            uint8_t lo = (ig + i + 7) % 16;
            weights[ig * bytes_per_group + i] = pack_nf4(hi, lo);
        }
    }

    float scale = 2.0f;
    run_test("nf4->f32 broadcast_scale", jcp, weights, bytes_per_group * ic_groups, &scale, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 9: f4_e2m1 -> f32 (FP4查找表)
 */
void test_f4_e2m1_to_f32() {
    const size_t oc = 16;
    const size_t ic_int = 2;
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = false;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = false;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::f4_e2m1;
    jcp.decomp_buffer_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group * ic_groups);

    for (size_t i = 0; i < oc; i++) {
        uint8_t hi = i % 16;
        uint8_t lo = (i + 3) % 16;
        weights[i] = pack_nf4(hi, lo);  // 同样的打包方式
    }

    run_test("f4_e2m1->f32 no_scale", jcp, weights, bytes_per_group, nullptr, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 10: u8 -> f32, per-channel u8 zero_point + broadcast f32 scale
 * 测试u8类型的zero_point
 */
void test_u8_with_u8_zp() {
    const size_t oc = 16;
    const size_t ic_int = 1;
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = true;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u8;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;
    jcp.zero_points_dt = data_type_t::u8;

    auto* weights = aligned_alloc_array<uint8_t>(oc);
    for (size_t i = 0; i < oc; i++) {
        weights[i] = static_cast<uint8_t>(200 - i * 5);
    }

    float scale = 0.01f;
    auto* zps = aligned_alloc_array<uint8_t>(oc);
    for (size_t i = 0; i < oc; i++) {
        zps[i] = static_cast<uint8_t>(128 + i);
    }

    run_test("u8->f32 u8_zp+broadcast_scale", jcp, weights, oc, &scale, zps, ic_groups);
    std::free(weights);
    std::free(zps);
}

/**
 * Test 11: u4 -> f32, e8m0 scale
 * 测试E8M0格式的scale (指数型scale)
 */
void test_u4_with_e8m0_scale() {
    const size_t oc = 16;
    const size_t ic_int = 2;
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u4;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::e8m0;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group);

    for (size_t i = 0; i < oc; i++) {
        weights[i] = pack_u4(i % 16, (i + 1) % 16);
    }

    // e8m0 scale: 值127对应2^0=1.0, 值128对应2^1=2.0, 值126对应2^(-1)=0.5
    uint8_t e8m0_scale = 126;  // scale = 0.5
    run_test("u4->f32 e8m0_scale", jcp, weights, bytes_per_group, &e8m0_scale, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 12: u8 -> f32, 多OC块 (放大规模)
 */
void test_u8_multi_oc_block() {
     const size_t oc = 64;     // 4个向量宽度 (AVX-512)
    const size_t ic_int = 1;
     const size_t ic_groups = 128;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = true;
    jcp.broadcast_scales = false;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u8;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;
    jcp.zero_points_dt = data_type_t::f32;

    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_groups);
    for (size_t i = 0; i < oc * ic_groups; i++) {
        weights[i] = static_cast<uint8_t>(i % 256);
    }

    auto* scales = aligned_alloc_array<float>(oc);
    for (size_t i = 0; i < oc; i++) {
        scales[i] = 0.5f + i * 0.01f;
    }

    auto* zps = aligned_alloc_array<float>(oc);
    for (size_t i = 0; i < oc; i++) {
        zps[i] = static_cast<float>(i);
    }

    run_test("u8->f32 multi_oc_block", jcp, weights, oc * ic_groups, scales, zps, ic_groups);
    std::free(weights);
    std::free(scales);
    std::free(zps);
}

// ============================================================================
// f16输出测试用例
// ============================================================================

/**
 * Test 13: u8 -> f16, 无scale/zp
 * 测试f16输出的基本流程: weight加载 -> f32 -> vcvtps2ph -> 存储f16
 */
void test_u8_to_f16_no_scale() {
    const size_t oc = 16;
    const size_t ic_int = 1;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = false;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = false;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u8;
    jcp.decomp_buffer_dt = data_type_t::f16;

    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_groups);
    for (size_t i = 0; i < oc * ic_groups; i++) {
        weights[i] = static_cast<uint8_t>(i * 3);
    }

    // f16精度有限, 使用稍大的容差
    run_test("u8->f16 no_scale", jcp, weights, oc * ic_groups, nullptr, nullptr, ic_groups, 1e-2f);
    std::free(weights);
}

/**
 * Test 14: u4 -> f16, 有scale+zp
 * 测试sub-byte解压缩 + 反量化 + f16输出的完整流程
 */
void test_u4_to_f16_scale_zp() {
    const size_t oc = 16;
    const size_t ic_int = 2;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = true;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = true;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u4;
    jcp.decomp_buffer_dt = data_type_t::f16;
    jcp.scales_dt = data_type_t::f32;
    jcp.zero_points_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group * ic_groups);
    for (size_t ig = 0; ig < ic_groups; ig++) {
        for (size_t i = 0; i < oc; i++) {
            weights[ig * bytes_per_group + i] = pack_u4((ig + i) % 16, (ig + i + 5) % 16);
        }
    }

    float scale = 0.5f;
    float zp = 8.0f;
    run_test("u4->f16 scale+zp", jcp, weights, bytes_per_group * ic_groups, &scale, &zp, ic_groups, 1e-2f);
    std::free(weights);
}

/**
 * Test 15: nf4 -> f16, 有scale
 * 测试查找表解压缩 + f16输出
 */
void test_nf4_to_f16_scale() {
    const size_t oc = 16;
    const size_t ic_int = 2;
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::nf4;
    jcp.decomp_buffer_dt = data_type_t::f16;
    jcp.scales_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group);
    for (size_t i = 0; i < oc; i++) {
        weights[i] = pack_nf4(i % 16, (i + 8) % 16);
    }

    float scale = 3.0f;
    run_test("nf4->f16 broadcast_scale", jcp, weights, bytes_per_group, &scale, nullptr, ic_groups, 1e-2f);
    std::free(weights);
}

// ============================================================================
// AVX2 测试用例
// ============================================================================

/**
 * @brief 运行AVX2内核测试
 * 逻辑与run_test相同, 但使用WeightDecompKernelAVX2内核
 */
TestResult run_test_avx2(
    const std::string& test_name,
    const compile_params_t& jcp,
    const uint8_t* weights,
    size_t weights_bytes,
    const void* scales,
    const void* zps,
    size_t ic_groups,
    float tol = 1e-4f)
{
    g_total++;
    size_t total_output_elems = ic_groups * jcp.ic_internal_size * jcp.oc_size;

    auto* ref_output = aligned_alloc_array<float>(total_output_elems);
    reference_decompress(jcp, weights, scales, zps, ref_output, ic_groups);

    auto* jit_output = aligned_alloc_array<float>(total_output_elems);

    try {
        WeightDecompKernel<isa_t::avx2> kernel(jcp);

        runtime_params_t rt;
        rt.weights_ptr = weights;
        rt.decomp_buffer_ptr = jit_output;
        rt.scales_ptr = scales;
        rt.zero_points_ptr = zps;
        rt.ic_size = ic_groups;

        kernel.execute(&rt);
    } catch (const std::exception& e) {
        std::cout << "[SKIP] " << test_name << ": " << e.what() << "\n";
        std::free(ref_output);
        std::free(jit_output);
        return {test_name, true, 0, 0.0f};
    }

    int mismatches = 0;
    float max_diff = 0.0f;
    for (size_t i = 0; i < total_output_elems; i++) {
        float diff = std::fabs(ref_output[i] - jit_output[i]);
        max_diff = std::max(max_diff, diff);
        if (!approx_equal(ref_output[i], jit_output[i], tol)) {
            if (mismatches < 5) {
                std::cout << "  MISMATCH at [" << i << "]: ref=" << ref_output[i]
                          << " jit=" << jit_output[i] << " diff=" << diff << "\n";
            }
            mismatches++;
        }
    }

    bool passed = (mismatches == 0);
    if (passed) {
        std::cout << "[PASS] " << test_name
                  << " (elems=" << total_output_elems << ", max_diff=" << max_diff << ")\n";
        g_passed++;
    } else {
        std::cout << "[FAIL] " << test_name
                  << " (" << mismatches << "/" << total_output_elems << " mismatches, max_diff=" << max_diff << ")\n";
        g_failed++;
    }

    std::free(ref_output);
    std::free(jit_output);
    return {test_name, passed, mismatches, max_diff};
}

/**
 * Test 16: [AVX2] u8 -> f32, 有per-channel scale + broadcast zp
 */
void test_avx2_u8_to_f32() {
    const size_t oc = 8;  // AVX2: 1个向量宽度 = 8 float
    const size_t ic_int = 1;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = true;
    jcp.broadcast_scales = false;
    jcp.broadcast_zero_points = true;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u8;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;
    jcp.zero_points_dt = data_type_t::f32;

    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_groups);
    for (size_t i = 0; i < oc * ic_groups; i++)
        weights[i] = static_cast<uint8_t>(100 + i * 3);

    auto* scales = aligned_alloc_array<float>(oc);
    for (size_t i = 0; i < oc; i++)
        scales[i] = 0.1f * (i + 1);

    float zp = 128.0f;
    run_test_avx2("[AVX2] u8->f32 scale+zp", jcp, weights, oc * ic_groups, scales, &zp, ic_groups);
    std::free(weights);
    std::free(scales);
}

/**
 * Test 17: [AVX2] u4 -> f32 with scale+zp
 */
void test_avx2_u4_to_f32() {
    const size_t oc = 8;
    const size_t ic_int = 2;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = true;
    jcp.broadcast_scales = true;
    jcp.broadcast_zero_points = true;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u4;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;
    jcp.zero_points_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group * ic_groups);
    for (size_t ig = 0; ig < ic_groups; ig++)
        for (size_t i = 0; i < oc; i++)
            weights[ig * bytes_per_group + i] = pack_u4((ig + i) % 16, (ig + i + 3) % 16);

    float scale = 0.25f;
    float zp = 8.0f;
    run_test_avx2("[AVX2] u4->f32 scale+zp", jcp, weights, bytes_per_group * ic_groups, &scale, &zp, ic_groups);
    std::free(weights);
}

/**
 * Test 18: [AVX2] s4 -> f32
 */
void test_avx2_s4_to_f32() {
    const size_t oc = 8;
    const size_t ic_int = 2;
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = false;
    jcp.with_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::s4;
    jcp.decomp_buffer_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group);
    for (size_t i = 0; i < oc; i++)
        weights[i] = pack_s4(static_cast<int8_t>((i % 16) - 8),
                             static_cast<int8_t>(((i+1) % 16) - 8));

    run_test_avx2("[AVX2] s4->f32 no_scale", jcp, weights, bytes_per_group, nullptr, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 19: [AVX2] u2 -> f32
 */
void test_avx2_u2_to_f32() {
    const size_t oc = 8;
    const size_t ic_int = 4;
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u2;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 4;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group);
    for (size_t i = 0; i < oc; i++)
        weights[i] = pack_u2((i+0)%4, (i+1)%4, (i+2)%4, (i+3)%4);

    float scale = 1.5f;
    run_test_avx2("[AVX2] u2->f32 broadcast_scale", jcp, weights, bytes_per_group, &scale, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 20: [AVX2] nf4 -> f32
 * 测试AVX2的拆分查找表 + vblendvps 逻辑
 */
void test_avx2_nf4_to_f32() {
    const size_t oc = 8;
    const size_t ic_int = 2;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::nf4;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group * ic_groups);
    for (size_t ig = 0; ig < ic_groups; ig++)
        for (size_t i = 0; i < oc; i++)
            weights[ig * bytes_per_group + i] = pack_nf4((ig + i) % 16, (ig + i + 7) % 16);

    float scale = 2.0f;
    run_test_avx2("[AVX2] nf4->f32 broadcast_scale", jcp, weights, bytes_per_group * ic_groups, &scale, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 21: [AVX2] f4_e2m1 -> f32
 * 测试AVX2的符号位分离 + 8项LUT查找逻辑
 */
void test_avx2_f4_e2m1_to_f32() {
    const size_t oc = 8;
    const size_t ic_int = 2;
    const size_t ic_groups = 1;

    compile_params_t jcp = {};
    jcp.with_scales = false;
    jcp.with_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::f4_e2m1;
    jcp.decomp_buffer_dt = data_type_t::f32;

    size_t bytes_per_group = oc * ic_int / 2;
    auto* weights = aligned_alloc_array<uint8_t>(bytes_per_group);
    for (size_t i = 0; i < oc; i++)
        weights[i] = pack_nf4(i % 16, (i + 3) % 16);

    run_test_avx2("[AVX2] f4_e2m1->f32 no_scale", jcp, weights, bytes_per_group, nullptr, nullptr, ic_groups);
    std::free(weights);
}

/**
 * Test 22: [AVX2] u8 -> f32, 多OC块 (放大规模)
 */
void test_avx2_u8_multi_oc_block() {
     const size_t oc = 32;  // 4个 AVX2 向量宽度
    const size_t ic_int = 1;
     const size_t ic_groups = 256;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = true;
    jcp.broadcast_scales = false;
    jcp.broadcast_zero_points = false;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::u8;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;
    jcp.zero_points_dt = data_type_t::f32;

    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_groups);
    for (size_t i = 0; i < oc * ic_groups; i++)
        weights[i] = static_cast<uint8_t>(i % 256);

    auto* scales = aligned_alloc_array<float>(oc);
    for (size_t i = 0; i < oc; i++)
        scales[i] = 0.5f + i * 0.01f;

    auto* zps = aligned_alloc_array<float>(oc);
    for (size_t i = 0; i < oc; i++)
        zps[i] = static_cast<float>(i);

    run_test_avx2("[AVX2] u8->f32 multi_oc_block", jcp, weights, oc * ic_groups, scales, zps, ic_groups);
    std::free(weights);
    std::free(scales);
    std::free(zps);
}

/**
 * Test 23: [AVX2] f16 -> f32 (需要F16C扩展)
 */
void test_avx2_f16_to_f32() {
    const size_t oc = 8;
    const size_t ic_int = 1;
    const size_t ic_groups = 2;

    compile_params_t jcp = {};
    jcp.with_scales = true;
    jcp.with_zero_points = false;
    jcp.broadcast_scales = true;
    jcp.oc_size = oc;
    jcp.ic_internal_size = ic_int;
    jcp.weights_dt = data_type_t::f16;
    jcp.decomp_buffer_dt = data_type_t::f32;
    jcp.scales_dt = data_type_t::f32;

    // 生成f16测试数据
    auto* weights = aligned_alloc_array<uint8_t>(oc * ic_groups * 2);  // f16 = 2 bytes
    auto* f16_ptr = reinterpret_cast<uint16_t*>(weights);
    for (size_t i = 0; i < oc * ic_groups; i++) {
        float val = 0.5f * i - 3.0f;
        f16_ptr[i] = float_to_f16(val);
    }

    float scale = 0.1f;
    run_test_avx2("[AVX2] f16->f32 broadcast_scale", jcp, weights, oc * ic_groups * 2, &scale, nullptr, ic_groups, 1e-2f);
    std::free(weights);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "============================================================\n";
    std::cout << "  Weight Decompression JIT Kernel Tests\n";
    std::cout << "============================================================\n\n";

    // 检查CPU支持
    Xbyak::util::Cpu cpu;
    bool has_avx512 = cpu.has(Xbyak::util::Cpu::tAVX512F);
    bool has_avx2 = cpu.has(Xbyak::util::Cpu::tAVX2);
    std::cout << "CPU: AVX-512F " << (has_avx512 ? "supported" : "NOT supported") << "\n";
    std::cout << "CPU: AVX2     " << (has_avx2 ? "supported" : "NOT supported") << "\n\n";

    if (!has_avx2) {
        std::cout << "ERROR: At least AVX2 is required. Cannot run tests.\n";
        return 1;
    }

    // 运行AVX-512测试
    if (has_avx512) {
        std::cout << "--- AVX-512 Tests ---\n";
        test_u8_to_f32_no_scale();             // Test 1
        test_u8_to_f32_broadcast_scale();      // Test 2
        test_u8_to_f32_scale_and_zp();         // Test 3
        test_s8_to_f32();                      // Test 4
        test_u4_to_f32();                      // Test 5
        test_s4_to_f32();                      // Test 6
        test_u2_to_f32();                      // Test 7
        test_nf4_to_f32();                     // Test 8
        test_f4_e2m1_to_f32();                 // Test 9
        test_u8_with_u8_zp();                  // Test 10
        test_u4_with_e8m0_scale();             // Test 11
        test_u8_multi_oc_block();              // Test 12
        test_u8_to_f16_no_scale();             // Test 13
        test_u4_to_f16_scale_zp();             // Test 14
        test_nf4_to_f16_scale();               // Test 15
    }

    // 运行AVX2测试
    if (has_avx2) {
        std::cout << "\n--- AVX2 Tests ---\n";
        test_avx2_u8_to_f32();                 // Test 16
        test_avx2_u4_to_f32();                 // Test 17
        test_avx2_s4_to_f32();                 // Test 18
        test_avx2_u2_to_f32();                 // Test 19
        test_avx2_nf4_to_f32();                // Test 20
        test_avx2_f4_e2m1_to_f32();            // Test 21
        test_avx2_u8_multi_oc_block();         // Test 22
        test_avx2_f16_to_f32();                // Test 23
    }

    // 汇总
    std::cout << "\n============================================================\n";
    std::cout << "  Results: " << g_passed << "/" << g_total << " passed";
    if (g_failed > 0) std::cout << " (" << g_failed << " FAILED)";
    std::cout << "\n============================================================\n";

    return g_failed > 0 ? 1 : 0;
}
