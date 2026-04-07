/**
 * @file weight_decomp_types.hpp
 * @brief 独立于oneDNN的权重解压缩类型定义
 *
 * 本文件定义了权重解压缩内核所需的数据类型枚举、编译时参数和运行时参数。
 * 这些类型替代了oneDNN中的 data_type_t, weights_decompression_compile_params_t,
 * weights_decompression_runtime_params_t 等。
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cassert>

namespace weight_decomp {

// ============================================================================
// 数据类型枚举 —— 对应oneDNN中的 data_type_t
// ============================================================================
enum class data_type_t {
    f32,      // 32-bit float
    bf16,     // bfloat16 (16-bit)
    f16,      // float16  (16-bit)
    u8,       // unsigned 8-bit integer
    s8,       // signed 8-bit integer
    u4,       // unsigned 4-bit integer (每byte存2个值)
    s4,       // signed 4-bit integer   (每byte存2个值)
    u2,       // unsigned 2-bit integer (每byte存4个值)
    nf4,      // NormalFloat 4-bit (QLoRA格式, 16个查找表值)
    f4_e2m1,  // FP4 E2M1 (4-bit浮点: 2位指数 + 1位尾数)
    e8m0,     // E8M0 (8位指数, 无尾数的scale格式)
};

// ============================================================================
// 每种数据类型占用的位数和字节数
// ============================================================================
inline size_t data_type_size(data_type_t dt) {
    switch (dt) {
        case data_type_t::f32:     return 4;
        case data_type_t::bf16:    return 2;
        case data_type_t::f16:     return 2;
        case data_type_t::u8:      return 1;
        case data_type_t::s8:      return 1;
        case data_type_t::u4:      return 1;  // 2个值共享1 byte
        case data_type_t::s4:      return 1;
        case data_type_t::u2:      return 1;  // 4个值共享1 byte
        case data_type_t::nf4:     return 1;
        case data_type_t::f4_e2m1: return 1;
        case data_type_t::e8m0:    return 1;
        default: assert(!"unknown data type"); return 0;
    }
}

/// 获取typesize_scale, 表示一个byte中打包了多少个元素的倒数关系
/// u2: 4个值/byte -> scale=4; u4/s4/nf4/f4_e2m1: 2个值/byte -> scale=2; 其余: 1
inline size_t type_pack_scale(data_type_t dt) {
    switch (dt) {
        case data_type_t::u2:      return 4;
        case data_type_t::u4:
        case data_type_t::s4:
        case data_type_t::nf4:
        case data_type_t::f4_e2m1: return 2;
        default:                   return 1;
    }
}

// ============================================================================
// 编译时参数 —— 对应oneDNN中的 weights_decompression_compile_params_t
// ============================================================================
struct compile_params_t {
    bool with_scales;            // 是否使用scale进行反量化
    bool with_zero_points;       // 是否使用zero_point进行反量化
    bool broadcast_scales;       // scale是否为标量广播 (所有OC共享一个值)
    bool broadcast_zero_points;  // zero_point是否为标量广播
    size_t oc_size;              // 输出通道数(一次处理的OC维度大小)
    size_t ic_internal_size;     // 内部IC分组大小 (sub-byte打包时 >1, e.g. u4->2, u2->4)
    data_type_t weights_dt;      // 权重(压缩)数据类型
    data_type_t decomp_buffer_dt;// 解压缩输出数据类型 (f32 or bf16)
    data_type_t scales_dt;       // scale的数据类型
    data_type_t zero_points_dt;  // zero_point的数据类型
};

// ============================================================================
// 运行时参数 —— 通过寄存器传递给JIT kernel
// ============================================================================
struct runtime_params_t {
    const void *weights_ptr;       // 压缩权重指针
    const void *decomp_buffer_ptr; // 解压缩输出缓冲区指针
    const void *scales_ptr;        // scale数组指针
    const void *zero_points_ptr;   // zero_point数组指针
    size_t ic_size;                // 当前需要处理的IC分组数
};

// ============================================================================
// 辅助函数
// ============================================================================
inline size_t div_up(size_t a, size_t b) {
    return (a + b - 1) / b;
}

} // namespace weight_decomp
