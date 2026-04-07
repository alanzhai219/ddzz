#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>
#include <algorithm>

// 引入 Xbyak
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

// ==========================================
// 1. 定义数据结构 (脱离 OneDNN)
// ==========================================

struct src_quantization_compile_params_t {
    size_t ic_quant_block;
    // 简化：这里我们硬编码只支持 float -> int8，不再需要 data_type_t 枚举
};

struct src_quantization_runtime_params_t {
    const void *src_ptr;        // float*
    const void *qsrc_ptr;       // int8_t*
    const void *src_scales_ptr; // float*
    size_t ic_size;
};

// 定义支持的指令集架构枚举
enum class Arch {
    AVX2,
    AVX512
};

// ==========================================
// 2. JIT Kernel 实现 (纯 Xbyak)
// ==========================================

template <Arch arch>
class JitQuantizationKernel : public Xbyak::CodeGenerator {
public:
    JitQuantizationKernel(const src_quantization_compile_params_t& jcp)
        : jcp_(jcp) {
        
        // 确定向量长度 (bytes)
        vec_len_ = (arch == Arch::AVX512) ? 64 : 32;
        // 确定 float 个数
        vec_size_ = vec_len_ / sizeof(float);

        generate();
    }

    // 函数指针类型转换，方便调用
    void operator()(const src_quantization_runtime_params_t* params) {
        auto func = getCode<void (*)(const src_quantization_runtime_params_t*)>();
        func(params);
    }

private:
    src_quantization_compile_params_t jcp_;
    size_t vec_len_;
    size_t vec_size_;

    // ------------------------------------------------------
    // 寄存器映射
    // ------------------------------------------------------
    // 根据 ABI，第一个参数(params结构体指针)在 Linux 下是 rdi，Windows 下是 rcx
#ifdef _WIN32
    const Xbyak::Reg64 param1 = rcx;
#else
    const Xbyak::Reg64 param1 = rdi;
#endif

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_qsrc = r9;
    Xbyak::Reg64 reg_src_scales = r10;
    Xbyak::Reg64 reg_ic_size = r11;
    Xbyak::Reg64 reg_tmp = r12; // r12 是非易失寄存器，需要保存/恢复

    // 辅助函数：根据 Arch 返回对应的向量寄存器类型 (Ymm 或 Zmm)
    // 这里的 Vmm 是一个 helper，用于在 generate 中统一写法
    Xbyak::Operand vmm(int idx) {
        if (arch == Arch::AVX512) {
            return Xbyak::Zmm(idx);
        } else if (arch == Arch::AVX2) {
            return Xbyak::Ymm(idx);
        } else {
            assert("unsupported isa");
        }
    }

    // 具体寄存器别名
    Xbyak::Operand vmm_src()           { return vmm(0); }
    Xbyak::Operand vmm_max()           { return vmm(1); }
    Xbyak::Operand vmm_sign_bit_mask() { return vmm(2); }
    Xbyak::Operand vmm_aux()           { return vmm(3); }
    Xbyak::Operand vmm_int8_max()      { return vmm(4); }
    Xbyak::Operand vmm_qscale()        { return vmm(5); }
    Xbyak::Operand vmm_one()           { return vmm(6); }

    // 常量数据 (为了简单，直接定义为 static 数组，JIT 中通过指针访问)
    static constexpr float negative_zero[16] = { -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f, -0.f };
    static constexpr float positive_one[16]  = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
    static constexpr float int8_max_arr[16]  = { 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f, 127.f };

    // ------------------------------------------------------
    // 代码生成逻辑
    // ------------------------------------------------------
    void generate() {
        // --- Preamble (ABI 处理) ---
        // 保存 Callee-saved 寄存器 r12
        push(r12);

        // 加载参数
        // offsetof 宏用于获取结构体成员偏移
        mov(reg_src, ptr[param1 + offsetof(src_quantization_runtime_params_t, src_ptr)]);
        mov(reg_qsrc, ptr[param1 + offsetof(src_quantization_runtime_params_t, qsrc_ptr)]);
        mov(reg_src_scales, ptr[param1 + offsetof(src_quantization_runtime_params_t, src_scales_ptr)]);
        mov(reg_ic_size, ptr[param1 + offsetof(src_quantization_runtime_params_t, ic_size)]);

        Xbyak::Label ic_loop_label;
        Xbyak::Label ic_end_label;

        size_t src_dt_size = sizeof(float);
        size_t qsrc_dt_size = sizeof(int8_t);
        size_t src_scales_dt_size = sizeof(float);

        // --- 加载常量 ---
        // 将指针地址放入 reg_tmp，然后广播加载
        mov(reg_tmp, (size_t)negative_zero);
        // vmovups(vmm_sign_bit_mask(), ptr[reg_tmp]);
        if (arch == Arch::AVX512) {
            vmovups(Xbyak::Zmm(2), ptr[reg_tmp]); // vmm_sign_bit_mask
        } else {
            vmovups(Xbyak::Ymm(2), ptr[reg_tmp]);
        }

        mov(reg_tmp, (size_t)positive_one);
        if (arch == Arch::AVX512) {
            vmovups(Xbyak::Zmm(6), ptr[reg_tmp]); // vmm_one
        } else {
            vmovups(Xbyak::Ymm(6), ptr[reg_tmp]);
        }

        mov(reg_tmp, (size_t)int8_max_arr);
        if (arch == Arch::AVX512) {
            vmovups(Xbyak::Zmm(4), ptr[reg_tmp]); // vmm_int8_max
        } else {
            vmovups(Xbyak::Ymm(4), ptr[reg_tmp]);
        }

        // --- 主循环 ---
        L(ic_loop_label);
        {
            cmp(reg_ic_size, jcp_.ic_quant_block);
            jl(ic_end_label, T_NEAR);

            assert(!(jcp_.ic_quant_block % vec_size_));
            int ic_blocks = jcp_.ic_quant_block / vec_size_;

            // 清零 vmm_max
            if (arch == Arch::AVX512) {
                vpxord(Xbyak::Zmm(1), Xbyak::Zmm(1), Xbyak::Zmm(1));
            } else {
                vpxor(Xbyak::Ymm(1), Xbyak::Ymm(1), Xbyak::Ymm(1));
            }

            // 1. 寻找 Block 内的最大绝对值
            for (int icb = 0; icb < ic_blocks; icb++) {
                // Load src
                if (arch == Arch::AVX512) {
                    vmovups(Xbyak::Zmm(0), ptr[reg_src + icb * vec_size_ * src_dt_size]);
                } else {
                    vmovups(Xbyak::Ymm(0), ptr[reg_src + icb * vec_size_ * src_dt_size]);
                }

                // abs(src) = src & ~sign_mask (using vandnps)
                if (arch == Arch::AVX512) {
                    vandnps(Xbyak::Zmm(0), Xbyak::Zmm(2), Xbyak::Zmm(0));
                } else {
                    vandnps(Xbyak::Ymm(0), Xbyak::Ymm(2), Xbyak::Ymm(0));
                }

                // max reduction
                if (arch == Arch::AVX512) {
                    vmaxps(Xbyak::Zmm(1), Xbyak::Zmm(1), Xbyak::Zmm(0));
                } else {
                    vmaxps(Xbyak::Ymm(1), Xbyak::Ymm(1), Xbyak::Ymm(0));
                }
            }

            // 2. 寄存器内规约 (Horizontal Reduction)
            if (arch == Arch::AVX512) {
                Xbyak::Zmm max_zmm = Xbyak::Zmm(1); // vmm_max
                Xbyak::Zmm aux_zmm = Xbyak::Zmm(3); // vmm_aux
                vshuff32x4(aux_zmm, max_zmm, max_zmm, 0x4E);
                vmaxps(max_zmm, max_zmm, aux_zmm);
                vshuff32x4(aux_zmm, max_zmm, max_zmm, 0xB1);
                vmaxps(max_zmm, max_zmm, aux_zmm);
            } else if (arch == Arch::AVX2) { // AVX2
                Xbyak::Ymm max_ymm = Xbyak::Ymm(1);
                Xbyak::Ymm aux_ymm = Xbyak::Ymm(3);
                vperm2i128(aux_ymm, max_ymm, max_ymm, 0x01);
                vmaxps(max_ymm, max_ymm, aux_ymm);
            } else {
                assert(!"unsupported isa");
            }

            // Common lane reduction (128-bit)
            // vmm_max 现在已经是 Xmm(1) 的别名了
            Xbyak::Xmm max_xmm = Xbyak::Xmm(1);
            Xbyak::Xmm aux_xmm = Xbyak::Xmm(3);
            
            vshufps(aux_xmm, max_xmm, max_xmm, 0x4E);
            vmaxps(max_xmm, max_xmm, aux_xmm);
            vshufps(aux_xmm, max_xmm, max_xmm, 0xB1);
            vmaxps(max_xmm, max_xmm, aux_xmm);

            // 3. 计算 Scale
            // dscale = max_val
            // dscale = dscale / 127.0f
            // qscale = 1.0f / dscale
            
            // vbroadcastss vmm_dscale(1), xmm_dscale(1)
            if (arch == Arch::AVX512) {
                vbroadcastss(Xbyak::Zmm(1), Xbyak::Xmm(1));
            } else {
                vbroadcastss(Xbyak::Ymm(1), Xbyak::Xmm(1));
            }

            // div by 127
            if (arch == Arch::AVX512) {
                vdivps(Xbyak::Zmm(1), Xbyak::Zmm(1), Xbyak::Zmm(4)); // vmm_int8_max
            } else {
                vdivps(Xbyak::Ymm(1), Xbyak::Ymm(1), Xbyak::Ymm(4));
            }

            // qscale = 1.0 / dscale. result in vmm_qscale(5)
            if (arch == Arch::AVX512) {
                vdivps(Xbyak::Zmm(5), Xbyak::Zmm(6), Xbyak::Zmm(1)); // one / dscale
            } else {
                vdivps(Xbyak::Ymm(5), Xbyak::Ymm(6), Xbyak::Ymm(1));
            }

            // Store dscale to src_scales_ptr (only 1 float)
            vmovss(ptr[reg_src_scales], Xbyak::Xmm(1));

            // 4. 量化并存储
            for (int icb = 0; icb < ic_blocks; icb++) {
                // Load src
                if (arch == Arch::AVX512) {
                    vmovups(Xbyak::Zmm(0), ptr[reg_src + icb * vec_size_ * src_dt_size]);
                } else {
                    vmovups(Xbyak::Ymm(0), ptr[reg_src + icb * vec_size_ * src_dt_size]);
                }

                // mul qscale
                if (arch == Arch::AVX512) {
                    vmulps(Xbyak::Zmm(0), Xbyak::Zmm(0), Xbyak::Zmm(5));
                } else {
                    vmulps(Xbyak::Ymm(0), Xbyak::Ymm(0), Xbyak::Ymm(5));
                }

                // cvt to int32 (vcvtps2dq)
                if (arch == Arch::AVX512) {
                    vcvtps2dq(Xbyak::Zmm(0), Xbyak::Zmm(0));
                } else {
                    vcvtps2dq(Xbyak::Ymm(0), Xbyak::Ymm(0));
                }

                // Pack and Store
                if (arch == Arch::AVX512) {
                    // AVX512: vpmovsdb (down convert dword to byte with saturation)
                    vpmovsdb(ptr[reg_qsrc + icb * vec_size_ * qsrc_dt_size], Xbyak::Zmm(0));
                } else {
                    // AVX2 Packing sequence (Ported from original snippet)
                    // vpackssdw: 32bit -> 16bit
                    vpackssdw(Xbyak::Ymm(0), Xbyak::Ymm(0), Xbyak::Ymm(0));
                    
                    // vpermq: reorder 64-bit lanes. 0x08 = 00 00 10 00 -> q0, q2, q0, q0? 
                    // 注意：原代码此处是 0x08，严格保持原逻辑。
                    vpermq(Xbyak::Ymm(0), Xbyak::Ymm(0), 0x08); 
                    
                    // vpacksswb: 16bit -> 8bit
                    vpacksswb(Xbyak::Ymm(0), Xbyak::Ymm(0), Xbyak::Ymm(0));
                    
                    // Store lower 128 bits (since data was packed into lower half)
                    // 实际上 Ymm pack 后只有低 128 位是有效数据（如果输入填满了Ymm）
                    // 32个float -> 32个int8 (32 bytes) = 256 bits?
                    // 等等，AVX2 路径 vec_size=8 (256/32). 
                    // 一个 YMM 存 8 个 float. 量化后变成 8 个 byte.
                    // vmovq 存 64 位 (8 bytes).
                    vmovq(ptr[reg_qsrc + icb * vec_size_ * qsrc_dt_size], Xbyak::Xmm(0));
                }
            }

            // 指针步进
            sub(reg_ic_size, jcp_.ic_quant_block);
            add(reg_src, src_dt_size * jcp_.ic_quant_block);
            add(reg_qsrc, qsrc_dt_size * jcp_.ic_quant_block);
            add(reg_src_scales, src_scales_dt_size);

            jmp(ic_loop_label, T_NEAR);
        }
        L(ic_end_label);

        // --- Postamble ---
        pop(r12);
        ret();
    }
};

// 静态成员定义
template<Arch arch> constexpr float JitQuantizationKernel<arch>::negative_zero[16];
template<Arch arch> constexpr float JitQuantizationKernel<arch>::positive_one[16];
template<Arch arch> constexpr float JitQuantizationKernel<arch>::int8_max_arr[16];

void ref_src_quantization(const src_quantization_runtime_params_t* args, 
                          const src_quantization_compile_params_t& jcp) {
    
    const float* src = static_cast<const float*>(args->src_ptr);
    int8_t* qsrc = static_cast<int8_t*>(const_cast<void*>(args->qsrc_ptr));
    float* scales = static_cast<float*>(const_cast<void*>(args->src_scales_ptr));
    
    size_t ic_size = args->ic_size;
    size_t block_size = jcp.ic_quant_block;

    // 遍历每一个 Block
    for (size_t i = 0; i < ic_size; i += block_size) {
        // 处理边界情况（虽然 JIT 代码假设 size 是 block 的倍数，但 C++ 一般写得健壮些）
        size_t current_block_len = std::min(block_size, ic_size - i);

        // -----------------------------------------------------------
        // 1. 寻找 Block 内绝对值的最大值 (JIT: vmm_max + vandnps)
        // -----------------------------------------------------------
        float max_abs_val = 0.0f;
        for (size_t j = 0; j < current_block_len; ++j) {
            float val = std::abs(src[i + j]);
            if (val > max_abs_val) {
                max_abs_val = val;
            }
        }

        // -----------------------------------------------------------
        // 2. 计算 Scale (JIT: scale = max / 127.0f)
        // -----------------------------------------------------------
        // JIT中使用 127.0f 作为分母
        float scale = max_abs_val / 127.0f;
        
        // 存储 scale 到输出数组
        // 注意：src_scales_ptr 在 JIT 中是随着 block 递增的，每个 block 存一个 float
        scales[i / block_size] = scale;

        // -----------------------------------------------------------
        // 3. 计算用于乘法的反向 Scale (JIT: qscale = 1.0f / scale)
        // -----------------------------------------------------------
        float inverse_scale;
        // JIT 代码中有一行注释 // todo: check zero case
        // 在 C++ 中如果不处理，全0数据会导致除以0 (Inf)。
        // 为了安全，这里做一个简单的零检查：
        if (std::abs(scale) < 1e-9f) {
            inverse_scale = 0.0f;
        } else {
            inverse_scale = 1.0f / scale;
        }

        // -----------------------------------------------------------
        // 4. 量化循环 (JIT: vmulps -> vcvtps2dq -> saturate)
        // -----------------------------------------------------------
        for (size_t j = 0; j < current_block_len; ++j) {
            float raw_val = src[i + j];
            
            // 乘法量化
            float quantized_f = raw_val * inverse_scale;

            // 取整
            // JIT 指令 vcvtps2dq 默认使用 "Round to nearest even" (MXCSR缺省设置)
            // std::nearbyint 是 C++ 中最接近此行为的函数
            // 也可以简单的用 std::round (四舍五入)
            int32_t quantized_i = static_cast<int32_t>(std::nearbyint(quantized_f));

            // 饱和截断 (Saturate)
            // JIT AVX512 使用 vpmovsdb, AVX2 使用 vpacksswb
            // 它们都会将超出 int8 范围的数值截断到 [-128, 127]
            if (quantized_i > 127) {
                quantized_i = 127;
            } else if (quantized_i < -128) {
                quantized_i = -128;
            }

            // 写入结果
            qsrc[i + j] = static_cast<int8_t>(quantized_i);
        }
    }
}


// ==========================================
// 3. 测试 Main 函数
// ==========================================
int main() {
    // 设置 Xbyak 只读执行权限
    // 注意：默认 Xbyak 构造函数已处理，但某些环境可能需要手动 protect
    
    // 初始化参数
    // 假设 Block Size = 32 (需要是 vec_size 的倍数)
    // AVX2 vec_size = 8, AVX512 vec_size = 16
    size_t ic_quant_block = 32; // 每ic_quant_block share一个scale
    size_t data_size = 64; // 总数据量

    src_quantization_compile_params_t jcp = { ic_quant_block };

    // 分配内存
    std::vector<float> src(data_size);
    std::vector<int8_t> qsrc(data_size);
    std::vector<float> scales(data_size / ic_quant_block);

    // 初始化输入数据
    for (size_t i = 0; i < data_size; ++i) {
        src[i] = (i % 2 == 0) ? (float)i : -(float)i; // 0, -1, 2, -3...
        // 使得 block 内 max 逐渐变大
    }

    src_quantization_runtime_params_t params;
    params.src_ptr = src.data();
    params.qsrc_ptr = qsrc.data();
    params.src_scales_ptr = scales.data();
    params.ic_size = data_size;

    // 实例化 Kernel (选择 AVX2 或 AVX512)
    // 检测当前 CPU 支持
    Xbyak::util::Cpu cpu;
    if (cpu.has(Xbyak::util::Cpu::tAVX512F)) {
        printf("Generating AVX512 Kernel...\n");
        JitQuantizationKernel<Arch::AVX512> kernel(jcp);
        kernel(&params);
    } else if (cpu.has(Xbyak::util::Cpu::tAVX2)) {
        printf("Generating AVX2 Kernel...\n");
        JitQuantizationKernel<Arch::AVX2> kernel(jcp);
        kernel(&params);
    } else {
        printf("Error: CPU does not support AVX2 or AVX512\n");
        return -1;
    }

    // 验证结果
    printf("Verification:\n");
    for (size_t b = 0; b < data_size / ic_quant_block; ++b) {
        float max_val = 0.0f;
        for (size_t i = 0; i < ic_quant_block; ++i) {
            max_val = std::max(max_val, std::abs(src[b * ic_quant_block + i]));
        }
        float scale = max_val / 127.0f;
        
        printf("Block %zu: Max=%.2f, JIT_Scale=%.6f, Ref_Scale=%.6f\n", 
               b, max_val, scales[b], scale);

        // 验证部分数据
        for(size_t i=0; i<4; ++i) {
            size_t idx = b * ic_quant_block + i;
            float qscale = 1.0f / scale;
            int8_t expected = (int8_t)(src[idx] * qscale);
            printf("  src[%.2f] -> qsrc[%d] (Expected: %d)\n", 
                   src[idx], qsrc[idx], expected);
        }
    }

    return 0;
}
