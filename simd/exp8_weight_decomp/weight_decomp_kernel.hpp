/**
 * @file weight_decomp_kernel.hpp
 * @brief 基于Xbyak的权重解压缩JIT内核 (独立于oneDNN, 支持AVX-512和AVX2)
 *
 * 本内核实现了oneDNN中 jit_brgemm_weights_decompression_kernel_t 的核心逻辑,
 * 将压缩格式权重(u8/s8/u4/s4/u2/nf4/f4_e2m1/f16/bf16)解压缩为f32/f16,
 * 并可选地应用 scale 和 zero_point 进行反量化:
 *
 *    output = (weight - zero_point) * scale
 *
 * ============================================================================
 * 设计模式: ISA 模板参数 (参考 oneDNN)
 * ============================================================================
 *
 * 使用模板参数 isa_t 在编译时选择 ISA:
 *   - isa_t::avx512: 使用 ZMM 寄存器 (512-bit, 16 float, 32 regs)
 *   - isa_t::avx2:   使用 YMM 寄存器 (256-bit, 8 float, 16 regs)
 *
 * ISA 差异通过 if constexpr 在编译时消除, 最终生成无分支的 JIT 代码。
 * 关键差异点:
 *   - nf4:     AVX-512 单 ZMM 放 16 项 LUT, 一条 vpermd;
 *              AVX2 拆为 low/high 两个 YMM + vblendvps
 *   - f4_e2m1: AVX-512 直接 16 项 LUT + vpermd;
 *              AVX2 符号分离 + 8 项 abs LUT + vorps
 *   - 输出:    AVX-512 支持 f32/f16; AVX2 仅支持 f32
 *   - 地址:    AVX-512 用 zword; AVX2 用 yword/qword
 *
 * 寄存器分配:
 *   vmm[0..3]:                权重 (unroll_factor=4)
 *   vmm[4..4+uf-1]:           scale
 *   vmm[2*uf..2*uf+uf-1]:    zero_point
 *   vmm[n_vregs-1..n_vregs-4]: 临时/查找表
 *
 * ============================================================================
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <functional>
#include <stdexcept>

#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"
#include "weight_decomp_types.hpp"

namespace weight_decomp {

// ISA 枚举 (参考 oneDNN 的 cpu_isa_t)
enum class isa_t {
    avx2,
    avx512
};

/**
 * @class WeightDecompKernel
 * @brief 模板化 JIT 权重解压缩内核
 *
 * @tparam isa  目标 ISA (avx2 或 avx512)
 *
 * 使用方式:
 *   compile_params_t params = {...};
 *   WeightDecompKernel<isa_t::avx512> kernel(params);
 *   runtime_params_t rt = {weights_ptr, buffer_ptr, scales_ptr, zp_ptr, ic_groups};
 *   kernel.execute(&rt);
 */
template <isa_t isa>
class WeightDecompKernel : public Xbyak::CodeGenerator {
public:
    WeightDecompKernel(const compile_params_t& jcp) : Xbyak::CodeGenerator(4096 * 4, Xbyak::AutoGrow), jcp_(jcp) {
        // 检查 CPU 是否支持目标 ISA
        Xbyak::util::Cpu cpu;
        if constexpr (isa == isa_t::avx512) {
            if (!cpu.has(Xbyak::util::Cpu::tAVX512F))
                throw std::runtime_error("AVX-512F is required for this kernel");
        } else {
            if (!cpu.has(Xbyak::util::Cpu::tAVX2))
                throw std::runtime_error("AVX2 is required for this kernel");
        }

        // AVX2 限制
        if constexpr (isa == isa_t::avx2) {
            if (jcp_.decomp_buffer_dt != data_type_t::f32)
                throw std::runtime_error("AVX2 kernel only supports f32 output");
            if (jcp_.weights_dt == data_type_t::bf16)
                throw std::runtime_error("bf16 weight input requires AVX-512");
            if (jcp_.weights_dt == data_type_t::f16 && !cpu.has(Xbyak::util::Cpu::tF16C))
                throw std::runtime_error("F16C is required for f16 weight input");
        }

        assert(div_up(jcp_.oc_size, get_vec_size()) <= unroll_factor && "oc_size too large for register allocation");

        generate();
        ready();
    }

    void execute(const runtime_params_t* args) const {
        auto fn = getCode<void (*)(const runtime_params_t*)>();
        fn(args);
    }

    size_t get_vec_size() const { return vec_size; }

private:
    // ========================================================================
    // 成员变量
    // ========================================================================
    compile_params_t jcp_;

    // ========================================================================
    // ISA 相关类型与常量 (编译时确定)
    // ========================================================================

    // Vmm: AVX-512 → Zmm (512-bit), AVX2 → Ymm (256-bit)
    using Vmm = std::conditional_t<isa == isa_t::avx512, Xbyak::Zmm, Xbyak::Ymm>;
    using Ymm = Xbyak::Ymm;
    using Xmm = Xbyak::Xmm;

    // 向量宽度 (float 个数)
    static constexpr size_t vec_size = (isa == isa_t::avx512) ? 16 : 8;

    // 向量寄存器总数: AVX-512 = 32, AVX2 = 16
    static constexpr int n_vregs = (isa == isa_t::avx512) ? 32 : 16;

    // ========================================================================
    // 寄存器分配
    // ========================================================================
    static constexpr int unroll_factor = 4;

    Vmm vmm_weights(int ocb)     const { return Vmm(ocb); } // vmm0-vmm3 用于权重加载和处理
    Vmm vmm_scales(int ocb)      const { return Vmm(unroll_factor + ocb); } // vmm4-vmm7 用于 scale
    Vmm vmm_zero_points(int ocb) const { return Vmm(2 * unroll_factor + ocb); } // vmm8-vmm11 用于 zero_point
    Vmm vmm_tmp(int idx)         const { return Vmm(n_vregs - idx - 1); } // vmm28-vmm31 用于临时计算

    // 查找表寄存器
    // - AVX-512 nf4/f4_e2m1:  vmm_lookup() = 完整 16 项 LUT
    // - AVX2 nf4:             vmm_lookup_low() + vmm_lookup_high() = 2×8 项
    // - AVX2 f4_e2m1:         vmm_lookup() = 8 项 abs LUT
    Vmm vmm_lookup()      const { return vmm_tmp(0); }  // n_vregs-1
    Vmm vmm_lookup_low()  const { return vmm_tmp(0); }  // 同 vmm_lookup (nf4低8项)
    Vmm vmm_lookup_high() const { return vmm_tmp(1); }  // n_vregs-2 (nf4高8项)

    // 辅助掩码寄存器
    Vmm vmm_mask()  const { return vmm_tmp(1); }  // n_vregs-2, f4_e2m1 sign mask
    Vmm vmm_mask8() const { return vmm_tmp(2); }  // n_vregs-3, nf4 broadcast(8)
    Vmm vmm_mask7() const { return vmm_tmp(3); }  // n_vregs-4, nf4 broadcast(7)

    // bf16交织辅助 (仅AVX-512使用)
    Vmm vmm_aux0() const { return Vmm(14); }
    Vmm vmm_aux1() const { return Vmm(15); }

    // --- GPR ---
    const Xbyak::Reg64 reg_param        = Xbyak::util::rdi;
    const Xbyak::Reg64 reg_weights      = Xbyak::util::r8;
    const Xbyak::Reg64 reg_decomp_buf   = Xbyak::util::r9;
    const Xbyak::Reg64 reg_scales       = Xbyak::util::r10;
    const Xbyak::Reg64 reg_zero_points  = Xbyak::util::r11;
    const Xbyak::Reg64 reg_ic_size      = Xbyak::util::r12;
    const Xbyak::Reg64 reg_tmp          = Xbyak::util::r13;

    // ========================================================================
    // uni_* 指令封装: 自动适配不同 ISA 的内存操作数大小
    // ========================================================================
    // AVX-512 zword = 64字节, AVX2 yword = 32字节
    // 用于 vmovups 等需要指定内存大小的场景

    auto make_vmm_addr(const Xbyak::Reg64& base, size_t offset = 0) const {
        if constexpr (isa == isa_t::avx512) {
            return zword[base + offset];
        } else {
            return yword[base + offset];
        }
    }

    // 权重加载地址: byte/sub-byte类型需要 xword(128-bit) 或 qword(64-bit)
    // AVX-512: vpmovzxbd zmm, xword (从16字节读16值)
    // AVX2:    vpmovzxbd ymm, qword (从8字节读8值)
    auto make_byte_load_addr(const Xbyak::Reg64& base, size_t offset = 0) const {
        if constexpr (isa == isa_t::avx512) {
            return xword[base + offset];
        } else {
            return qword[base + offset];
        }
    }

    // f16 加载地址: AVX-512 从 yword (32字节=16×f16), AVX2 从 xword (16字节=8×f16)
    auto make_f16_load_addr(const Xbyak::Reg64& base, size_t offset = 0) const {
        if constexpr (isa == isa_t::avx512) {
            return yword[base + offset];
        } else {
            return xword[base + offset];
        }
    }

    // ========================================================================
    // JIT 代码生成入口
    // ========================================================================
    void generate() {
        // Preamble: 保存 callee-saved 寄存器 (System V ABI)
        // 帧指针建立（frame pointer setup / establish frame pointer）
        push(Xbyak::util::rbp);
        mov(Xbyak::util::rbp, Xbyak::util::rsp);
        // push 5 个 callee-saved 寄存器 (rbx, r12-r15)
        push(Xbyak::util::rbx);
        push(Xbyak::util::r12);
        push(Xbyak::util::r13);
        push(Xbyak::util::r14);
        push(Xbyak::util::r15);

        // 从 runtime_params_t 加载参数到 GPR
        mov(reg_weights,    qword[reg_param + offsetof(runtime_params_t, weights_ptr)]);
        mov(reg_decomp_buf, qword[reg_param + offsetof(runtime_params_t, decomp_buffer_ptr)]);
        if (jcp_.with_scales) {
            mov(reg_scales, qword[reg_param + offsetof(runtime_params_t, scales_ptr)]);
        }
        if (jcp_.with_zero_points) {
            mov(reg_zero_points, qword[reg_param + offsetof(runtime_params_t, zero_points_ptr)]);
        }
        mov(reg_ic_size, qword[reg_param + offsetof(runtime_params_t, ic_size)]);

        // 加载查找表常量 (nf4 / f4_e2m1)
        load_lookup_tables();

        // 预加载 scale 和 zero_point 到向量寄存器
        if (jcp_.with_scales) {
            init_decomp_params([this](int ocb) { return vmm_scales(ocb); },
                               reg_scales,
                               jcp_.broadcast_scales,
                               jcp_.scales_dt);
        }
        if (jcp_.with_zero_points) {
            init_decomp_params([this](int ocb) { return vmm_zero_points(ocb); },
                               reg_zero_points,
                               jcp_.broadcast_zero_points,
                               jcp_.zero_points_dt);
        }

        // IC group 循环
        generate_ic_loop();

        // Postamble
        pop(Xbyak::util::r15);
        pop(Xbyak::util::r14);
        pop(Xbyak::util::r13);
        pop(Xbyak::util::r12);
        pop(Xbyak::util::rbx);
        // 恢复栈帧并返回 Restore stack frame and return
        // leave() 伪指令等价于:
        mov(Xbyak::util::rsp, Xbyak::util::rbp);
        pop(Xbyak::util::rbp);
        ret();
    }

    // ========================================================================
    // 查找表加载 —— ISA 差异集中在此
    // ========================================================================
    void load_lookup_tables() {
        if (jcp_.weights_dt == data_type_t::nf4) {
            static const float nf4_lookup[16] = {
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

            if constexpr (isa == isa_t::avx512) {
                // AVX-512: 完整 16 项放入单个 ZMM
                mov(reg_tmp, reinterpret_cast<size_t>(nf4_lookup));
                vmovups(vmm_lookup(), zword[reg_tmp]);
            } else {
                // AVX2: 拆为低 8 项 + 高 8 项 (vpermd 只能索引 0..7)
                mov(reg_tmp, reinterpret_cast<size_t>(nf4_lookup));
                vmovups(vmm_lookup_low(), yword[reg_tmp]);
                vmovups(vmm_lookup_high(), yword[reg_tmp + 8 * sizeof(float)]);

                // 辅助常量: broadcast(8) 和 broadcast(7)
                static const int32_t mask8_data[8] = {8, 8, 8, 8, 8, 8, 8, 8};
                static const int32_t mask7_data[8] = {7, 7, 7, 7, 7, 7, 7, 7};
                mov(reg_tmp, reinterpret_cast<size_t>(mask8_data));
                vmovups(vmm_mask8(), yword[reg_tmp]);
                mov(reg_tmp, reinterpret_cast<size_t>(mask7_data));
                vmovups(vmm_mask7(), yword[reg_tmp]);
            }

        } else if (jcp_.weights_dt == data_type_t::f4_e2m1) {
            if constexpr (isa == isa_t::avx512) {
                // AVX-512: 完整 16 项 LUT (正值 + 负值)
                static const float f4_lookup[16] = {
                    0.0f,
                    0.5f,
                    1.0f,
                    1.5f,
                    2.0f,
                    3.0f,
                    4.0f,
                    6.0f,
                    -0.0f,
                    -0.5f,
                    -1.0f,
                    -1.5f,
                    -2.0f,
                    -3.0f,
                    -4.0f,
                    -6.0f
                };
                mov(reg_tmp, reinterpret_cast<size_t>(f4_lookup));
                vmovups(vmm_lookup(), zword[reg_tmp]);
            } else {
                // AVX2: 仅 8 项绝对值 LUT, 符号单独处理
                static const float f4_lookup_abs[8] = {
                    0.0f,
                    0.5f,
                    1.0f,
                    1.5f,
                    2.0f,
                    3.0f,
                    4.0f,
                    6.0f
                };
                mov(reg_tmp, reinterpret_cast<size_t>(f4_lookup_abs));
                vmovups(vmm_lookup(), yword[reg_tmp]);

                // 符号位掩码: 0x80000000
                static const uint32_t sign_mask[8] = {
                    0x80000000, 0x80000000, 0x80000000, 0x80000000,
                    0x80000000, 0x80000000, 0x80000000, 0x80000000
                };
                mov(reg_tmp, reinterpret_cast<size_t>(sign_mask));
                vmovups(vmm_mask(), yword[reg_tmp]);
            }
        }
    }

    // ========================================================================
    // 初始化反量化参数 (scale / zero_point) 到向量寄存器
    // ========================================================================
    void init_decomp_params(std::function<Vmm(int)> vmm_fn,
                            Xbyak::Reg64 reg_ptr,
                            bool broadcast,
                            data_type_t dt) {
        // there is a limitation that oc_blocks_num must be <= unroll_factor (4)
        // otherwise we need to spill registers or use a loop to load parameters in chunks.
        size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);

        for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
            if (broadcast) {
                switch (dt) {
                    case data_type_t::f32: {
                        vbroadcastss(vmm_fn(ocb), dword[reg_ptr]);
                        break;
                    }
                    case data_type_t::u8: {
                        auto xmm = Xmm(vmm_fn(ocb).getIdx());
                        auto reg32 = Xbyak::Reg32(reg_tmp.getIdx());
                        movzx(reg32, byte[reg_ptr]);
                        if constexpr (isa == isa_t::avx512) {
                            vmovq(xmm, reg_tmp);
                        } else {
                            vmovd(xmm, reg32);
                        }
                        vcvtdq2ps(xmm, xmm);
                        vbroadcastss(vmm_fn(ocb), xmm);
                        break;
                    }
                    case data_type_t::u2: {
                        auto xmm = Xmm(vmm_fn(ocb).getIdx());
                        auto reg32 = Xbyak::Reg32(reg_tmp.getIdx());
                        movzx(reg32, byte[reg_ptr]);
                        and_(reg32, 0x3);
                        if constexpr (isa == isa_t::avx512) {
                            vmovq(xmm, reg_tmp);
                        } else {
                            vmovd(xmm, reg32);
                        }
                        vcvtdq2ps(xmm, xmm);
                        vbroadcastss(vmm_fn(ocb), xmm);
                        break;
                    }
                    case data_type_t::e8m0: {
                        auto xmm = Xmm(vmm_fn(ocb).getIdx());
                        auto reg32 = Xbyak::Reg32(reg_tmp.getIdx());
                        movzx(reg32, byte[reg_ptr]);
                        if constexpr (isa == isa_t::avx512) {
                            vmovq(xmm, reg_tmp);
                        } else {
                            vmovd(xmm, reg32);
                        }
                        vpslld(xmm, xmm, 23);
                        vbroadcastss(vmm_fn(ocb), xmm);
                        break;
                    }
                    default: assert(!"unsupported scale/zp data type for broadcast");
                }
            } else {
                size_t offset = ocb * vec_size * data_type_size(dt);
                switch (dt) {
                    case data_type_t::f32: {
                        vmovups(vmm_fn(ocb), make_vmm_addr(reg_ptr, offset));
                        break;
                    }
                    case data_type_t::u8: {
                        vpmovzxbd(vmm_fn(ocb), make_byte_load_addr(reg_ptr, offset));
                        vcvtdq2ps(vmm_fn(ocb), vmm_fn(ocb));
                        break;
                    }
                    case data_type_t::e8m0: {
                        vpmovzxbd(vmm_fn(ocb), make_byte_load_addr(reg_ptr, offset));
                        vpslld(vmm_fn(ocb), vmm_fn(ocb), 23);
                        break;
                    }
                    default: assert(!"unsupported scale/zp data type for vector load");
                }
            }
        }
    }

    // ========================================================================
    // 权重加载与解压缩
    // ========================================================================
    void load_weights(Vmm vmm_dst, const Xbyak::Address& addr, int ic) {
        switch (jcp_.weights_dt) {
            case data_type_t::u8: {
                vpmovzxbd(vmm_dst, addr);
                vcvtdq2ps(vmm_dst, vmm_dst);
                break;
            }
            case data_type_t::s8: {
                vpmovsxbd(vmm_dst, addr);
                vcvtdq2ps(vmm_dst, vmm_dst);
                break;
            }
            case data_type_t::u4: {
                vpmovzxbd(vmm_dst, addr);
                if (ic % 2 == 0) {
                    vpsrld(vmm_dst, vmm_dst, 4);
                } else {
                    vpslld(vmm_dst, vmm_dst, 28);
                    vpsrld(vmm_dst, vmm_dst, 28);
                }
                vcvtdq2ps(vmm_dst, vmm_dst);
                break;
            }
            case data_type_t::s4: {
                vpmovsxbd(vmm_dst, addr);
                if (ic % 2 == 0) {
                    vpsrad(vmm_dst, vmm_dst, 4);
                } else {
                    vpslld(vmm_dst, vmm_dst, 28);
                    vpsrad(vmm_dst, vmm_dst, 28);
                }
                vcvtdq2ps(vmm_dst, vmm_dst);
                break;
            }
            case data_type_t::u2: {
                vpmovzxbd(vmm_dst, addr);
                if (ic == 0) {
                    vpsrld(vmm_dst, vmm_dst, 6);
                } else {
                    vpslld(vmm_dst, vmm_dst, 24 + 2 * ic);
                    vpsrld(vmm_dst, vmm_dst, 30);
                }
                vcvtdq2ps(vmm_dst, vmm_dst);
                break;
            }
            case data_type_t::nf4: {
                // 先提取 4-bit index (同 u4)
                vpmovzxbd(vmm_dst, addr);
                if (ic % 2 == 0) {
                    vpsrld(vmm_dst, vmm_dst, 4);
                } else {
                    vpslld(vmm_dst, vmm_dst, 28);
                    vpsrld(vmm_dst, vmm_dst, 28);
                }

                if constexpr (isa == isa_t::avx512) {
                    // AVX-512: 一条 vpermd 完成 16 路查表
                    vpermd(vmm_dst, vmm_dst, vmm_lookup());
                } else {
                    // AVX2: 拆分查找 + vblendvps 合并
                    // 借用 vmm_weights(1), vmm_weights(2) 为临时
                    auto res = vmm_weights(1);
                    auto mask = vmm_weights(2);
                    vpcmpgtd(mask, vmm_dst, vmm_mask7());     // mask = idx > 7
                    vpermd(res, vmm_dst, vmm_lookup_low());   // 查低 8 项
                    vpsubd(vmm_dst, vmm_dst, vmm_mask8());    // idx -= 8
                    vpermd(vmm_dst, vmm_dst, vmm_lookup_high()); // 查高 8 项
                    vblendvps(vmm_dst, res, vmm_dst, mask);   // 合并
                }
                break;
            }
            case data_type_t::f4_e2m1: {
                if constexpr (isa == isa_t::avx512) {
                    // AVX-512: 无符号提取 + 16 项 LUT 直接查找
                    vpmovzxbd(vmm_dst, addr);
                    if (ic % 2 == 0) {
                        vpsrld(vmm_dst, vmm_dst, 4);
                    } else {
                        vpslld(vmm_dst, vmm_dst, 28);
                        vpsrld(vmm_dst, vmm_dst, 28);
                    }
                    vpermd(vmm_dst, vmm_dst, vmm_lookup());
                } else {
                    // AVX2: 有符号提取 → 分离符号 → 8 项 abs LUT → OR 符号
                    vpmovsxbd(vmm_dst, addr);
                    if (ic % 2 == 0) {
                        vpsrad(vmm_dst, vmm_dst, 4);
                    } else {
                        vpslld(vmm_dst, vmm_dst, 28);
                        vpsrad(vmm_dst, vmm_dst, 28);
                    }
                    auto sign = vmm_weights(1);
                    vpand(sign, vmm_dst, vmm_mask());         // sign = val & 0x80000000
                    vpermd(vmm_dst, vmm_dst, vmm_lookup());   // abs LUT 查找
                    vorps(vmm_dst, vmm_dst, sign);            // 合并符号位
                }
                break;
            }
            case data_type_t::f16: {
                // f16->f32: vcvtph2ps, AVX-512 从 ymm, AVX2 从 xmm (F16C)
                vcvtph2ps(vmm_dst, addr);
                break;
            }
            case data_type_t::bf16: {
                // bf16->f32: zero-extend + 左移 16 (仅 AVX-512)
                static_assert(isa == isa_t::avx512 || true, "bf16 handled at runtime");
                vpmovzxwd(vmm_dst, addr);
                vpslld(vmm_dst, vmm_dst, 16);
                break;
            }
            default:
                assert(!"unsupported weights data type");
        }
    }

    // ========================================================================
    // 输出存储
    // ========================================================================
    void store_weights(const Xbyak::Address& addr, Vmm vmm_store) {
        switch (jcp_.decomp_buffer_dt) {
            case data_type_t::f32: {
                vmovups(addr, vmm_store);
                break;
            }
            case data_type_t::f16: {
                // f32 → f16 (仅 AVX-512)
                Ymm ymm_store = Ymm(vmm_store.getIdx());
                vcvtps2ph(ymm_store, vmm_store, 4);
                vmovdqu16(addr, ymm_store);
                break;
            }
            case data_type_t::bf16: {
                // f32 → bf16 (仅 AVX-512)
                Ymm ymm_store = Ymm(vmm_store.getIdx());
                vcvtneps2bf16(ymm_store, vmm_store);
                vmovdqu16(addr, ymm_store);
                break;
            }
            default:
                assert(!"unsupported decomp buffer data type");
        }
    }

    // ========================================================================
    // IC 循环主体
    // ========================================================================
    void generate_ic_loop() {
        size_t oc_blocks_num = div_up(jcp_.oc_size, vec_size);
        size_t weights_dt_size = data_type_size(jcp_.weights_dt);
        size_t pack_scale = type_pack_scale(jcp_.weights_dt);
        size_t decomp_dt_size = data_type_size(jcp_.decomp_buffer_dt);

        Xbyak::Label ic_loop_label;
        Xbyak::Label ic_end_label;

        align(64);
        L(ic_loop_label);
        {
            cmp(reg_ic_size, 1);
            jl(ic_end_label, Xbyak::CodeGenerator::T_NEAR);

            if (jcp_.decomp_buffer_dt == data_type_t::bf16) {
                generate_bf16_path(oc_blocks_num, weights_dt_size, pack_scale, decomp_dt_size);
            } else {
                generate_f32_path(oc_blocks_num, weights_dt_size, pack_scale, decomp_dt_size);
            }

            dec(reg_ic_size);
            add(reg_weights, jcp_.oc_size * weights_dt_size);
            add(reg_decomp_buf, jcp_.oc_size * jcp_.ic_internal_size * decomp_dt_size);

            jmp(ic_loop_label, Xbyak::CodeGenerator::T_NEAR);
        }
        L(ic_end_label);
    }

    // ========================================================================
    // f32/f16 输出路径
    // ========================================================================
    void generate_f32_path(size_t oc_blocks_num,
                           size_t weights_dt_size,
                           size_t pack_scale,
                           size_t decomp_dt_size) {
        for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
            for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                // vec_size: 向量寄存器的字节数
                // 每次外层循环，只解压缩一个 oc block (vec_size 个输出通道)，但需要处理所有 ic_internal_size 个输入通道
                // 每次内层循环，加载一个 ic_internal 的权重块 (oc block × 1 ic)，解压缩并存储到对应位置
                size_t weights_offset = ocb * vec_size * weights_dt_size;
                auto weights_addr = make_byte_load_addr(reg_weights, weights_offset);

                load_weights(vmm_weights(0), weights_addr, ic);

                if (jcp_.with_zero_points) {
                    vsubps(vmm_weights(0), vmm_weights(0), vmm_zero_points(ocb));
                }
                if (jcp_.with_scales) {
                    vmulps(vmm_weights(0), vmm_weights(0), vmm_scales(ocb));
                }

                size_t decomp_offset = (ic * jcp_.oc_size + ocb * vec_size) * decomp_dt_size;
                store_weights(make_vmm_addr(reg_decomp_buf, decomp_offset), vmm_weights(0));
            }
        }
    }

    // ========================================================================
    // bf16 输出路径 (仅 AVX-512)
    // ========================================================================
    void generate_bf16_path(size_t oc_blocks_num,
                            size_t weights_dt_size,
                            size_t pack_scale,
                            size_t decomp_dt_size) {
        for (size_t ocb = 0; ocb < oc_blocks_num; ocb++) {
            // 加载并反量化所有 ic_internal 权重
            for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                size_t weights_offset;
                if (jcp_.weights_dt == data_type_t::u8 || jcp_.weights_dt == data_type_t::s8) {
                    weights_offset = (ic * jcp_.oc_size + ocb * vec_size) * weights_dt_size / pack_scale;
                } else {
                    weights_offset = ocb * jcp_.ic_internal_size * vec_size * weights_dt_size / pack_scale;
                }
                auto vmm_load = vmm_weights(ic);
                load_weights(vmm_load, make_byte_load_addr(reg_weights, weights_offset), ic);

                if (jcp_.with_zero_points) {
                    vsubps(vmm_load, vmm_load, vmm_zero_points(ocb));
                }
                if (jcp_.with_scales) {
                    vmulps(vmm_load, vmm_load, vmm_scales(ocb));
                }
            }

            // f32 → bf16 + 交织
            for (size_t ic = 0; ic < jcp_.ic_internal_size; ic += 2) {
                auto ymm0 = Ymm(vmm_weights(ic).getIdx());
                auto ymm1 = Ymm(vmm_weights(ic + 1).getIdx());
                auto ymm_a0 = Ymm(vmm_aux0().getIdx());
                auto ymm_a1 = Ymm(vmm_aux1().getIdx());

                vcvtneps2bf16(ymm0, vmm_weights(ic));
                vcvtneps2bf16(ymm1, vmm_weights(ic + 1));
                vpunpcklwd(ymm_a0, ymm0, ymm1);
                vpunpckhwd(ymm_a1, ymm0, ymm1);
                vperm2i128(ymm0, ymm_a0, ymm_a1, 0x20);
                vperm2i128(ymm1, ymm_a0, ymm_a1, 0x31);
            }

            // 存储交织后的 bf16
            for (size_t ic = 0; ic < jcp_.ic_internal_size; ic++) {
                auto ymm_store = Ymm(vmm_weights(ic).getIdx());
                size_t decomp_offset;
                if (jcp_.weights_dt == data_type_t::u2) {
                    decomp_offset = (((ic / 2) * div_up(jcp_.oc_size, vec_size) + ocb) * 2 + (ic % 2)) * vec_size * decomp_dt_size;
                } else {
                    decomp_offset = (ocb * jcp_.ic_internal_size + ic) * vec_size * decomp_dt_size;
                }
                vmovdqu16(yword[reg_decomp_buf + decomp_offset], ymm_store);
            }
        }
    }
};

} // namespace weight_decomp