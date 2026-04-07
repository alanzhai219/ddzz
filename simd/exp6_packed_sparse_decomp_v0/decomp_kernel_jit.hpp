#include <stddef.h>
// 包含 xbyak 头文件
#include "xbyak/xbyak.h"

// -----------------------------------------------------------------
// 1. 定义 JIT 内核将访问的参数结构体
// -----------------------------------------------------------------

struct jit_decomp_params_t {
    const void* compressed_buf;        // 指向压缩权重
    const void* bitmask_ptr;  // 指向位掩码
    void* decomp_buf;  // 指向解压的目标缓冲区
};

// 定义宏以匹配原始代码中的 GET_OFF
#define GET_OFF(field) (offsetof(jit_decomp_params_t, field))

// -----------------------------------------------------------------
// 2. 完整的 JIT 内核类 (包含您的 generate() 函数)
// -----------------------------------------------------------------

class jit_decompress_kernel_t : public Xbyak::CodeGenerator {
public:
    // --- 寄存器定义 ---
    // 假设是 Linux x64 ABI: param1 在 rdi
    Xbyak::Reg64 param1 = rdi;

    // 工作指针
    Xbyak::Reg64 compressed_ptr = r8;
    Xbyak::Reg64 reg_ptr_compressed_bitmask = r9;
    Xbyak::Reg64 reg_ptr_decomp_dst = r10;
    Xbyak::Reg64 reg_ptr_compressed_src = r11;

    // 临时通用寄存器
    Xbyak::Reg64 reg_comp_mask_tmp1 = r12;
    Xbyak::Reg64 reg_comp_mask_tmp2 = r13;
    Xbyak::Reg64 reg_comp_mask_tmp3 = r14;
    Xbyak::Reg64 reg_comp_mask_tmp4 = r15;
    Xbyak::Reg64 reg_popcnt = rax;
    Xbyak::Reg64 reg_ptr_compressed_src_align = rdx;

    // K-Mask (Opmask) 寄存器
    Xbyak::Opmask reg_comp_mask1 = k1;
    Xbyak::Opmask reg_comp_mask2 = k2;
    Xbyak::Opmask reg_comp_mask3 = k3;
    Xbyak::Opmask reg_comp_mask4 = k4;

    // ZMM (AVX-512) 寄存器
    Xbyak::Zmm zmm_comp0 = zmm0;
    Xbyak::Zmm zmm_comp1 = zmm1;
    Xbyak::Zmm zmm_comp2 = zmm2;
    Xbyak::Zmm zmm_comp3 = zmm3;

    // xbyak 定义的零掩码 (Zeroing mask)
    // Xbyak::Emask T_z = T_z;

    // 成员变量
    int blocks_;
    const int bytes_per_block = 4096;
    const int chunks_per_block = 64;
    const int bytes_per_chunk = 64;

    // --- 构造函数 ---
    jit_decompress_kernel_t(int blocks) : Xbyak::CodeGenerator(4096 * 4096), blocks_(blocks) {
        generate(); // JIT 编译在构造时发生
    }

    // --- 您的 generate() 函数 ---
    // (已从您的提示中复制并粘贴到此类中)
    void generate() {
        // preamble();
        // --- 替换 preamble(); ---
        // 保存 GPRs (r12-r15)
        push(r12);
        push(r13);
        push(r14);
        push(r15);

        // 保存 K 寄存器 (k1-k4)
        // 无法直接 push K 寄存器, 必须先 mov 到 GPR
        kmovq(rax, k1); push(rax);
        kmovq(rax, k2); push(rax);
        kmovq(rax, k3); push(rax);
        kmovq(rax, k4); push(rax);

        // input
        mov(compressed_ptr, ptr[param1 + GET_OFF(compressed_buf)]);
        mov(reg_ptr_compressed_bitmask, ptr[param1 + GET_OFF(bitmask_ptr)]);
        // output
        mov(reg_ptr_decomp_dst, ptr[param1 + GET_OFF(decomp_buf)]);

        // like address assignment:
        // uint8_t* reg_ptr_compressed_src = compressed_ptr;
        lea(reg_ptr_compressed_src, ptr[compressed_ptr]);

        for (int block = 0; block < blocks_; block++) {
            int wei_offset = block * bytes_per_block;
            int current_bitmask_offset = block * chunks_per_block;

            for (int cl = 0; cl < chunks_per_block; cl = cl + 4) {
                // step 1. loading mask
                // 0 <-> chunk <-> mask
                mov(reg_comp_mask_tmp1, ptr[reg_ptr_compressed_bitmask + current_bitmask_offset + cl * 8]);
                kmovq(reg_comp_mask1, reg_comp_mask_tmp1);

                // 1 <-> chunk <-> mask
                mov(reg_comp_mask_tmp2, ptr[reg_ptr_compressed_bitmask + current_bitmask_offset + (cl + 1) * 8]);
                kmovq(reg_comp_mask2, reg_comp_mask_tmp2);

                // 2 <-> chunk <-> mask
                mov(reg_comp_mask_tmp3, ptr[reg_ptr_compressed_bitmask + current_bitmask_offset + (cl + 2) * 8]);
                kmovq(reg_comp_mask3, reg_comp_mask_tmp3);

                // 3 <-> chunk <-> mask
                mov(reg_comp_mask_tmp4, ptr[reg_ptr_compressed_bitmask + current_bitmask_offset + (cl + 3) * 8]);
                kmovq(reg_comp_mask4, reg_comp_mask_tmp4);

                // step 2. loading compressed data ptr by mask
                // 0 <-> zmm1 <-> mask
                vmovdqu8(zmm_comp0, ptr[reg_ptr_compressed_src]);
                popcnt(reg_popcnt, reg_comp_mask_tmp1);
                add(reg_ptr_compressed_src, reg_popcnt);

                // 1 <-> chunk <-> mask
                vmovdqu8(zmm_comp1, ptr[reg_ptr_compressed_src]);
                popcnt(reg_popcnt, reg_comp_mask_tmp2);
                add(reg_ptr_compressed_src, reg_popcnt);

                // 2 <-> chunk <-> mask
                vmovdqu8(zmm_comp2, ptr[reg_ptr_compressed_src]);
                popcnt(reg_popcnt, reg_comp_mask_tmp3);
                add(reg_ptr_compressed_src, reg_popcnt);

                // 3 <-> chunk <-> mask
                vmovdqu8(zmm_comp3, ptr[reg_ptr_compressed_src]);
                popcnt(reg_popcnt, reg_comp_mask_tmp4);
                add(reg_ptr_compressed_src, reg_popcnt);

                // step 3. decompress by mask
                vpexpandb(zmm_comp0 | reg_comp_mask1 | T_z, zmm_comp0);
                vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + cl * 64], zmm_comp0);

                vpexpandb(zmm_comp1 | reg_comp_mask2 | T_z, zmm_comp1);
                vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + (cl + 1) * 64], zmm_comp1);

                vpexpandb(zmm_comp2 | reg_comp_mask3 | T_z, zmm_comp2);
                vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + (cl + 2) * 64], zmm_comp2);

                vpexpandb(zmm_comp3 | reg_comp_mask4 | T_z, zmm_comp3);
                vmovdqu8(ptr[reg_ptr_decomp_dst + wei_offset + (cl + 3) * 64], zmm_comp3);
            }

            // XXX: memory alignment of weights buffer can lead to issues.
            // padding = (64 - (ptr % 64)) % 64
            // 获取当前地址
            // size_t offset = (size_t)current_src_ptr;
            // // 计算未对齐量 (offset % 64)
            // size_t misalignment = offset & 0x3F; 
            // if (misalignment != 0) {
            //     // 移动指针到下一个边界
            //     current_src_ptr += (64 - misalignment);
            // }

            mov(reg_ptr_compressed_src_align, reg_ptr_compressed_src);
            neg(reg_ptr_compressed_src_align);
            // not_(reg_ptr_compressed_src_align);
            // and_(reg_ptr_compressed_src_align, 0x3f); // get 6 LSBs
            // add(reg_ptr_compressed_src_align, 0x1);
            and_(reg_ptr_compressed_src_align, 0x3f); // 0x0 if already aligned to cacheline
            add(reg_ptr_compressed_src, reg_ptr_compressed_src_align);
        }
        // postamble();
        // --- 替换 postamble(); ---
        // 恢复 K 寄存器 (k1-k4) (顺序相反)
        pop(rax); kmovq(k4, rax);
        pop(rax); kmovq(k3, rax);
        pop(rax); kmovq(k2, rax);
        pop(rax); kmovq(k1, rax);

        // 恢复 GPRs (r12-r15) (顺序相反)
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);

        ret(); // 函数返回
    }
};