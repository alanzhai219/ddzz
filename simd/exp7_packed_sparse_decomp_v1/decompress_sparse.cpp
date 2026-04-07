#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <limits.h>
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

// 结构体定义保持不变，用于参数传递
struct call_params_t {
    const void *src_ptr;      // 压缩后的数据源指针
    const void *bitmask_ptr;  // 稀疏位掩码指针 (uint64_t 数组)
    const void *dst_ptr;      // 解压缩后的目标指针
};

// 宏定义用于获取结构体成员的偏移量
#define GET_OFF(field) offsetof(call_params_t, field)

// 使用 Xbyak::CodeGenerator 替换 jit_generator_t
class SparseDecompressKernel : public Xbyak::CodeGenerator {
public:
    // CodeGenerator 构造函数: (Code Size, JIT mode)
    SparseDecompressKernel(const int mat_K_blk) : Xbyak::CodeGenerator(4096 * 4) {
        // 检查 CPU feature 是否支持 AVX512 (简化检查)
        if (!m_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
            throw std::runtime_error("AVX512 is not supported.");
        }
        
        K_blk = mat_K_blk;

        b_blk_sz_ = 64; // aligned with zmm width
        blk_sz_ = a_outter_blk_sz_ * b_blk_sz_ * a_inner_blk_sz_; // 16 * 64 * 4 = 4096
        nblks_to_decompress_ = K_blk * b_blk_sz_ / blk_sz_; // 简化为只处理一个 K 块

        generate_code(); // 调用核心生成逻辑
    }

    // 获取 JIT 函数指针
    void (*get_jit_func() const)(call_params_t *) {
        return reinterpret_cast<void (*)(call_params_t *)>(getCode());
    }

public:
    int K_blk;

    // 要解压的块数量 (Number of Blocks)
    // 这个变量是用来控制 JIT Kernel 需要处理多少个 blk_sz_ 大小的逻辑块。
    int nblks_to_decompress_ = 0; // 简化为只处理一个 K 块

    // 总块大小 (Total Block Size)：这是由前三个参数计算得出的一个完整的逻辑块包含的元素总数：
    int blk_sz_ = 0; // 16 * 64 * 4 = 4096

    // (Inner Block Size on B/Rows): 用于一次处理的基本块的行数, 通常是N维度 or B矩阵的行。
    // 在这个 JIT Kernel 中，它与 ZMM 寄存器 (64 字节/行) 的宽度相匹配。
    int b_blk_sz_ = 0;

    // Outer Block Size on K/Columns: 外层块大小: 通常是 K 维度上的一个更大块，用于提高数据局部性和并行度
    // 16 可能是与缓存线或 L2/L3 缓存大小相关的常数。
    const int a_outter_blk_sz_ = 16;

    // Inner Block Size on K/Columns: 内层块大小: 通常是K维度 or A矩阵的列
    const int a_inner_blk_sz_ = 4;

private:
    // 寄存器映射 (与原代码保持一致)
    const Xbyak::Reg64 reg_src_ptr = r8;
    const Xbyak::Reg64 reg_dst_ptr = r9;
    const Xbyak::Reg64 reg_bitmask_ptr = r10;
    const Xbyak::Reg64 reg_tmp = r11;
    const Xbyak::Reg64 reg_popcnt_tmp = r12;
    const Xbyak::Reg64 reg_popcnt = rcx; // 用于 popcnt 结果和 shl 计数

    Xbyak::Reg64 param1 = rdi;

    // Unroll Factor (UF) 辅助寄存器和 ZMM/Opmask
    const int unroll_factor = 4;

    Xbyak::Zmm get_zmm(int idx) {
        switch (idx) {
            case 0: return Xbyak::Zmm(25);
            case 1: return Xbyak::Zmm(26);
            case 2: return Xbyak::Zmm(27);
            case 3: return Xbyak::Zmm(28);
            default: assert(!"incorrect index"); return Xbyak::Zmm(0);
        }
    }

    Xbyak::Opmask get_opmask(int idx) {
        switch (idx) {
            case 0: return k1; break;
            case 1: return k2; break;
            case 2: return k3; break;
            case 3: return k4; break;
            default: assert(!"incorrect index"); return k0;
        }
    }

    Xbyak::Opmask get_load_mask(int idx) {
        // This function prepares a load mask for loading packed values. The minimum
        // and maximum number of bits that can be set is 0 and 64 respectively.
        // Since `shl` instruction doesn't work when the `count` operand is 64
        // (the instruction actually uses `count % 64` instead of just `count`)
        // we have to split 1 shift into 2 shifts (3 if there is a tail).
        //
        //
        // The following pseudo-code is implemented in JIT below:
        // shift = reg_popcnt / 2;
        // res = 1;
        // res = res << shift;
        // res = res << shift;
        // shift_tail = reg_popcnt % 2;
        // res = res << shift_tail;

        // Save original number of bits set in 1.
        mov(reg_popcnt_tmp, reg_popcnt);

        mov(reg_tmp, 1);

        // shift = reg_popcnt / 2
        shr(reg_popcnt, 1);

        // Apply shift two times.
        shl(reg_tmp, reg_popcnt.cvt8());
        shl(reg_tmp, reg_popcnt.cvt8());

        // Calculate shift_tail = reg_popcnt % 2.
        mov(reg_popcnt, reg_popcnt_tmp);
        and_(reg_popcnt, 1);

        // Apply shift_tail.
        shl(reg_tmp, reg_popcnt.cvt8());

        sub(reg_tmp, 1);

        // Restore the value (used to advance the pointer to packed values).
        mov(reg_popcnt, reg_popcnt_tmp);

        auto opmask = get_opmask(idx);
        kmovq(opmask, reg_tmp);

        return opmask;
    }

    Xbyak::Opmask get_expand_mask(int idx) {
        return get_opmask(idx);
    }

    Xbyak::Reg64 get_reg_mask_tmp(int idx) {
        switch (idx) {
            case 0: return r13;
            case 1: return r14;
            case 2: return r15;
            case 3: return rax;
            default: assert(!"incorrect index"); return Xbyak::Reg64(0);
        }
    };

    Xbyak::util::Cpu m_cpu;

    // 核心代码生成逻辑
    void generate_code() {
        // Xbyak 约定：param1 是第一个参数（本例中为 call_params_t*）

        // Xbyak::CodeGenerator 默认有 preamble/postamble，但为了纯粹性，手动实现
        // 手动保存 caller-save 寄存器 r8, r9, r10, r11, r12, r13, r14, r15, rax, rcx
        push(r13);
        push(r14);
        push(r15);
        push(rax);
        push(rcx); 
        push(r8);
        push(r9);
        push(r10);
        push(r11);
        push(r12); // 保存 callee-save/temp 

        // 2. 从 call_params_t* 中加载指针
        mov(reg_bitmask_ptr, ptr[param1 + GET_OFF(bitmask_ptr)]);
        mov(reg_dst_ptr, ptr[param1 + GET_OFF(dst_ptr)]);
        mov(reg_src_ptr, ptr[param1 + GET_OFF(src_ptr)]);

        assert(unroll_factor == 4);

        for (int b = 0; b < nblks_to_decompress_; b++) {
            const int blk_offset = b * blk_sz_;
            const int bitmask_off = blk_offset / CHAR_BIT;
            const int nbytes_per_load = 64;

            for (int i = 0; i < b_blk_sz_; i += unroll_factor) {
                for (int uf = 0; uf < unroll_factor; uf++) {
                    auto reg_mask_tmp = get_reg_mask_tmp(uf);
                    mov(reg_mask_tmp, ptr[reg_bitmask_ptr + (i + uf) * sizeof(uint64_t) + bitmask_off]);
                    popcnt(reg_popcnt, reg_mask_tmp);

                    auto load_mask = get_load_mask(uf);
                    auto zmm_reg = get_zmm(uf);
                    vmovdqu8(zmm_reg | load_mask | T_z, ptr[reg_src_ptr]);
                    add(reg_src_ptr, reg_popcnt);

                    auto expand_mask = get_expand_mask(uf);
                    kmovq(expand_mask, reg_mask_tmp);
                    vpexpandb(zmm_reg | expand_mask | T_z, zmm_reg);
                    vmovdqu8(ptr[reg_dst_ptr + blk_offset + (i + uf) * nbytes_per_load], zmm_reg);
                }
            }
        }

        // 5. 恢复寄存器
        pop(r12);
        pop(r11);
        pop(r10);
        pop(r9);
        pop(r8);
        pop(rcx);
        pop(rax);
        pop(r15);
        pop(r14);
        pop(r13);
        
        ret();
    }
};

#undef GET_OFF

// --- 测试代码 ---
void run_test() {
    std::cout << "--- 稀疏解压缩 JIT Kernel 测试 ---" << std::endl;

    // 1. 初始化 Kernel
    const int K_blk = 128;
    SparseDecompressKernel kernel(K_blk);
    auto jit_func = kernel.get_jit_func();

    // 2. 设置参数
    const int B_BLK = kernel.b_blk_sz_; // 64
    const int BYTES_PER_ROW = 64; // ZMM load size (64 bytes)
    const int NUM_ROWS = B_BLK; // 64 行
    const int nblks = kernel.nblks_to_decompress_;
    const int BITMASK_SIZE = NUM_ROWS * sizeof(uint64_t);

    // a. 稀疏位掩码 (64行，每行一个 uint64_t 掩码)
    std::vector<uint64_t> bitmask(NUM_ROWS, 0);
    // 示例掩码: 
    // row 0: 0xF (低 4 位稀疏) -> popcnt=4
    // row 1: 0xFF (低 8 位稀疏) -> popcnt=8
    // row 2: 0x0 (全稀疏) -> popcnt=0
    // row 3: 0xFFFFFFFFFFFFFFFF (全密) -> popcnt=64
    // row 4: 0x8000000000000000 (最高 1 位稀疏) -> popcnt=1

    bitmask[0] = 0xF; 
    bitmask[1] = 0xFF;
    bitmask[2] = 0x0;
    bitmask[3] = 0xFFFFFFFFFFFFFFFF;
    bitmask[4] = 0x8000000000000000;
    // 填充剩余行，以确保循环执行完整
    for (int i = 5; i < NUM_ROWS; ++i) {
         if (i % 2 == 0) bitmask[i] = 0x5555555555555555; // 32 活跃
         else bitmask[i] = 0xAAAAAAAAAAAAAA; // 32 活跃
    }

    // b. 压缩后的源数据 (Src)
    // 需要空间容纳所有 popcnt 值的和
    size_t total_packed_size = 0;
    for (uint64_t mask : bitmask) {
        total_packed_size += __builtin_popcountll(mask);
    }

    std::vector<uint8_t> src_data(total_packed_size);
    // 填充测试数据: 1, 2, 3, 4, 5, ...
    for (size_t i = 0; i < total_packed_size; ++i) {
        src_data[i] = (uint8_t)(i + 1); 
    }

    // c. 解压缩后的目标数据 (Dst) - 稀疏矩阵
    const int DST_SIZE = nblks * NUM_ROWS * BYTES_PER_ROW; // 64 * 64 = 4096 字节
    std::vector<uint8_t> dst_data(DST_SIZE, 0xCC); // 填充 CC 方便查看未写入的区域

    // d. JIT 参数结构体
    call_params_t params = {
        src_data.data(),
        bitmask.data(),
        dst_data.data()
    };
    
    // 3. 执行 JIT Kernel
    std::cout << "-> 执行 JIT Kernel..." << std::endl;
    jit_func(&params);
    std::cout << "-> 执行完成。" << std::endl;

    // 4. 验证结果
    std::cout << "--- 验证结果 ---" << std::endl;
    int src_idx = 0;
    bool success = true;

    for (int i = 0; i < NUM_ROWS; ++i) {
        uint64_t mask = bitmask[i];
        int popcnt = __builtin_popcountll(mask);
        
        std::cout << "Row " << i << ": Mask=0x" << std::hex << mask << ", Popcnt=" << std::dec << popcnt << std::endl;

        for (int j = 0; j < BYTES_PER_ROW; ++j) {
            bool is_active = (mask >> j) & 0x1;
            uint8_t expected_val;

            if (is_active) {
                // 预期值应该是压缩数据中的下一个值
                expected_val = src_data[src_idx];
                src_idx++;
            } else {
                // 稀疏位置应为 0 (T_z 零掩码作用)
                expected_val = 0; 
            }

            uint8_t actual_val = dst_data[i * BYTES_PER_ROW + j];

            if (actual_val != expected_val) {
                std::cerr << "FAIL at Dst[" << i << ", " << j << "]: Expected 0x" << std::hex << (int)expected_val 
                          << ", Actual 0x" << (int)actual_val << std::endl;
                success = false;
                // 为了避免输出过多错误，只展示前几个失败
                if (src_idx > 100) return; 
            } else if (is_active && j < 4) {
                 // 打印前几个活跃元素以供查看
                 std::cout << "  Active[" << j << "]: Value=" << (int)actual_val << std::endl;
            }
        }
    }

    if (success) {
        std::cout << "\n✅ 所有验证通过！稀疏解压缩成功。" << std::endl;
    } else {
        std::cout << "\n❌ 验证失败！请检查 JIT 代码中的寄存器使用和寻址逻辑。" << std::endl;
    }
}

int main() {
    try {
        run_test();
    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}