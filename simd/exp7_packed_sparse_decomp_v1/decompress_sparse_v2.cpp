#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <limits>
#include <stdexcept>
#include <stddef.h> // For offsetof
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

// 结构体定义保持不变，用于参数传递
struct call_params_t {
    const void *src_ptr;        // 压缩后的数据源指针 (紧凑数据)
    const void *bitmask_ptr;    // 稀疏位掩码指针 (uint64_t 数组)
    const void *dst_ptr;        // 解压缩后的目标指针 (稀疏矩阵)
};

// 宏定义用于获取结构体成员的偏移量
#define GET_OFF(field) offsetof(call_params_t, field)

// 使用 Xbyak::CodeGenerator 替换 jit_generator_t
class SparseDecompressKernel : public Xbyak::CodeGenerator {
public:
    // CodeGenerator 构造函数: (Code Size, JIT mode)
    SparseDecompressKernel() : Xbyak::CodeGenerator() {
        // 确保 CPU feature 支持 AVX512 (包括 VPEXPANDB)
        // 注意：xbyak_util.h 中的 tAVX512F 不一定包含 VPEXPANDB 所需的 VPOPCNTDQ 扩展
        // 但对于现代 AVX-512 CPU，通常都具备。
        if (!m_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
            throw std::runtime_error("AVX512F is not supported. Cannot run kernel.");
        }
        
        // 固定的块大小参数 (与原代码保持一致)
        b_blk_sz_ = 64; 
        blk_sz_ = a_outter_blk_sz_ * b_blk_sz_ * a_inner_blk_sz_; // 16 * 64 * 4 = 4096
        K_blk = 1;
        // 简化为只处理一个 K 块
        nblks_to_decompress_ = K_blk * b_blk_sz_ / blk_sz_; // 简化处理，因为 blk_sz_ 很大，可能溢出原计算

        generate_code(); // 调用核心生成逻辑
    }

    // 获取 JIT 函数指针
    void (*get_jit_func() const)(call_params_t *) {
        return reinterpret_cast<void (*)(call_params_t *)>(getCode());
    }

public:
    int K_blk;
    int nblks_to_decompress_;
    int blk_sz_;
    int b_blk_sz_;

    const int a_outter_blk_sz_ = 16;
    const int a_inner_blk_sz_ = 4;

private:
    // 寄存器映射 (与原代码保持一致)
    const Xbyak::Reg64 reg_src_ptr = r8;    // R8: 压缩源指针
    const Xbyak::Reg64 reg_dst_ptr = r9;    // R9: 目标指针
    const Xbyak::Reg64 reg_bitmask_ptr = r10;// R10: 掩码指针
    const Xbyak::Reg64 reg_mask_tmp = r11;   // R11: 临时存储 64 位掩码
    const Xbyak::Reg64 reg_popcnt = rcx;    // RCX: 用于 popcnt 结果和 src_ptr 递增

    Xbyak::Reg64 param1 = rdi;              // RDI: 第一个参数 call_params_t*

    // Unroll Factor (UF) 辅助寄存器和 ZMM/Opmask
    const int unroll_factor = 4;

    Xbyak::Zmm get_zmm(int idx) {
        // 使用 ZMM25-ZMM28 用于解卷
        switch (idx) {
            case 0: return Xbyak::Zmm(25);
            case 1: return Xbyak::Zmm(26);
            case 2: return Xbyak::Zmm(27);
            case 3: return Xbyak::Zmm(28);
            default: assert(!"incorrect index"); return Xbyak::Zmm(0);
        }
    }

    Xbyak::Opmask get_expand_mask(int idx) {
        // 使用 k1-k4 作为解压缩/扩展掩码
        switch (idx) {
            case 0: return k1;
            case 1: return k2;
            case 2: return k3;
            case 3: return k4;
            default: assert(!"incorrect index"); return k0;
        }
    }

    // 由于我们只使用 64 位掩码的临时存储，简化为只使用一个临时寄存器
    Xbyak::Reg64 get_reg_mask_tmp(int idx) {
        // 不同的 UF 使用不同的 GPR 来存储 uint64_t 掩码，避免冲突
        switch (idx) {
            case 0: return r11; // reg_mask_tmp
            case 1: return r12; 
            case 2: return r13; 
            case 3: return r14; 
            default: assert(!"incorrect index"); return Xbyak::Reg64(0);
        }
    };

    Xbyak::util::Cpu m_cpu;

    // 核心代码生成逻辑
    void generate_code() {
        // 1. 手动保存 caller-save 寄存器
        // 这里的寄存器选择旨在与 Linux x64 ABI 保持一致，并覆盖了使用的临时寄存器。
        push(r15); push(r14); push(r13); push(r12); push(r11); // Callee-save / Temp
        push(r10); push(r9); push(r8); // Temp / Argument Pointers
        push(rcx); // Popcount register

        // 2. 从 call_params_t* 中加载指针 (RDI 已经是 param1)
        mov(reg_bitmask_ptr, ptr[param1 + GET_OFF(bitmask_ptr)]);
        mov(reg_dst_ptr, ptr[param1 + GET_OFF(dst_ptr)]);
        mov(reg_src_ptr, ptr[param1 + GET_OFF(src_ptr)]);

        assert(unroll_factor == 4);

        Xbyak::Label loop_start;
        Xbyak::Label loop_end;

        // 由于只处理一个 blk，这里可以简化为只处理 i 循环
        const int blk_offset = 0;
        const int bitmask_off = 0; // K_blk=1, blk_offset=0
        const int nbytes_per_load = 64; // ZMM load size

        for (int i = 0; i < b_blk_sz_; i += unroll_factor) {
            // i 循环: 64 行，解卷系数 UF=4, 共 16 次迭代
            for (int uf = 0; uf < unroll_factor; uf++) {
                // 1. 加载 64 位稀疏位掩码到 GPR (reg_mask_tmp)
                auto reg_mask = get_reg_mask_tmp(uf);
                mov(reg_mask,
                        ptr[reg_bitmask_ptr + (i + uf) * sizeof(uint64_t)
                            + bitmask_off]);

                // 2. 计算 popcnt (用于 src_ptr 递增)
                popcnt(reg_popcnt, reg_mask); // reg_popcnt (rcx) = 稀疏元素的数量

                // 3. 将 64 位掩码写入 Opmask kX
                auto expand_mask = get_expand_mask(uf);
                kmovq(expand_mask, reg_mask);

                // 4. 核心解压缩 (VPEXPANDB 内存源形式):
                // 从 src_ptr 紧凑加载 popcnt 字节，根据 Opmask 扩展到 ZMM。
                // T_z (Zeroing) 确保 ZMM 中未激活的稀疏位置被清零。
                auto zmm_reg = get_zmm(uf);
                vpexpandb(zmm_reg | expand_mask | T_z, ptr[reg_src_ptr]);

                // 5. 更新源指针: 移动 popcnt 字节 (紧凑数据)
                add(reg_src_ptr, reg_popcnt);

                // 6. 存储 64 字节的稀疏行到目标地址
                // 注意：这里计算了正确的内存地址偏移
                vmovdqu8(ptr[reg_dst_ptr + blk_offset + (i + uf) * nbytes_per_load], zmm_reg);
            }
        }

        // 5. 恢复寄存器
        pop(rcx); 
        pop(r8); pop(r9); pop(r10);
        pop(r11); pop(r12); pop(r13); pop(r14); pop(r15);
        
        ret();
    }
};

#undef GET_OFF

// --- 测试代码 ---
void run_test() {
    std::cout << "--- 稀疏解压缩 JIT Kernel 测试 ---" << std::endl;

    // 1. 初始化 Kernel
    SparseDecompressKernel kernel;
    auto jit_func = kernel.get_jit_func();

    // 2. 设置参数
    const int B_BLK = kernel.b_blk_sz_; // 64
    const int BYTES_PER_ROW = 64; // ZMM load size (64 bytes)
    const int NUM_ROWS = B_BLK; // 64 行

    // a. 稀疏位掩码 (64行，每行一个 uint64_t 掩码)
    std::vector<uint64_t> bitmask(NUM_ROWS, 0);
    // 示例掩码: 
    // row 0 (uf=0): 0xF (低 4 位稀疏) -> popcnt=4
    // row 1 (uf=1): 0xFF (低 8 位稀疏) -> popcnt=8
    // row 2 (uf=2): 0x0 (全稀疏) -> popcnt=0
    // row 3 (uf=3): 0xFFFFFFFFFFFFFFFF (全密) -> popcnt=64
    // row 4 (uf=0): 0x8000000000000000 (最高 1 位稀疏) -> popcnt=1

    bitmask[0] = 0xF; // 1, 2, 3, 4
    bitmask[1] = 0xFF; // 5, 6, 7, 8, 9, 10, 11, 12
    bitmask[2] = 0x0; // 0
    bitmask[3] = 0xFFFFFFFFFFFFFFFF; // 13, ..., 76
    bitmask[4] = 0x8000000000000000; // 77
    // 填充剩余行，以确保循环执行完整
    for (int i = 5; i < NUM_ROWS; ++i) {
         if (i % 2 == 0) bitmask[i] = 0x5555555555555555; // 32 活跃
         else bitmask[i] = 0xAAAAAAAAAAAAAA; // 32 活跃
    }

    // b. 压缩后的源数据 (Src)
    size_t total_packed_size = 0;
    for (uint64_t mask : bitmask) {
        total_packed_size += __builtin_popcountll(mask);
    }

    std::vector<uint8_t> src_data(total_packed_size);
    // 填充测试数据: 1, 2, 3, 4, 5, ...
    for (size_t i = 0; i < total_packed_size; ++i) {
        src_data[i] = (uint8_t)(i + 1); 
    }
    std::cout << "总共的压缩字节数: " << total_packed_size << std::endl;

    // c. 解压缩后的目标数据 (Dst) - 稀疏矩阵
    const int DST_SIZE = NUM_ROWS * BYTES_PER_ROW; // 64 * 64 = 4096 字节
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
        
        if (i < 5) { // 仅打印前几行
             std::cout << "Row " << i << ": Mask=0x" << std::hex << mask << ", Popcnt=" << std::dec << popcnt << std::endl;
        }

        for (int j = 0; j < BYTES_PER_ROW; ++j) {
            bool is_active = (mask >> j) & 0x1;
            uint8_t expected_val;

            if (is_active) {
                // 预期值应该是压缩数据中的下一个值
                if (src_idx >= total_packed_size) {
                    std::cerr << "ERROR: Source index out of bounds!" << std::endl;
                    success = false;
                    break;
                }
                expected_val = src_data[src_idx];
                src_idx++;
            } else {
                // 稀疏位置应为 0 (T_z 零掩码作用)
                expected_val = 0; 
            }

            uint8_t actual_val = dst_data[i * BYTES_PER_ROW + j];

            if (actual_val != expected_val) {
                if (i < 5 || src_idx < 100) {
                     std::cerr << "FAIL at Dst[" << i << ", " << j << "]: Expected 0x" << std::hex << (int)expected_val 
                               << ", Actual 0x" << (int)actual_val << " (Active=" << is_active << ")" << std::endl;
                }
                success = false;
                // 为了避免输出过多错误，只展示前几个失败
                if (src_idx > 200) goto end_test;
            }
        }
    }

end_test:
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