#include <iostream>
#include <vector>
#include <cstdint>
#include <cstddef>  // 用于 offsetof
#include <cstring>  // 用于 memset
#include <cassert>
#include <sys/mman.h> // mmap, mprotect, munmap

#include "decomp_kernel_jit.hpp"

// -----------------------------------------------------------------
// 3. Main() 函数 - 测试样本
// -----------------------------------------------------------------

// 辅助函数：打印 ZMM 大小的缓冲区
void print_buffer(const char* title, const uint8_t* buf) {
    printf("%s:\n", title);
    for (int i = 0; i < 64; ++i) {
        printf("%02X ", buf[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    printf("\n");
}

int main() {
    std::cout << "Starting JIT Decompression Test (requires AVX-512)..." << std::endl;

    // --- 1. 定义测试参数 ---
    const int num_blocks = 1;
    const int chunks_per_block = 64; // cl 循环从 0 到 60 (共 64 个)
    const int chunk_size_bytes = 64; // ZMM 寄存器大小
    const int output_size = num_blocks * chunks_per_block * chunk_size_bytes; // 1*64*64 = 4096 字节

    // --- 2. 分配和创建测试数据 ---

    // 掩码：每个 chunk (64字节) 需要一个 64-bit 掩码
    std::vector<uint64_t> bitmask(chunks_per_block);

    // 压缩数据：大小是动态的，取决于 popcnt
    std::vector<uint8_t> compressed_data;

    // 预期输出：用于验证
    std::vector<uint8_t> expected_output(output_size, 0);

    // 目标缓冲区（实际输出）
    // 使用 aligned_alloc 以确保缓冲区对齐
    uint8_t* actual_output = (uint8_t*)aligned_alloc(64, output_size);
    memset(actual_output, 0, output_size);

    uint8_t data_val_counter = 0;

    for (int i = 0; i < chunks_per_block; ++i) {
        // 创建一个简单的、可预测的掩码
        // 例如：只选择第 0, 8, 16, 24, 32, 40, 48, 56 字节
        uint64_t mask = 0x0101010101010101;
        bitmask[i] = mask;

        // 计算这个掩码中有多少 '1'
        int popcnt = __builtin_popcountll(mask); // 8

        // 1. 填充压缩数据
        // 我们需要添加 'popcnt' 个字节的数据
        for (int p = 0; p < popcnt; ++p) {
            uint8_t val = static_cast<uint8_t>(0xA0 + i + p);
            compressed_data.push_back(val);
        }

        // 2. 填充预期输出
        size_t output_offset = i * chunk_size_bytes;
        int data_idx = 0;
        for (int bit = 0; bit < 64; ++bit) {
            if ((mask >> bit) & 1) {
                // 这是 'vpexpandb' 将放置数据的地方
                uint8_t val = static_cast<uint8_t>(0xA0 + i + data_idx);
                expected_output[output_offset + bit] = val;
                data_idx++;
            }
        }
    }

    // 确保压缩数据有足够的填充，以防对齐逻辑读取超出范围
    compressed_data.resize(compressed_data.size() + 128);


    std::cout << "Test data created." << std::endl;
    std::cout << " - Bitmask size: " << bitmask.size() * 8 << " bytes" << std::endl;
    std::cout << " - Compressed data size: " << compressed_data.size() - 128 << " bytes" << std::endl;
    std::cout << " - Output buffer size: " << output_size << " bytes" << std::endl;

    // --- 3. JIT 编译内核 ---
    jit_decompress_kernel_t kernel(num_blocks);

    // 获取指向 JIT 生成的机器码的函数指针
    auto* generated_func = kernel.getCode<void (*)(jit_decomp_params_t*)>();

    // 使代码可执行 (xbyak 默认是可写的)
    // mprotect 是更安全的方式，但 kernel.ready() 更简单
    kernel.ready();

    // --- 4. 设置参数并执行内核 ---
    jit_decomp_params_t args;
    args.compressed_buf = compressed_data.data();
    args.bitmask_ptr = bitmask.data();
    args.decomp_buf = actual_output;

    std::cout << "Executing JIT-compiled kernel..." << std::endl;
    generated_func(&args);
    std::cout << "Execution finished." << std::endl;

    // --- 5. 验证结果 ---
    int mismatches = 0;
    for (size_t i = 0; i < output_size; ++i) {
        if (actual_output[i] != expected_output[i]) {
            mismatches++;
        }
    }

    if (mismatches == 0) {
        std::cout << "\nSUCCESS! Decompression matches expected output." << std::endl;
    } else {
        std::cout << "\nFAILURE! Found " << mismatches << " mismatches." << std::endl;

        // 打印第一个不匹配的块
        for (size_t i = 0; i < output_size; i += 64) {
            if (memcmp(actual_output + i, expected_output.data() + i, 64) != 0) {
                print_buffer("Actual Output (Block)", actual_output + i);
                print_buffer("Expected Output (Block)", expected_output.data() + i);
                break;
            }
        }
    }

    // 释放内存
    free(actual_output);

    return mismatches > 0;
}
