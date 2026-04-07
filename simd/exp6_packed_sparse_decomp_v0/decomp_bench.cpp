#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>  // memset, memcmp
#include <cassert>
#include <chrono>
#include <cstdlib>  // aligned_alloc, free
#include <iomanip>  // 用于 std::setw, std::fixed, std::hex, std::setfill

#include "decomp_kernel_ref.hpp"
#include "decomp_kernel_jit.hpp"
#include "decomp_kernel_avx512.hpp"
#include "decomp_kernel_ref_opt.hpp"
#include "decomp_kernel_avx512_opt.hpp"
#include "decomp_kernel_jit_opt.hpp"

void gen_origin_data(std::vector<uint8_t>& original_uncompressed_data,
                     const size_t output_size) {
    std::cout << "Step 1: Original uncompressed sparse data created (Random)." << std::endl;
    // 使用随机数据填充，模拟稀疏性
    srand(42);
    for (size_t i = 0; i < output_size; ++i) {
        // 70% 概率为 0
        if (rand() % 10 < 7) {
            original_uncompressed_data[i] = 0;
        } else {
            original_uncompressed_data[i] = (uint8_t)((rand() % 255) + 1);
        }
    }
}

// =================================================================
// ---- 新增的压缩函数 ----
// 这个函数接收未压缩的稀疏数据，并在压缩过程中生成掩码和压缩后的数据流
// =================================================================
void compress_weights(
    std::vector<uint8_t>& compressed_output,
    std::vector<uint64_t>& bitmask_output,
    const std::vector<uint8_t>& uncompressed_input,
    int num_blocks,
    int chunks_per_block) {
    
    compressed_output.clear(); // data size is aligned with 64B instead of address.
    bitmask_output.resize(num_blocks * chunks_per_block);

    for (int block = 0; block < num_blocks; ++block) {
        for (int cl = 0; cl < chunks_per_block; ++cl) {
            size_t chunk_idx = (size_t)block * chunks_per_block + cl;
            const uint8_t* uncompressed_chunk_ptr = &uncompressed_input[chunk_idx * 64];

            // 遍历原始数据，生成掩码并提取非零字节
            uint64_t mask = 0;
            for (int b = 0; b < 64; ++b) {
                if (uncompressed_chunk_ptr[b] != 0) {
                    mask |= (1ULL << b);
                    compressed_output.push_back(uncompressed_chunk_ptr[b]);
                }
            }
            bitmask_output[chunk_idx] = mask;
        }

        // --- 缓存行对齐逻辑 (与内核中镜像) ---
        // 在compressed data每个block的末尾，添加填充以模拟指针推进
        size_t current_size = compressed_output.size();
        size_t misalignment = current_size & 0x3F;
        if (misalignment != 0) {
            size_t padding_needed = 64 - misalignment;
            for (size_t p = 0; p < padding_needed; ++p) {
                compressed_output.push_back(0xDD); // 插入“死”字节作为填充
            }
        }
    }
    
    // 确保末尾有足够的填充，以防越界读取
    compressed_output.resize(compressed_output.size() + 128, 0xDD);
    std::cout << "Step 2: Compression logic executed." << std::endl;
}

// -----------------------------------------------------------------
// ---- 5. Main 函数 - 基准测试和验证 ----
// -----------------------------------------------------------------
int main() {
    // --- 1. 设置测试参数 ---
    const int num_blocks = 1000; // 运行 1000 个 4KB 的块
    const int elts_per_block = 4096;
    const int chunks_per_block = 64;
    const int total_chunks = num_blocks * chunks_per_block;
    const size_t output_size = (size_t)num_blocks * elts_per_block;
    const int num_runs = 10; // 多次运行取平均

    std::cout << "--- AVX-512 Decompression Benchmark ---" << std::endl;
    std::cout << "Total output size: " << (output_size / 1024.0 / 1024.0) << " MB" << std::endl;
    std::cout << "Num runs per kernel: " << num_runs << std::endl;

    // --- 2. 分配缓冲区 ---
    uint8_t* output_scalar_nounroll = (uint8_t*)aligned_alloc(64, output_size);
    uint8_t* output_scalar_unroll = (uint8_t*)aligned_alloc(64, output_size);
    uint8_t* output_avx512_nounroll = (uint8_t*)aligned_alloc(64, output_size);
    uint8_t* output_avx512_unroll = (uint8_t*)aligned_alloc(64, output_size);
    uint8_t* output_jit = (uint8_t*)aligned_alloc(64, output_size);
    uint8_t* output_avx512_opt = (uint8_t*)aligned_alloc(64, output_size);
    uint8_t* output_jit_opt = (uint8_t*)aligned_alloc(64, output_size);

    if (!output_scalar_nounroll || !output_scalar_unroll || !output_avx512_nounroll ||
        !output_avx512_unroll || !output_jit || !output_avx512_opt || !output_jit_opt) {
        std::cerr << "Failed to allocate aligned memory" << std::endl;
        return 1;
    }

    // =================================================================
    // ---- 第 1 步: 构建原始非压缩数据 ----
    // =================================================================
    std::vector<uint8_t> original_uncompressed_data(output_size, 0);
    gen_origin_data(original_uncompressed_data, output_size);

    // =================================================================
    // ---- 第 2 步: 实现压缩逻辑 (同时生成掩码) ----
    // =================================================================
    std::vector<uint8_t> compressed_data;
    std::vector<uint64_t> bitmask(total_chunks);
    compress_weights(compressed_data, bitmask, original_uncompressed_data, num_blocks, chunks_per_block);
    std::cout << "Compressed size (with padding): " << (compressed_data.size() / 1024.0 / 1024.0) << " MB (70% sparsity)" << std::endl;

    // =================================================================
    // ---- 第 3 步: 验证所有内核的正确性 (单次运行) ----
    // =================================================================
    std::cout << "\n--- Correctness Verification ---" << std::endl;

    auto verify = [&](const char* name, uint8_t* buf) -> bool {
        if (memcmp(buf, original_uncompressed_data.data(), output_size) != 0) {
            std::cerr << "FAILURE: " << name << " verification failed!" << std::endl;
            return false;
        }
        std::cout << "SUCCESS: " << name << " verified." << std::endl;
        return true;
    };

    // Scalar (No Unroll)
    memset(output_scalar_nounroll, 0, output_size);
    decompress_scalar_nounroll(output_scalar_nounroll, compressed_data.data(), bitmask.data(), num_blocks);
    verify("Scalar (No Unroll)", output_scalar_nounroll);

    // Scalar (4x Unroll)
    memset(output_scalar_unroll, 0, output_size);
    decompress_scalar(output_scalar_unroll, compressed_data.data(), bitmask.data(), num_blocks);
    verify("Scalar (4x Unroll)", output_scalar_unroll);

    // AVX-512 (No Unroll)
    memset(output_avx512_nounroll, 0, output_size);
    decompress_avx512_nounroll(output_avx512_nounroll, compressed_data.data(), bitmask.data(), num_blocks);
    verify("AVX-512 (No Unroll)", output_avx512_nounroll);

    // AVX-512 (4x Unroll)
    memset(output_avx512_unroll, 0, output_size);
    decompress_avx512(output_avx512_unroll, compressed_data.data(), bitmask.data(), num_blocks);
    verify("AVX-512 (4x Unroll)", output_avx512_unroll);

    // AVX-512 Opt (4x Unroll)
    memset(output_avx512_opt, 0, output_size);
    decompress_avx512_opt(output_avx512_opt, compressed_data.data(), bitmask.data(), num_blocks);
    verify("AVX-512 Opt (4x Unroll)", output_avx512_opt);

    // JIT (original)
    memset(output_jit, 0, output_size);
    jit_decompress_kernel_t kernel_orig(num_blocks);
    auto* jit_func_orig = kernel_orig.getCode<void (*)(jit_decomp_params_t*)>();
    kernel_orig.ready();
    jit_decomp_params_t args_orig;
    args_orig.compressed_buf = compressed_data.data();
    args_orig.bitmask_ptr = bitmask.data();
    args_orig.decomp_buf = (void*)output_jit;
    jit_func_orig(&args_orig);
    verify("JIT (original)", output_jit);

    // JIT Opt (prefix-sum)
    memset(output_jit_opt, 0, output_size);
    jit_decompress_kernel_t_opt kernel_opt(num_blocks);
    auto* jit_func_opt = kernel_opt.getCode<void (*)(jit_decomp_params_t_opt*)>();
    kernel_opt.ready();
    jit_decomp_params_t_opt args_opt;
    args_opt.compressed_buf = compressed_data.data();
    args_opt.bitmask_ptr = bitmask.data();
    args_opt.decomp_buf = (void*)output_jit_opt;
    jit_func_opt(&args_opt);
    verify("JIT Opt (prefix-sum)", output_jit_opt);

    // =================================================================
    // ---- 第 4 步: 性能基准测试 (多次运行取平均) ----
    // =================================================================
    std::cout << "\n--- Running Performance Benchmarks (" << num_runs << " runs each) ---" << std::endl;

    // Helper lambda for benchmarking
    auto bench = [&](const char* name, auto func, int runs) -> double {
        double total = 0;
        for (int r = 0; r < runs; ++r) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            total += elapsed.count();
        }
        return total / runs;
    };

    double time_s_nounroll = bench("Scalar (No Unroll)", [&]() {
        decompress_scalar_nounroll(output_scalar_nounroll, compressed_data.data(), bitmask.data(), num_blocks);
    }, num_runs);

    double time_s_unroll = bench("Scalar (4x Unroll)", [&]() {
        decompress_scalar(output_scalar_unroll, compressed_data.data(), bitmask.data(), num_blocks);
    }, num_runs);

    double time_avx_nounroll = bench("AVX-512 (No Unroll)", [&]() {
        decompress_avx512_nounroll(output_avx512_nounroll, compressed_data.data(), bitmask.data(), num_blocks);
    }, num_runs);

    double time_avx_unroll = bench("AVX-512 (4x Unroll)", [&]() {
        decompress_avx512(output_avx512_unroll, compressed_data.data(), bitmask.data(), num_blocks);
    }, num_runs);

    double time_avx_opt = bench("AVX-512 Opt (4x Unroll)", [&]() {
        decompress_avx512_opt(output_avx512_opt, compressed_data.data(), bitmask.data(), num_blocks);
    }, num_runs);

    double time_jit_orig = bench("JIT (original)", [&]() {
        jit_func_orig(&args_orig);
    }, num_runs);

    double time_jit_opt = bench("JIT Opt (prefix-sum)", [&]() {
        jit_func_opt(&args_opt);
    }, num_runs);

    // --- 打印结果 ---
    std::cout << "\n--- Performance Results (avg over " << num_runs << " runs) ---" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    const int w = 35;

    auto print_row = [&](const char* name, double time_ms, double baseline_ms) {
        std::cout << std::setw(w) << std::left << name << ": "
                  << std::setw(10) << std::right << time_ms << " ms"
                  << "  (" << std::setprecision(2) << (baseline_ms / time_ms) << "x vs baseline)"
                  << std::setprecision(4) << std::endl;
    };

    print_row("Scalar (No Unroll) [baseline]", time_s_nounroll, time_s_nounroll);
    print_row("Scalar (4x Unroll)", time_s_unroll, time_s_nounroll);
    print_row("AVX-512 (No Unroll)", time_avx_nounroll, time_s_nounroll);
    print_row("AVX-512 (4x Unroll)", time_avx_unroll, time_s_nounroll);
    print_row("AVX-512 Opt (4x Unroll)", time_avx_opt, time_s_nounroll);
    print_row("JIT (original, 4x Unroll)", time_jit_orig, time_s_nounroll);
    print_row("JIT Opt (prefix-sum, 4x Unroll)", time_jit_opt, time_s_nounroll);

    std::cout << "\n--- JIT vs AVX-512 Comparison ---" << std::endl;
    std::cout << "AVX-512 Opt (4x Unroll):             " << time_avx_opt << " ms" << std::endl;
    std::cout << "JIT Opt (prefix-sum, 4x Unroll):     " << time_jit_opt << " ms" << std::endl;
    double gap_pct = (time_jit_opt - time_avx_opt) / time_avx_opt * 100.0;
    std::cout << "JIT vs AVX-512 gap:                  " << (gap_pct > 0 ? "+" : "") << gap_pct << "%" << std::endl;
    std::cout << "JIT Opt vs JIT Original speedup:     " << (time_jit_orig / time_jit_opt) << "x" << std::endl;

    // --- 前 10 个值比较 ---
    std::cout << "\n--- Value Comparison (First 10 Bytes) ---" << std::endl;
    std::cout << std::hex << std::setfill('0');
    std::cout << std::setw(w) << std::left << "Original Data" << ": ";
    for (int i = 0; i < 10; ++i) std::cout << "0x" << std::setw(2) << (int)original_uncompressed_data[i] << " ";
    std::cout << std::endl;
    std::cout << std::setw(w) << std::left << "JIT Opt Output" << ": ";
    for (int i = 0; i < 10; ++i) std::cout << "0x" << std::setw(2) << (int)output_jit_opt[i] << " ";
    std::cout << std::endl << std::dec;

    // 释放内存
    free(output_scalar_nounroll);
    free(output_scalar_unroll);
    free(output_avx512_nounroll);
    free(output_avx512_unroll);
    free(output_jit);
    free(output_avx512_opt);
    free(output_jit_opt);

    return 0;
}
