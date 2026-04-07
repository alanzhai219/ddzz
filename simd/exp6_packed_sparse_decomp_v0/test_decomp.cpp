#include <cstdint>  // For uint8_t, uint64_t
#include <cstdlib>  // For malloc, free
#include <cstring>  // For memset, memcmp
#include <iostream> // For std::cout

#include <immintrin.h>
#include <nmmintrin.h> // for _mm_popcnt_u64

// The function to test
void decompress_scalar_nounroll(uint8_t* scratch_buf,
                                const uint8_t* ptr_B,
                                const uint64_t* bitmask_ptr,
                                int blocks) {
    
    const uint8_t* current_src_ptr = ptr_B;
    const int chunks_per_block = 64; // 4096 / 64 = 64

    for (int block = 0; block < blocks; ++block) {
        size_t wei_offset = (size_t)block * 4096;
        const uint64_t* current_mask_ptr = bitmask_ptr + (block * chunks_per_block);

        // 循环步进改为 1 (cl++)
        for (int cl = 0; cl < chunks_per_block; cl++) {
            
            // --- 只保留 1 次循环体 ---
            uint64_t mask1 = current_mask_ptr[cl];
            uint8_t* dst1 = scratch_buf + wei_offset + cl * 64;
            size_t popcnt1 = 0;
            for (int i = 0; i < 64; ++i) {
                if ((mask1 >> i) & 1) {
                    dst1[i] = current_src_ptr[popcnt1++];
                } else {
                    dst1[i] = 0;
                }
            }
            current_src_ptr += popcnt1; // update src_ptr for next chunk
        }

        // --- 缓存行对齐逻辑 (不变) ---
        size_t offset = (size_t)current_src_ptr;
        size_t misalignment = offset & 0x3F; // 6 LSBs (63)
        if (misalignment != 0) {
            current_src_ptr += (64 - misalignment);
        }
    }
}

void decompress_avx512_nounroll(uint8_t* scratch_buf,
                                const uint8_t* ptr_B,
                                const uint64_t* bitmask_ptr,
                                int blocks) {
    
    const uint8_t* current_src_ptr = ptr_B;
    const int chunks_per_block = 64;

    for (int block = 0; block < blocks; ++block) {
        size_t wei_offset = (size_t)block * 4096;
        const uint64_t* current_mask_ptr = bitmask_ptr + (block * chunks_per_block);

        // 循环步进改为 1 (cl++)
        for (int cl = 0; cl < chunks_per_block; cl++) {
            
            // --- 只保留 1 次循环体 ---
            uint64_t mask1_u64 = current_mask_ptr[cl];
            __mmask64 mask1 = mask1_u64;
            
            __m512i zmm1 = _mm512_maskz_expandloadu_epi8(mask1, current_src_ptr);
            
            _mm512_storeu_si512((__m512i*)(scratch_buf + wei_offset + cl * 64), zmm1);
            
            current_src_ptr += _mm_popcnt_u64(mask1_u64);
        }

        // --- 缓存行对齐逻辑 (不变) ---
        size_t offset = (size_t)current_src_ptr;
        size_t misalignment = offset & 0x3F;
        if (misalignment != 0) {
            current_src_ptr += (64 - misalignment);
        }
    }
}

// Test function
void test_decompress_scalar_nounroll() {
    // Test parameters
    const int blocks = 4; // Single block for simplicity
    const size_t block_size = 4096; // Bytes per block
    const int chunks_per_block = 64;
    const int elements_per_chunk = 64;

    const size_t total_size = blocks * block_size;

    // Allocate buffers
    uint8_t* scratch_buf = (uint8_t*)malloc(total_size);
    if (!scratch_buf) {
        std::cerr << "Failed to allocate scratch_buf" << std::endl;
        return;
    }
    memset(scratch_buf, 0xFF, total_size); // Initialize to non-zero for verification

    // Prepare bitmask: For simplicity, use one chunk with a simple mask, and zero others
    uint64_t* bitmask_ptr = (uint64_t*)malloc(sizeof(uint64_t) * chunks_per_block * blocks);
    if (!bitmask_ptr) {
        std::cerr << "Failed to allocate bitmask_ptr" << std::endl;
        free(scratch_buf);
        return;
    }
    memset(bitmask_ptr, 0, sizeof(uint64_t) * chunks_per_block * blocks); // All zeros initially

    // Set a simple mask for the first chunk: bits 0,2,4 set (non-zero positions)
    // Mask: 0b...010101 (for positions 0,2,4)
    bitmask_ptr[0] = (1ULL << 0) | (1ULL << 2) | (1ULL << 4); // 3 non-zeros

    // Prepare compressed data (ptr_B): Only non-zero values, continuous
    // For the whole block, but since only first chunk has non-zeros, and assuming padding if needed
    const size_t compressed_size = 3 + 64; // 3 values + potential padding to 64-byte align
    uint8_t* ptr_B = (uint8_t*)malloc(compressed_size);
    if (!ptr_B) {
        std::cerr << "Failed to allocate ptr_B" << std::endl;
        free(scratch_buf);
        free(bitmask_ptr);
        return;
    }
    // Non-zero values: say 10, 20, 30
    ptr_B[0] = 10;
    ptr_B[1] = 20;
    ptr_B[2] = 30;
    // Fill rest with junk, but since alignment might skip, it's ok

    // Call the function
    decompress_scalar_nounroll(scratch_buf, ptr_B, bitmask_ptr, blocks);

    // Prepare expected output for verification
    uint8_t expected[total_size];
    memset(expected, 0, total_size); // All zeros by default
    // First chunk: positions 0,2,4 = 10,20,30
    expected[0] = 10;
    expected[2] = 20;
    expected[4] = 30;

    // Verify
    bool passed = true;
    if (memcmp(scratch_buf, expected, total_size) != 0) {
        passed = false;
        std::cerr << "Output mismatch!" << std::endl;
    }

    // Check alignment effect: After processing, current_src_ptr should be advanced by 3 + padding to next 64-byte boundary
    // But since blocks=1, we don't check further blocks

    if (passed) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }

    // Cleanup
    free(scratch_buf);
    free(bitmask_ptr);
    free(ptr_B);
}

int main() {
    test_decompress_scalar_nounroll();
    return 0;
}