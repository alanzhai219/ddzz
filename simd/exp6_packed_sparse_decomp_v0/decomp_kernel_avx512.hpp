// AVX-512 和 POPCNT 的头文件
#include <immintrin.h>
#include <nmmintrin.h> // for _mm_popcnt_u64

// -----------------------------------------------------------------
// ---- 3. AVX-512 Intrinsics 实现 - 不展开 ----
// -----------------------------------------------------------------
void decompress_avx512_nounroll(uint8_t* decomp_buf,
                                const uint8_t* compressed_buf,
                                const uint64_t* bitmask_ptr,
                                int blocks) {
    const uint8_t* current_src_ptr = compressed_buf;
    const int chunks_per_block = 64;

    for (int block = 0; block < blocks; ++block) {
        size_t wei_offset = (size_t)block * 4096;
        const uint64_t* current_mask_ptr = bitmask_ptr + (block * chunks_per_block);

        for (int cl = 0; cl < chunks_per_block; cl++) {
            uint64_t mask1_u64 = current_mask_ptr[cl];
            __mmask64 mask1 = mask1_u64;
            __m512i zmm1 = _mm512_maskz_expandloadu_epi8(mask1, current_src_ptr);
            _mm512_storeu_si512((__m512i*)(decomp_buf + wei_offset + cl * 64), zmm1);
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

// -----------------------------------------------------------------
// ---- 4. AVX-512 Intrinsics 实现 - 4x 展开 ----
// -----------------------------------------------------------------
void decompress_avx512(uint8_t* decomp_buf,
                       const uint8_t* compressed_buf,
                       const uint64_t* bitmask_ptr,
                       int blocks) {
    const uint8_t* current_src_ptr = compressed_buf;
    const int chunks_per_block = 64;

    for (int block = 0; block < blocks; ++block) {
        size_t wei_offset = (size_t)block * 4096;
        const uint64_t* current_mask_ptr = bitmask_ptr + (block * chunks_per_block);

        for (int cl = 0; cl < chunks_per_block; cl += 4) {
            // --- Unroll 1 ---
            uint64_t mask1_u64 = current_mask_ptr[cl + 0];
            __mmask64 mask1 = mask1_u64;
            __m512i zmm1 = _mm512_maskz_expandloadu_epi8(mask1, current_src_ptr);
            _mm512_storeu_si512((__m512i*)(decomp_buf + wei_offset + (cl + 0) * 64), zmm1);
            current_src_ptr += _mm_popcnt_u64(mask1_u64);

            // --- Unroll 2 ---
            uint64_t mask2_u64 = current_mask_ptr[cl + 1];
            __mmask64 mask2 = mask2_u64;
            __m512i zmm2 = _mm512_maskz_expandloadu_epi8(mask2, current_src_ptr);
            _mm512_storeu_si512((__m512i*)(decomp_buf + wei_offset + (cl + 1) * 64), zmm2);
            current_src_ptr += _mm_popcnt_u64(mask2_u64);

            // --- Unroll 3 ---
            uint64_t mask3_u64 = current_mask_ptr[cl + 2];
            __mmask64 mask3 = mask3_u64;
            __m512i zmm3 = _mm512_maskz_expandloadu_epi8(mask3, current_src_ptr);
            _mm512_storeu_si512((__m512i*)(decomp_buf + wei_offset + (cl + 2) * 64), zmm3);
            current_src_ptr += _mm_popcnt_u64(mask3_u64);

            // --- Unroll 4 ---
            uint64_t mask4_u64 = current_mask_ptr[cl + 3];
            __mmask64 mask4 = mask4_u64;
            __m512i zmm4 = _mm512_maskz_expandloadu_epi8(mask4, current_src_ptr);
            _mm512_storeu_si512((__m512i*)(decomp_buf + wei_offset + (cl + 3) * 64), zmm4);
            current_src_ptr += _mm_popcnt_u64(mask4_u64);
        }

        // --- 缓存行对齐逻辑 (不变) ---
        size_t offset = (size_t)current_src_ptr;
        size_t misalignment = offset & 0x3F;
        if (misalignment != 0) {
            current_src_ptr += (64 - misalignment);
        }
    }
}