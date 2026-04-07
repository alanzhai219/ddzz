// -----------------------------------------------------------------
// ---- 1. 纯 C++ (标量) 实现 - 不展开 ----
// -----------------------------------------------------------------
void decompress_scalar_nounroll(uint8_t* decomp_buf,
                                const uint8_t* compressed_buf,
                                const uint64_t* bitmask_ptr,
                                int blocks) {
    const uint8_t* current_src_ptr = compressed_buf;
    const int chunks_per_block = 64;
    const int bytes_per_chunk = 64;

    for (int block = 0; block < blocks; ++block) {
        size_t wei_offset = (size_t)block * 4096;
        const uint64_t* current_block_mask_ptr = bitmask_ptr + (block * chunks_per_block);

        for (int cl = 0; cl < chunks_per_block; cl++) {
            uint64_t mask1 = current_block_mask_ptr[cl];
            uint8_t* dst1 = decomp_buf + wei_offset + bytes_per_chunk * cl;
            size_t popcnt1 = 0;
            for (int i = 0; i < bytes_per_chunk; ++i) {
                if ((mask1 >> i) & 1) {
                    dst1[i] = current_src_ptr[popcnt1++];
                } else {
                    dst1[i] = 0;
                }
            }
            current_src_ptr += popcnt1;
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
// ---- 2. 纯 C++ (标量) 实现 - 4x 展开 ----
// -----------------------------------------------------------------
void decompress_scalar(uint8_t* decomp_buf,
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
            uint64_t mask1 = current_mask_ptr[cl + 0];
            uint8_t* dst1 = decomp_buf + wei_offset + (cl + 0) * 64;
            size_t popcnt1 = 0;
            for (int i = 0; i < 64; ++i) {
                if ((mask1 >> i) & 1) {
                    dst1[i] = current_src_ptr[popcnt1++];
                } else {
                    dst1[i] = 0;
                }
            }
            current_src_ptr += popcnt1;

            // --- Unroll 2 ---
            uint64_t mask2 = current_mask_ptr[cl + 1];
            uint8_t* dst2 = decomp_buf + wei_offset + (cl + 1) * 64;
            size_t popcnt2 = 0;
            for (int i = 0; i < 64; ++i) {
                if ((mask2 >> i) & 1) {
                    dst2[i] = current_src_ptr[popcnt2++];
                } else {
                    dst2[i] = 0;
                }
            }
            current_src_ptr += popcnt2;
            
            // --- Unroll 3 ---
            uint64_t mask3 = current_mask_ptr[cl + 2];
            uint8_t* dst3 = decomp_buf + wei_offset + (cl + 2) * 64;
            size_t popcnt3 = 0;
            for (int i = 0; i < 64; ++i) {
                if ((mask3 >> i) & 1) {
                    dst3[i] = current_src_ptr[popcnt3++];
                } else {
                    dst3[i] = 0;
                }
            }
            current_src_ptr += popcnt3;

            // --- Unroll 4 ---
            uint64_t mask4 = current_mask_ptr[cl + 3];
            uint8_t* dst4 = decomp_buf + wei_offset + (cl + 3) * 64;
            size_t popcnt4 = 0;
            for (int i = 0; i < 64; ++i) {
                if ((mask4 >> i) & 1) {
                    dst4[i] = current_src_ptr[popcnt4++];
                } else {
                    dst4[i] = 0;
                }
            }
            current_src_ptr += popcnt4;
        }

        // --- 缓存行对齐逻辑 (不变) ---
        size_t offset = (size_t)current_src_ptr;
        size_t misalignment = offset & 0x3F;
        if (misalignment != 0) {
            current_src_ptr += (64 - misalignment);
        }
    }
}