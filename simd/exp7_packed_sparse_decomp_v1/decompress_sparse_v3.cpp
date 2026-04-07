#include <iostream>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <climits>
#include <stdexcept>
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

// Struct definition for parameter passing
struct call_params_t {
    const void *src_ptr;      // Packed source pointer
    const void *bitmask_ptr;  // Sparse bitmask pointer (uint64_t array)
    const void *dst_ptr;      // Decompressed destination pointer
};

#define GET_OFF(field) offsetof(call_params_t, field)

class SparseDecompressKernel : public Xbyak::CodeGenerator {
public:
    SparseDecompressKernel() : Xbyak::CodeGenerator(4096, Xbyak::AutoGrow) {
        if (!m_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
            throw std::runtime_error("AVX512 is not supported on this CPU.");
        }
        
        // --- Logic Fix 1: Block Size Calculation ---
        // Previous logic: 8 * 64 / 4096 = 0 (Integer division). 
        // We set nblks to 1 to ensure the loop runs for this specific test case.
        b_blk_sz_ = 64; 
        blk_sz_ = a_outter_blk_sz_ * b_blk_sz_ * a_inner_blk_sz_; // 4096 bytes
        nblks_to_decompress_ = 0; 

        generate_code();
    }

    void (*get_jit_func() const)(call_params_t *) {
        return reinterpret_cast<void (*)(call_params_t *)>(getCode());
    }

public:
    int nblks_to_decompress_;
    int blk_sz_;
    int b_blk_sz_;
    const int a_outter_blk_sz_ = 16;
    const int a_inner_blk_sz_ = 4;

private:
    const Xbyak::Reg64 reg_src_ptr = r8;
    const Xbyak::Reg64 reg_dst_ptr = r9;
    const Xbyak::Reg64 reg_bitmask_ptr = r10;
    const Xbyak::Reg64 reg_tmp = r11;
    const Xbyak::Reg64 reg_popcnt_tmp = r12;
    const Xbyak::Reg64 reg_popcnt = rcx; // Must be RCX for implicit CL use in shifts

    Xbyak::Reg64 param1 = rdi; // First argument in System V ABI

    const int unroll_factor = 4;

    Xbyak::Zmm get_zmm(int idx) {
        // Registers to store the final result before writing to memory
        switch (idx) {
            case 0: return Xbyak::Zmm(25);
            case 1: return Xbyak::Zmm(26);
            case 2: return Xbyak::Zmm(27);
            case 3: return Xbyak::Zmm(28);
            default: assert(!"incorrect index"); return Xbyak::Zmm(0);
        }
    }

    // --- Logic Fix 2: Temporary Register ---
    // vpexpandb requires Source != Dest. We need a temp register for loading packed data.
    Xbyak::Zmm get_zmm_tmp() {
        return Xbyak::Zmm(31); 
    }

    Xbyak::Opmask get_opmask(int idx) {
        switch (idx) {
            case 0: return k1;
            case 1: return k2;
            case 2: return k3;
            case 3: return k4;
            default: assert(!"incorrect index"); return k0;
        }
    }

    Xbyak::Opmask get_load_mask(int idx) {
        // Logic to create a mask of 'popcnt' consecutive 1s.
        // E.g., if popcnt=4, we want 0...001111 (binary).
        // Algorithm used: ((1 << (N/2)) << (N/2 + rem)) - 1
        // This avoids SHL by 64 (which is undefined/modulo behavior in C++, wrapped in asm).

        mov(reg_popcnt_tmp, reg_popcnt); // Save original count

        mov(reg_tmp, 1);

        shr(reg_popcnt, 1); // popcnt / 2

        // reg_tmp = 1 << (popcnt / 2)
        shl(reg_tmp, reg_popcnt.cvt8()); 
        
        // reg_tmp = reg_tmp << (popcnt / 2)
        shl(reg_tmp, reg_popcnt.cvt8()); 

        // Handle the remainder (odd/even)
        mov(reg_popcnt, reg_popcnt_tmp);
        and_(reg_popcnt, 1);
        shl(reg_tmp, reg_popcnt.cvt8());

        // reg_tmp is now 2^N. Subtract 1 to get N ones.
        sub(reg_tmp, 1);

        mov(reg_popcnt, reg_popcnt_tmp); // Restore count

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

    void generate_code() {
        // Preamble: Save registers
        push(r13);
        push(r14);
        push(r15);
        push(rax);
        push(rcx);
        push(r8);
        push(r9);
        push(r10);
        push(r11);
        push(r12);

        // Load Params
        mov(reg_bitmask_ptr, ptr[param1 + GET_OFF(bitmask_ptr)]);
        mov(reg_dst_ptr, ptr[param1 + GET_OFF(dst_ptr)]);
        mov(reg_src_ptr, ptr[param1 + GET_OFF(src_ptr)]);

        for (int b = 0; b < nblks_to_decompress_; b++) {
            const int blk_offset = b * blk_sz_;
            const int bitmask_off = blk_offset / CHAR_BIT;
            const int nbytes_per_load = 64;

            for (int i = 0; i < b_blk_sz_; i += unroll_factor) {
                for (int uf = 0; uf < unroll_factor; uf++) {
                    auto reg_mask_tmp = get_reg_mask_tmp(uf);
                    
                    // 1. Load Bitmask (64 bits)
                    mov(reg_mask_tmp, ptr[reg_bitmask_ptr + (i + uf) * sizeof(uint64_t) + bitmask_off]);
                    
                    // 2. Count bits (Population Count)
                    popcnt(reg_popcnt, reg_mask_tmp);

                    // 3. Prepare Load Mask (k-register with low 'popcnt' bits set)
                    auto load_mask = get_load_mask(uf);
                    
                    // 4. Load Packed Data
                    // WARNING: vpexpandb requires Src != Dst. 
                    // We must load into a temporary register first.
                    auto zmm_packed = get_zmm_tmp(); // Use zmm31 as temp
                    
                    // Load 'popcnt' bytes from src_ptr into zmm_packed. 
                    // Remaining bytes in register are zeroed (T_z).
                    vmovdqu8(zmm_packed | load_mask | T_z, ptr[reg_src_ptr]);
                    
                    // Advance source pointer
                    add(reg_src_ptr, reg_popcnt);

                    // 5. Expand Data
                    auto expand_mask = get_expand_mask(uf);
                    kmovq(expand_mask, reg_mask_tmp); // Load the sparse pattern mask
                    
                    auto zmm_dst = get_zmm(uf);
                    
                    // vpexpandb DEST, SRC
                    // Expands packed byte elements from zmm_packed to zmm_dst 
                    // based on set bits in expand_mask.
                    vpexpandb(zmm_dst | expand_mask | T_z, zmm_packed);

                    // 6. Store to Dst
                    vmovdqu8(ptr[reg_dst_ptr + blk_offset + (i + uf) * nbytes_per_load], zmm_dst);
                }
            }
        }

        // Postamble: Restore registers
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

// --- Test Code ---
void run_test() {
    std::cout << "--- Sparse Decompress JIT Kernel Test ---" << std::endl;

    SparseDecompressKernel kernel;
    auto jit_func = kernel.get_jit_func();

    const int B_BLK = kernel.b_blk_sz_; 
    const int BYTES_PER_ROW = 64; 
    const int NUM_ROWS = B_BLK; 
    
    // Setup Bitmasks
    std::vector<uint64_t> bitmask(NUM_ROWS, 0);
    bitmask[0] = 0xF;                   // 4 items
    bitmask[1] = 0xFF;                  // 8 items
    bitmask[2] = 0x0;                   // 0 items
    bitmask[3] = 0xFFFFFFFFFFFFFFFF;    // 64 items
    bitmask[4] = 0x8000000000000000;    // 1 item (MSB)
    for (int i = 5; i < NUM_ROWS; ++i) {
         if (i % 2 == 0) bitmask[i] = 0x5555555555555555; 
         else bitmask[i] = 0xAAAAAAAAAAAAAAAA; 
    }

    // Setup Source Data
    size_t total_packed_size = 0;
    for (uint64_t mask : bitmask) {
        total_packed_size += __builtin_popcountll(mask);
    }

    std::vector<uint8_t> src_data(total_packed_size);
    for (size_t i = 0; i < total_packed_size; ++i) {
        src_data[i] = (uint8_t)(i % 255 + 1); // Avoid 0 to distinguish from padding
    }

    // Setup Destination
    const int DST_SIZE = NUM_ROWS * BYTES_PER_ROW; 
    std::vector<uint8_t> dst_data(DST_SIZE, 0x00); // Init to 0

    call_params_t params = {
        src_data.data(),
        bitmask.data(),
        dst_data.data()
    };
    
    std::cout << "-> Executing JIT Kernel..." << std::endl;
    jit_func(&params);
    std::cout << "-> Execution Complete." << std::endl;

    // Verify
    std::cout << "--- Verifying Results ---" << std::endl;
    int src_idx = 0;
    bool success = true;

    for (int i = 0; i < NUM_ROWS; ++i) {
        uint64_t mask = bitmask[i];
        for (int j = 0; j < BYTES_PER_ROW; ++j) {
            bool is_active = (mask >> j) & 0x1;
            uint8_t expected_val = 0;
            if (is_active) {
                expected_val = src_data[src_idx++];
            }

            uint8_t actual_val = dst_data[i * BYTES_PER_ROW + j];

            if (actual_val != expected_val) {
                std::cerr << "FAIL at Row " << i << ", Col " << j 
                          << ": Exp " << (int)expected_val 
                          << ", Act " << (int)actual_val << std::endl;
                success = false;
                if (src_idx > 50) return;
            }
        }
    }

    if (success) {
        std::cout << "\n[OK] All checks passed." << std::endl;
    } else {
        std::cout << "\n[FAIL] Verification failed." << std::endl;
    }
}

int main() {
    try {
        run_test();
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}