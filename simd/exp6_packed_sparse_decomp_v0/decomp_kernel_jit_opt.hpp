#include <stddef.h>
#include "xbyak/xbyak.h"

struct jit_decomp_params_t_opt {
    const void* compressed_buf;
    const void* bitmask_ptr;
    void* decomp_buf;
};

#define GET_OFF_OPT(field) (offsetof(jit_decomp_params_t_opt, field))

// -----------------------------------------------------------------
// Optimized JIT Decompression Kernel
//
// Key optimization: prefix-sum addressing for vpexpandb
//
// PROBLEM (previous version):
//   Each vpexpandb depends on the previous chunk's popcnt+add result
//   to know the source address. This creates a serial dependency chain:
//     vpexpandb [src] → popcnt → add src → vpexpandb [src] → ...
//   Critical path: 4 × (3cy popcnt + 1cy add) = 16 cycles
//
// SOLUTION (this version):
//   1. Compute all 4 popcnts independently (can run in parallel on OoO CPU)
//   2. Use prefix-sum to compute all 4 offsets from the current src pointer
//   3. Issue all 4 vpexpandb with [src + offset] addressing
//   Critical path: 3cy popcnt + 3cy prefix-sum = 6 cycles
//   This is a 2.67x improvement in address-computation latency.
// -----------------------------------------------------------------

class jit_decompress_kernel_t_opt : public Xbyak::CodeGenerator {
public:
    // --- Register assignments ---
    Xbyak::Reg64 param1 = rdi;               // Linux ABI: 1st arg register

    // Persistent pointers (survive across iterations)
    Xbyak::Reg64 compressed_ptr = r8;         // base of compressed buffer (for alignment calc)
    Xbyak::Reg64 reg_bitmask = r9;            // current bitmask pointer (advances per block)
    Xbyak::Reg64 reg_decomp_dst = r10;        // current decomp destination (advances per block)
    Xbyak::Reg64 reg_src = r11;               // current compressed src pointer (advances per group)

    // Callee-saved registers (pushed/popped)
    Xbyak::Reg64 reg_block_cnt = r12;         // block loop counter
    Xbyak::Reg64 reg_off1 = r13;              // prefix-sum offset 1
    Xbyak::Reg64 reg_dst_local = r14;         // local destination pointer (per block)
    Xbyak::Reg64 reg_align_tmp = r15;         // alignment temp

    // Volatile registers for mask values / popcnt results
    Xbyak::Reg64 reg_m1 = rax;
    Xbyak::Reg64 reg_m2 = rcx;
    Xbyak::Reg64 reg_m3 = rdx;
    Xbyak::Reg64 reg_m4 = rsi;

    // Opmask registers
    Xbyak::Opmask kmask1 = k1;
    Xbyak::Opmask kmask2 = k2;
    Xbyak::Opmask kmask3 = k3;
    Xbyak::Opmask kmask4 = k4;

    int blocks_;

    jit_decompress_kernel_t_opt(int blocks) : Xbyak::CodeGenerator(4096 * 4096), blocks_(blocks) {
        generate();
    }

    void generate() {
        // Save callee-saved registers
        push(r12);
        push(r13);
        push(r14);
        push(r15);

        // Load parameters from struct
        mov(compressed_ptr, ptr[param1 + GET_OFF_OPT(compressed_buf)]);
        mov(reg_bitmask, ptr[param1 + GET_OFF_OPT(bitmask_ptr)]);
        mov(reg_decomp_dst, ptr[param1 + GET_OFF_OPT(decomp_buf)]);
        lea(reg_src, ptr[compressed_ptr]);

        // Block loop setup
        Xbyak::Label loop_blocks;
        Xbyak::Label end_blocks;

        mov(reg_block_cnt, blocks_);
        test(reg_block_cnt, reg_block_cnt);
        jz(end_blocks, T_NEAR);

        align(64);  // Align loop entry to cache line for better I-cache utilization
        L(loop_blocks);
        mov(reg_dst_local, reg_decomp_dst);

        // 16 iterations × 4 chunks/iteration = 64 chunks per block
        for (int i = 0; i < 16; ++i) {
            const int bm_off = i * 32;   // 4 × sizeof(uint64_t) = 32 bytes
            const int dst_off = i * 256; // 4 × 64 bytes = 256 bytes

            // ============================================================
            // Phase 1: Load all 4 bitmasks and set k-masks
            // ============================================================
            mov(reg_m1, ptr[reg_bitmask + bm_off + 0]);
            kmovq(kmask1, reg_m1);
            mov(reg_m2, ptr[reg_bitmask + bm_off + 8]);
            kmovq(kmask2, reg_m2);
            mov(reg_m3, ptr[reg_bitmask + bm_off + 16]);
            kmovq(kmask3, reg_m3);
            mov(reg_m4, ptr[reg_bitmask + bm_off + 24]);
            kmovq(kmask4, reg_m4);

            // ============================================================
            // Phase 2: Compute all 4 popcnts (INDEPENDENT - OoO parallel)
            //   On modern Intel, popcnt has 3-cycle latency, 1/cycle throughput.
            //   All 4 can issue in consecutive cycles with no dependency.
            // ============================================================
            popcnt(reg_m1, reg_m1);  // p1 = popcnt(mask1)
            popcnt(reg_m2, reg_m2);  // p2 = popcnt(mask2)
            popcnt(reg_m3, reg_m3);  // p3 = popcnt(mask3)
            popcnt(reg_m4, reg_m4);  // p4 = popcnt(mask4)

            // ============================================================
            // Phase 3: Prefix sum for offsets (3 sequential adds)
            //   off0 = 0          (use reg_src directly)
            //   off1 = p1         (stored in reg_off1)
            //   off2 = p1+p2      (stored in reg_m2)
            //   off3 = p1+p2+p3   (stored in reg_m3)
            //   total = p1+p2+p3+p4 (stored in reg_m4)
            // ============================================================
            mov(reg_off1, reg_m1);      // off1 = p1
            add(reg_m2, reg_off1);      // off2 = p1 + p2
            add(reg_m3, reg_m2);        // off3 = p1 + p2 + p3
            add(reg_m4, reg_m3);        // total = p1 + p2 + p3 + p4

            // ============================================================
            // Phase 4: Fused masked expand-load with pre-computed offsets
            //   All 4 addresses are known after the prefix sum completes,
            //   so the CPU can issue all 4 loads without waiting for
            //   sequential pointer updates.
            // ============================================================
            vpexpandb(zmm0 | kmask1 | T_z, ptr[reg_src]);              // src + 0
            vpexpandb(zmm1 | kmask2 | T_z, ptr[reg_src + reg_off1]);   // src + off1
            vpexpandb(zmm2 | kmask3 | T_z, ptr[reg_src + reg_m2]);     // src + off2
            vpexpandb(zmm3 | kmask4 | T_z, ptr[reg_src + reg_m3]);     // src + off3

            // ============================================================
            // Phase 5: Advance src pointer by total (single add)
            // ============================================================
            add(reg_src, reg_m4);

            // ============================================================
            // Phase 6: Store decompressed results
            //   Deferred after all expands to allow OoO overlap with
            //   next iteration's mask loads.
            // ============================================================
            vmovdqu8(ptr[reg_dst_local + dst_off + 0], zmm0);
            vmovdqu8(ptr[reg_dst_local + dst_off + 64], zmm1);
            vmovdqu8(ptr[reg_dst_local + dst_off + 128], zmm2);
            vmovdqu8(ptr[reg_dst_local + dst_off + 192], zmm3);
        }

        // Advance bitmask pointer: 64 chunks × 8 bytes/chunk = 512 bytes
        add(reg_bitmask, 512);

        // Align compressed source to cacheline boundary (relative to base)
        // padding = (-(src - base)) & 0x3F
        mov(reg_align_tmp, reg_src);
        sub(reg_align_tmp, compressed_ptr);
        neg(reg_align_tmp);
        and_(reg_align_tmp, 0x3f);
        add(reg_src, reg_align_tmp);

        // Advance destination: 4096 bytes per block
        add(reg_decomp_dst, 4096);
        dec(reg_block_cnt);
        jnz(loop_blocks, T_NEAR);

        L(end_blocks);
        // Restore callee-saved registers
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        ret();
    }
};
