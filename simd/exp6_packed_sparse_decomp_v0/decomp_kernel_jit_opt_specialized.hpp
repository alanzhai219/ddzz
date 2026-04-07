#include <xbyak/xbyak.h>
// =================================================================
// Mask-Specialized JIT Kernel
// =================================================================
//
// This kernel receives bitmask data at JIT-COMPILE TIME and pre-computes
// every address offset. At runtime, it only needs:
//   - compressed_buf pointer (base for all loads)
//   - decomp_buf pointer (base for all stores)
//
// ELIMINATED at runtime (vs generic kernels):
//   - ALL popcnt instructions (offsets are immediate constants)
//   - ALL prefix-sum / pointer-advance adds
//   - ALL bitmask memory loads (masks embedded as 64-bit immediates)
//   - ALL loop overhead (fully unrolled)
//   - vpexpandb for zero-mask chunks (replaced with zero-store)
//   - vpexpandb for dense chunks (replaced with plain copy)

struct jit_specialized_params_t {
    const void* compressed_buf;  // offset 0
    void* decomp_buf;            // offset 8
};

class jit_decompress_specialized_t : public Xbyak::CodeGenerator {
public:
    Xbyak::Reg64 param1 = rdi;
    Xbyak::Reg64 reg_src = r8;    // compressed base, NEVER modified at runtime
    Xbyak::Reg64 reg_dst = r9;    // decomp base, NEVER modified at runtime
    Xbyak::Reg64 reg_tmp = rax;   // temp for mask immediates

    int blocks_;
    const uint64_t* bitmask_;

    // Stats populated during code generation
    int zero_chunks_ = 0;
    int full_chunks_ = 0;
    int partial_chunks_ = 0;
    size_t code_size_ = 0;

    jit_decompress_specialized_t(int blocks, const uint64_t* bitmask)
        : Xbyak::CodeGenerator(4096 * 256),
          blocks_(blocks), bitmask_(bitmask) {
        generate();
    }

    void generate() {
        // Load data pointers from params struct
        mov(reg_src, ptr[param1 + offsetof(jit_specialized_params_t, compressed_buf)]);
        mov(reg_dst, ptr[param1 + offsetof(jit_specialized_params_t, decomp_buf)]);

        // Track compressed source offset entirely at JIT-compile time.
        // This is the key optimization: src_off is a C++ variable computed
        // during code generation, NOT a runtime register. All addresses
        // become [reg_src + IMMEDIATE].
        int src_off = 0;

        for (int block = 0; block < blocks_; ++block) {
            int bm_base = block * 64;
            int dst_base = block * 4096;

            for (int chunk = 0; chunk < 64; ++chunk) {
                uint64_t mask = bitmask_[bm_base + chunk];
                // it is a C++ api, which means it can be computed at JIT-compile time, NOT runtime!
                // so it becomes an immediate constant in the generated code, NOT a register variable.
                int popcnt = __builtin_popcountll(mask);
                int dst_off = dst_base + chunk * 64;

                // Rotate ZMM and K registers for ILP
                Xbyak::Zmm zr = Xbyak::Zmm(chunk & 3);
                Xbyak::Opmask kr = Xbyak::Opmask(1 + (chunk & 3));

                if (mask == 0) {
                    // ---- ZERO CHUNK: no data to load ----
                    // 2 instructions instead of 4 (skip mov+kmovq+vpexpandb)
                    vpxord(zr, zr, zr);
                    zero_chunks_++;
                } else if (mask == 0xFFFFFFFFFFFFFFFFULL) {
                    // ---- DENSE CHUNK: plain copy, no expand needed ----
                    // 2 instructions instead of 4 (skip mov+kmovq, use vmovdqu8)
                    vmovdqu8(zr, ptr[reg_src + src_off]);
                    full_chunks_++;
                } else {
                    // ---- PARTIAL CHUNK: embed mask, use static offset ----
                    // 4 instructions, but NO popcnt/add on critical path
                    mov(reg_tmp, mask);        // 64-bit immediate
                    kmovq(kr, reg_tmp);
                    vpexpandb(zr | kr | T_z, ptr[reg_src + src_off]);
                    partial_chunks_++;
                }
                vmovdqu8(ptr[reg_dst + dst_off], zr);

                src_off += popcnt;  // Advance at JIT-compile time, NOT runtime!
            }

            // Inter-block alignment (also pre-computed at JIT-compile time)
            int misalign = src_off & 0x3F;
            if (misalign) src_off += (64 - misalign);
        }

        ret();
        code_size_ = getSize();
    }
};