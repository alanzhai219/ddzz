#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>
#include <stddef.h> // For offsetof

#include "xbyak/xbyak.h"

enum cpu_isa_t {
    avx2,
    avx512
};

namespace utils {

// template <bool, typename, bool, typename, typename>
// struct conditional3 {}; // NOLINT(readability-identifier-naming)
// template <typename T, typename FT, typename FF>
// struct conditional3<true, T, false, FT, FF> {
//     using type = T;
// };
// template <typename T, typename FT, typename FF>
// struct conditional3<false, T, true, FT, FF> {
//     using type = FT;
// };
// template <typename T, typename FT, typename FF>
// struct conditional3<false, T, false, FT, FF> {
//     using type = FF;
// };
// 
// template <typename Vmm>
// struct vreg_traits_t {};
// 
// template <>
// struct vreg_traits_t<Xbyak::Zmm> {
//     using Vmm_lower_t = Xbyak::Ymm;
//     static constexpr size_t vlen = 64;
// };
// 
// template <>
// struct vreg_traits_t<Xbyak::Ymm> {
//     using Vmm_lower_t = Xbyak::Xmm;
//     static constexpr size_t vlen = 32;
// };
// 
// template <>
// struct vreg_traits_t<Xbyak::Xmm> {
//     using Vmm_lower_t = Xbyak::Xmm;
//     static constexpr size_t vlen = 16;
// };

// constexpr Xbyak::Operand::Code abi_param_regs[] = {
//     Xbyak::Operand::RDI,
//     Xbyak::Operand::RSI,
//     Xbyak::Operand::RDX,
//     Xbyak::Operand::RCX,
//     Xbyak::Operand::R8,
//     Xbyak::Operand::R9
// };

} // namespace utils

struct jit_matmul_small_config_params {
    size_t M = 0UL;
    size_t K = 0UL;
    size_t N = 0UL;
};

struct jit_matmul_small_call_args {
    const void* input1; // Matrix A (M x K)
    const void* input2; // Matrix B (K x N)
    void* output;       // Matrix C (M x N)
    size_t B;           // Batch size
};

#define GET_OFF(field) offsetof(jit_matmul_small_call_args, field)

template <cpu_isa_t isa>
class jit_uni_matmul_small_kernel_f32 : public Xbyak::CodeGenerator {
public:
    jit_uni_matmul_small_kernel_f32(const jit_matmul_small_config_params& jcp)
        : _jcp(jcp) {}
    
    // FIX: Removed the 'override' keyword
    void generate() {
        preamble();

        // 从第一个参数（RDI）加载结构体成员
        mov(reg_input1,      ptr[reg_params + GET_OFF(input1)]);
        mov(reg_input2,      ptr[reg_params + GET_OFF(input2)]);
        mov(reg_out,         ptr[reg_params + GET_OFF(output)]);
        mov(reg_work_amount, ptr[reg_params + GET_OFF(B)]); // B 是循环次数

        Xbyak::Label loop_label;
        Xbyak::Label loop_end_label;
        L(loop_label);
        {
            cmp(reg_work_amount, 0); 
            je(loop_end_label, T_NEAR);

            // load mat A to 4 vmm registers
            for (size_t m = 0; m < _jcp.M; m++) {
                for (size_t k = 0; k < _jcp.K; k++) {
                    uni_vmovss(vmm_input1[m * _jcp.K + k], ptr[reg_input1]);
                    add(reg_input1, sizeof(float));
                }
            }
            // load mat B to 4 vmm registers
            for (size_t k = 0; k < _jcp.K; k++) {
                for (size_t n = 0; n < _jcp.N; n++) {
                    uni_vmovss(vmm_input2[k * _jcp.N + n], ptr[reg_input2]);
                    add(reg_input2, sizeof(float));
                }
            }

            // zero output registers
            for (size_t m = 0; m < _jcp.M; m++) {
                for (size_t n = 0; n < _jcp.N; n++) {
                    uni_vpxor(vmm_output[m * _jcp.N + n], vmm_output[m * _jcp.N + n], vmm_output[m * _jcp.N + n]);
                }
            }
            
            // C[m,n] += A[m,k] * B[k,n]
            for (size_t k = 0; k < _jcp.K; k++) {
                for (size_t m = 0; m < _jcp.M; m++) {
                    for (size_t n = 0; n < _jcp.N; n++) {
                        uni_vfmadd231ps(vmm_output[m * _jcp.N + n], vmm_input1[m * _jcp.K + k], vmm_input2[k * _jcp.N + n]);
                    }
                }
            }

            // store
            for (size_t m = 0; m < _jcp.M; m++) {
                for (size_t n = 0; n < _jcp.N; n++) {
                    uni_vmovss(ptr[reg_out], vmm_output[m * _jcp.N + n]);
                    add(reg_out, sizeof(float));
                }
            }

            sub(reg_work_amount, 1); // B--
            jmp(loop_label, T_NEAR);
        }
        L(loop_end_label);

        postamble();
    }

private:
    void preamble() {
        push(rbp);
        mov(rbp, rsp);
    }

    void postamble() {
        mov(rsp, rbp);
        pop(rbp);
        ret();
    }
    // using Vmm = typename utils::conditional3<isa == cpu_isa_t::sse41,
    //                                          Xbyak::Xmm,
    //                                          isa == cpu_isa_t::avx2,
    //                                          Xbyak::Ymm,
    //                                          Xbyak::Zmm>::type;
    using Vmm = typename std::conditional<isa == cpu_isa_t::avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    // const int vlen = utils::vreg_traits_t<Vmm>::vlen; 

    // 寄存器分配
    Xbyak::Reg64 reg_input1 = r8;
    Xbyak::Reg64 reg_input2 = r9;
    Xbyak::Reg64 reg_out = r10;
    Xbyak::Reg64 reg_work_amount = r11;
    // Xbyak::Reg64 reg_params = Xbyak::Reg64(utils::abi_param_regs[0]); // RDI
    Xbyak::Reg64 reg_params = rdi;

    // 向量寄存器分配
    Vmm vmm_input1[4] = {Vmm(0), Vmm(1), Vmm(2), Vmm(3)};
    Vmm vmm_input2[4] = {Vmm(4), Vmm(5), Vmm(6), Vmm(7)};
    Vmm vmm_output[4] = {Vmm(8), Vmm(9), Vmm(10), Vmm(11)};

    // config
    jit_matmul_small_config_params _jcp;

    // --- 辅助函数 ---
    void uni_vmovss(const Xbyak::Address &addr, const Xbyak::Xmm &x) {
        if (isa == avx2 || isa == avx512) {
            vmovss(addr, x);
        } else {
            movss(addr, x);
        }
    }
    void uni_vmovss(const Xbyak::Xmm &x, const Xbyak::Address &addr) {
        if (isa == avx2 || isa == avx512) {
            vmovss(x, addr);
        } else {
            movss(x, addr);
        }
    }
    void uni_vmovss(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2) {
        if (isa == avx2 || isa == avx512) {
            vmovss(x1, x1, x2);
        } else {
            movss(x1, x2);
        }
    }
    void uni_vmovss(const Xbyak::Address &addr, const Xbyak::Ymm &x) {
        vmovss(addr, Xbyak::Xmm(x.getIdx()));
    }
    void uni_vmovss(const Xbyak::Ymm &x, const Xbyak::Address &addr) {
        vmovss(Xbyak::Xmm(x.getIdx()), addr);
    }
    void uni_vmovss(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2) {
        vmovss(Xbyak::Xmm(x1.getIdx()), Xbyak::Xmm(x2.getIdx()));
    }

    // 
    void uni_vpxor(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        if (isa == avx512) {
            vpxord(x1, x2, op);
        } else if (isa == avx2) {
            vpxor(x1, x2, op);
        } else {
            if (!x1.isEqualIfNotInherited(x2)) movdqa(x1, x2);
            pxor(x1, op);
        }
    }
    void uni_vpxor(const Xbyak::Ymm &x1, const Xbyak::Ymm &x2,
            const Xbyak::Operand &op) {
        if (isa == avx512) {
            vpxord(x1, x2, op);
        } else if (isa == avx2) {
            vpxor(x1, x2, op);
        } else {
            vxorps(x1, x2, op);
        }
    }
    void uni_vpxor(const Xbyak::Zmm &x1, const Xbyak::Zmm &x2,
            const Xbyak::Operand &op) {
        vpxord(x1, x2, op);
    }

    void uni_vfmadd231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op, const Xbyak::Xmm &buf) {
        if (isa == avx2 || isa == avx512) {
            vfmadd231ps(x1, x2, op);
        // } else if (isa == avx) {
        //     assert(buf.getIdx() != x1.getIdx());
        //     vmulps(buf, x2, op);
        //     vaddps(x1, x1, buf);
        } else {
            assert(buf.getIdx() != x1.getIdx());
            if (x2.getIdx() != buf.getIdx()) movups(buf, x2);
            mulps(buf, op);
            addps(x1, buf);
        }
    }

    void uni_vfmadd231ps(const Xbyak::Xmm &x1, const Xbyak::Xmm &x2,
            const Xbyak::Operand &op) {
        uni_vfmadd231ps(x1, x2, op, x2);
    }
};

// --- 补充：main 函数用于测试 ---

// FIX: Changed to accept a pointer (slice)
void print_matrix_slice(const std::string& name, const float* matrix, size_t rows, size_t cols) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            std::cout << std::setw(5) << std::fixed << std::setprecision(1) << matrix[r * cols + c] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// FIX: Changed to accept pointers (slices)
void cpp_matmul_slice(const float* A, const float* B, float* C,
                      size_t M, size_t K, size_t N) {
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    // 1. 配置
    jit_matmul_small_config_params jcp;
    jcp.M = 2;
    jcp.K = 2;
    jcp.N = 2;

    const size_t batch_size = 2;
    const size_t mk_size = jcp.M * jcp.K;
    const size_t kn_size = jcp.K * jcp.N;
    const size_t mn_size = jcp.M * jcp.N;


    // 2. 实例化 JIT 内核
    jit_uni_matmul_small_kernel_f32<cpu_isa_t::avx2> kernel(jcp);
    kernel.generate();
    
    auto jit_func_ptr = kernel.getCode<void (*)(jit_matmul_small_call_args*)>();

    // 3. 准备数据
    std::vector<float> A_data = {
        // Batch 0
        1.0f, 2.0f, // A[0,0], A[0,1]
        3.0f, 4.0f, // A[1,0], A[1,1]
        // Batch 1
        5.0f, 6.0f,
        7.0f, 8.0f
    };
    
    std::vector<float> B_data = {
        // Batch 0
        5.0f, 6.0f, // B[0,0], B[0,1]
        7.0f, 8.0f, // B[1,0], B[1,1]
        // Batch 1
        1.0f, 0.0f,
        0.0f, 1.0f
    };

    std::vector<float> C_data(batch_size * jcp.M * jcp.N, 0.0f);
    std::vector<float> C_expected(batch_size * jcp.M * jcp.N, 0.0f);

    // 4. 准备 JIT 函数的参数
    jit_matmul_small_call_args args;
    args.input1 = A_data.data();
    args.input2 = B_data.data();
    args.output = C_data.data();
    args.B = batch_size; 

    // 5. 执行 JIT 代码
    std::cout << "--- 运行 JIT 内核 (AVX2) ---\n";
    jit_func_ptr(&args);

    // 6. 验证结果
    std::cout << "--- JIT 结果 ---\n";
    // FIX: Use pointer-based helper
    print_matrix_slice("Batch 0 Output", C_data.data(), jcp.M, jcp.N);
    print_matrix_slice("Batch 1 Output", C_data.data() + mn_size, jcp.M, jcp.N);

    // 7. C++ 对比
    // FIX: Use pointer-based helper
    cpp_matmul_slice(A_data.data(),         
                     B_data.data(),         
                     C_expected.data(),     
                     jcp.M, jcp.K, jcp.N);
    cpp_matmul_slice(A_data.data() + mk_size,
                     B_data.data() + kn_size, 
                     C_expected.data() + mn_size,
                     jcp.M, jcp.K, jcp.N);

    std::cout << "--- C++ 预期结果 ---\n";
    // FIX: Use pointer-based helper
    print_matrix_slice("Batch 0 Expected", C_expected.data(), jcp.M, jcp.N);
    print_matrix_slice("Batch 1 Expected", C_expected.data() + mn_size, jcp.M, jcp.N);

    return 0;
}