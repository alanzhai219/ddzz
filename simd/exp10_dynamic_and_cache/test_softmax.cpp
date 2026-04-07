#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <unordered_map>
#include <memory>
#include <xbyak/xbyak.h>

#include "kernel_cache.hpp"

// ==========================================
// 1. JIT 代码生成器 (Dynamic Part)
// ==========================================
class SoftmaxJITGenerator : public Xbyak::CodeGenerator {
public:
    // type: 0 = Scalar, 1 = AVX Unrolled (for demo purposes)
    SoftmaxJITGenerator(int n, int type)
        : Xbyak::CodeGenerator(Xbyak::DEFAULT_MAX_CODE_SIZE, Xbyak::AutoGrow)
        , n_(n)
        , type_(type) {
        generate();
        // AutoGrow 模式下必须调用 ready() 来完成地址回填/权限设置
        ready();
    }

private:
    void generate() {
        // 寄存器约定 (System V AMD64 ABI / Windows x64 类似)
        // Linux(System V): rdi=input, rsi=output, rdx=n
        // Windows x64: rcx=input, rdx=output, r8=n

        Xbyak::Label loop_start, loop_end, norm_start, norm_end, done;

        push(rbp);
        mov(rbp, rsp);
        push(r12);
        push(r13);
        push(r14);
        push(r15);

        // 本地变量区：64 bytes（也兼容 Windows shadow space）
        sub(rsp, 64);

        // --- 参数搬运 ---
#ifdef _WIN32
        mov(r14, rcx); // input
        mov(r15, rdx); // output
        mov(r12, r8);  // n
#else
        mov(r14, rdi); // input
        mov(r15, rsi); // output
        mov(r12, rdx); // n
#endif
        xor_(r13, r13); // i = 0

        // sum = 0.0f at [rsp + 32]
        xor_(eax, eax);
        mov(dword[rsp + 32], eax);

        // n <= 0 -> return
        test(r12, r12);

        jle(done);

        // --- Pass 1: Exp & Sum ---
        L(loop_start);
        cmp(r13, r12);
        jge(loop_end);

        movss(xmm0, dword[r14 + r13 * 4]);
        mov(rax, reinterpret_cast<uintptr_t>(&::expf));
        call(rax);

        movss(dword[r15 + r13 * 4], xmm0);

        movss(xmm1, dword[rsp + 32]);
        addss(xmm1, xmm0);
        movss(dword[rsp + 32], xmm1);

        inc(r13);
        jmp(loop_start);

        L(loop_end);

        // --- Pass 2: Normalize ---
        movss(xmm1, dword[rsp + 32]);
        xor_(r13, r13);

        L(norm_start);
        cmp(r13, r12);
        jge(norm_end);

        movss(xmm0, dword[r15 + r13 * 4]);
        divss(xmm0, xmm1);
        movss(dword[r15 + r13 * 4], xmm0);

        inc(r13);
        jmp(norm_start);

        L(norm_end);

        L(done);

        add(rsp, 64);
        pop(r15);
        pop(r14);
        pop(r13);
        pop(r12);
        pop(rbp);
        ret();
    }

private:
    int n_;
    int type_;
};

// ==========================================
// 2. 缓存管理器 (Caching Part)
// ==========================================


// ==========================================
// 3. 测试主函数
// ==========================================
int main() {
    using SoftmaxKernelFunc = void (*)(float *, float *, int);

    KernelEngine<SoftmaxJITGenerator, SoftmaxKernelFunc, int, int> engine;

    // 准备数据
    const int N1 = 4;
    const int N2 = 5;
    std::vector<float> in1(N1), out1(N1);
    std::vector<float> in2(N2), out2(N2);

    for(int i=0; i<N1; ++i) in1[i] = (float)i;
    for(int i=0; i<N2; ++i) in2[i] = (float)i;

    auto select_strategy = [](int n) {
        return (n % 4 == 0) ? 1 : 0;
    };

    std::cout << "=== Test 1: Dynamic Shape N=4 (Compile) ===" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto func1 = engine.getOrCompile(N1, select_strategy(N1));
    func1(in1.data(), out1.data(), N1);
    // engine.run(in1.data(), out1.data(), N1);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;
    std::cout << "Output: " << out1[0] << ", " << out1[1] << "..." << std::endl;

    std::cout << "\n=== Test 2: Dynamic Shape N=5 (Compile) ===" << std::endl;
    auto func2 = engine.getOrCompile(N2, select_strategy(N2));
    func2(in2.data(), out2.data(), N2);
    // engine.run(in2.data(), out2.data(), N2);

    std::cout << "\n=== Test 3: Dynamic Shape N=4 (Cache Hit) ===" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    auto func3 = engine.getOrCompile(N1, select_strategy(N1));
    func3(in1.data(), out1.data(), N1);
    // engine.run(in1.data(), out1.data(), N1);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time: " << std::chrono::duration<double, std::milli>(end - start).count() << " ms" << std::endl;

    // 验证结果 (简单的和检查，Softmax 和应为 1)
    float sum = 0;
    for(float v : out1) sum += v;
    std::cout << "\nSum Check (should be ~1.0): " << sum << std::endl;

    return 0;
}