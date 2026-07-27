#ifndef XBYAK_STRICT_CHECK_MEM_REG_SIZE
#define XBYAK_STRICT_CHECK_MEM_REG_SIZE 1
#endif

#include "xbyak/xbyak.h"

#include <iostream>


struct MyCode : public Xbyak::CodeGenerator {
    MyCode() {
        // mov(rax, ptr[rdi]); // ok
        mov(rax, qword[rdi]); // ok
        // mov(rax, dword[rdi]); // fails with XBYAK_STRICT_CHECK_MEM_REG_SIZE=1
        ret();
    }
};

int main() {
    uint32_t data[] = {1,2,3,4,5,6,7,8,9,10};
    MyCode code;
    auto f = code.getCode<uint32_t (*)(uint32_t*)>();
    auto ret_value = f(data);
    std::cout << ret_value << std::endl; // Should print 1
    return 0;
}