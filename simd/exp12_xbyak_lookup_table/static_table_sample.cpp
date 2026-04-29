#include <array>
#include <cstdint>
#include <iostream>

#include <xbyak/xbyak.h>

namespace
{

    constexpr std::array<float, 4> kWeightLut = {0.125f, 0.25f, 0.5f, 1.0f};

} // namespace

class JitStaticFloatLut : public Xbyak::CodeGenerator
{
public:
    using fn_t = float (*)(uint8_t index);

    JitStaticFloatLut() : Xbyak::CodeGenerator(4096)
    {
        generate();
        ready();
        fn_ = getCode<fn_t>();
    }

    float operator()(uint8_t index) const
    {
        return fn_(index);
    }

private:
    fn_t fn_ = nullptr;

    void generate()
    {
        using namespace Xbyak;

        const Reg64 &reg_table = rax;
        const Reg32 &reg_index = edi;

        mov(reg_table, reinterpret_cast<size_t>(kWeightLut.data()));
        and_(reg_index, 0x03);
        movss(xmm0, ptr[reg_table + rdi * sizeof(float)]);
        ret();
    }
};

int main()
{
    JitStaticFloatLut jit;

    for (uint8_t index = 0; index < 4; ++index)
    {
        std::cout << "index=" << static_cast<int>(index)
                  << " value=" << jit(index) << '\n';
    }

    return 0;
}