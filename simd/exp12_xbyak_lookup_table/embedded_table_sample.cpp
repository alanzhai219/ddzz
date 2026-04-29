#include <array>
#include <cstdint>
#include <iostream>

#include <xbyak/xbyak.h>

namespace
{

    constexpr std::array<uint8_t, 16> kHexDigits = {
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
    };

    constexpr std::array<uint8_t, 16> kNibbleMask = {
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
        0x0F,
    };

    constexpr std::array<uint8_t, 16> kInput = {
        0x00,
        0x01,
        0x02,
        0x03,
        0x04,
        0x05,
        0x06,
        0x07,
        0x08,
        0x09,
        0x0A,
        0x0B,
        0x0C,
        0x0D,
        0x0E,
        0x0F,
    };

} // namespace

class JitEmbeddedNibbleLut : public Xbyak::CodeGenerator
{
public:
    using fn_t = void (*)(uint8_t *dst, const uint8_t *src);

    JitEmbeddedNibbleLut() : Xbyak::CodeGenerator(4096)
    {
        generate();
        ready();
        fn_ = getCode<fn_t>();
    }

    void operator()(uint8_t *dst, const uint8_t *src) const
    {
        fn_(dst, src);
    }

private:
    fn_t fn_ = nullptr;

    void generate()
    {
        using namespace Xbyak;

        const Reg64 &reg_dst = rdi;
        const Reg64 &reg_src = rsi;
        const Reg64 &reg_table = rax;

        const Xmm &xmm_idx = xmm0;
        const Xmm &xmm_table = xmm1;
        const Xmm &xmm_mask = xmm2;

        Label lut;
        Label mask;

        lea(reg_table, ptr[rip + lut]);
        movdqa(xmm_mask, ptr[rip + mask]);

        movdqu(xmm_idx, ptr[reg_src]);
        pand(xmm_idx, xmm_mask);
        movdqa(xmm_table, ptr[reg_table]);
        pshufb(xmm_table, xmm_idx);
        movdqu(ptr[reg_dst], xmm_table);
        ret();

        align(16);
        L(lut);
        for (auto value : kHexDigits)
        {
            db(value);
        }

        align(16);
        L(mask);
        for (auto value : kNibbleMask)
        {
            db(value);
        }
    }
};

int main()
{
#if defined(__GNUC__) || defined(__clang__)
    if (!__builtin_cpu_supports("ssse3"))
    {
        std::cerr << "This sample requires SSSE3.\n";
        return 1;
    }
#endif

    alignas(16) auto src = kInput;
    alignas(16) uint8_t dst[16] = {};

    JitEmbeddedNibbleLut jit;
    jit(dst, src.data());

    std::cout.write(reinterpret_cast<const char *>(dst), 16);
    std::cout << '\n';
    return 0;
}