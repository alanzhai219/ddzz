# xbyak 的 table 设计与使用

## 概述

在 Xbyak JIT 代码中，查找表（lookup table，LUT）是将离散映射关系编译进生成代码的常用手段。  
其核心问题只有一个：**表存在哪里，运行时如何拿到它的地址？**

两种主流方案：

| 方案 | 表的位置 | 地址获取方式 |
|------|----------|-------------|
| **方案 A：嵌入代码缓冲区**（Label + `db`/`dd`/`dq`） | JIT 代码缓冲区末尾 | `lea(reg, ptr[rip + label])` 或 `mov(reg, label)` |
| **方案 B：外部静态数组**（`static const` + `mov`） | 进程数据段（.rodata） | `mov(reg, (size_t)ptr)` |

---

## 方案 A：嵌入代码缓冲区（Label + db/dd/dq）

### 原理

在 `generate()` 函数末尾，用 `L(label)` 打标签，再用 `db`/`dw`/`dd`/`dq` 把表数据直接写进同一个 JIT 代码缓冲区。  
运行时用 RIP 相对寻址拿到地址，不依赖任何外部符号。

### 步骤

```
1. 声明标签          Label lut;
2. 在代码段拿地址    lea(reg_table, ptr[rip + lut]);
3. 生成执行代码      pshufb / vpshufb / movdqa ...
4. 跳过数据区        ret();
5. 对齐              align(16) / align(32);
6. 打标签并写数据    L(lut);  db(value0);  db(value1); ...
```

### 代码示例（SSSE3 16 项 nibble LUT）

```cpp
// ── 代码段 ──────────────────────────────────────────
Label lut, nibble_mask;

lea(reg_table, ptr[rip + lut]);         // ← RIP 相对取表基址
movdqa(xmm_mask, ptr[rip + nibble_mask]);

L(vector_loop);
movdqu(xmm_idx, ptr[reg_src]);
pand(xmm_idx, xmm_mask);               // 屏蔽高 4 bit，防止 pshufb 清零
movdqa(xmm_table, ptr[reg_table]);
pshufb(xmm_table, xmm_idx);            // ← SIMD 查表
movdqu(ptr[reg_dst], xmm_table);
// ...
ret();

// ── 数据段（嵌入同一缓冲区）────────────────────────
align(16);
L(lut);
for (auto v : kNibbleLut) db(v);        // 16 字节 nibble 表

align(16);
L(nibble_mask);
for (int i = 0; i < 16; ++i) db(0x0F); // 掩码常量
```

### 标量 fallback 查表

```cpp
// 取表基址
lea(reg_lut256, ptr[rip + lut256]);

// 标量 base + offset 查表
L(tail_body);
movzx(reg_tmp.cvt32(), byte[reg_src]);          // index → reg_tmp
mov(reg_tmp.cvt8(), byte[reg_lut256 + reg_tmp]); // dst = lut[index]
mov(byte[reg_dst], reg_tmp.cvt8());
```

### 注意事项

- `pshufb`/`vpshufb` 会把索引字节 **最高位为 1 时的输出清零**；查表前必须用 `pand` 掩掉高位。
- AVX2 的 `vpshufb` 是 **双 128 bit lane 独立**处理，表需要在两个 lane 中各放一份（32 字节重复）。
- 跳转距离超过 `±127` 字节时，`jz`/`jnz` 需要加 `T_NEAR`；`mov(reg, label)` 在大缓冲区中可能超范围，应改用 `lea(reg, ptr[rip + label])`。

---

## 方案 B：外部静态数组（`static const` + `mov`）

### 原理

在 C++ 侧定义 `static const` 数组，编译器将其放入进程数据段（`.rodata`）。  
在 `generate()` 中用 `mov(reg, (size_t)ptr)` 把该地址作为立即数写入机器码；  
运行时寄存器里就是那块内存的绝对地址，直接用 `ptr[reg]`/`ptr[reg + offset]` 访问。

### 步骤

```
1. C++ 侧声明         static const float lookup[16] = { ... };
2. 取地址写入寄存器   mov(reg_ptr, (size_t)lookup);
3. 加载进向量寄存器   uni_vmovups(vmm_lookup, ptr[reg_ptr]);
4. 用 vpermd/vpshufb 查表
```

### 代码示例（oneDNN brgemm NF4 量化 LUT）

```cpp
// ── C++ 数据（.rodata 段）───────────────────────────
static const float lookup[16] = {
    -1.0f, -0.6962f, -0.5251f, -0.3949f,
    -0.2844f, -0.1848f, -0.0911f,  0.0f,
     0.0796f,  0.1609f,  0.2461f,  0.3379f,
     0.4407f,  0.5626f,  0.7230f,  1.0f
};

// ── JIT generate() 内 ───────────────────────────────
mov(reg_ptr, (size_t)lookup);               // 绝对地址写入寄存器
uni_vmovups(vmm_lookup_low,  ptr[reg_ptr]);           // 加载低 8 项
uni_vmovups(vmm_lookup_high, ptr[reg_ptr + 8 * sizeof(float)]); // 加载高 8 项

// 查表（AVX2 vpermd：32 bit 粒度随机置换）
vpermd(vmm_result, vmm_index, vmm_lookup_low);
```

> 参见 `jit_brgemm_kernel.cpp` 第 3033 行（NF4）和第 3079 行（F4E2M1）。

---

## 两种方案对比

| 维度 | 方案 A：嵌入代码缓冲区 | 方案 B：外部静态数组 |
|------|----------------------|-------------------|
| **表存放位置** | JIT 分配的可执行内存缓冲区 | 进程 `.rodata` 段 |
| **地址获取** | `lea(reg, ptr[rip + label])` | `mov(reg, (size_t)ptr)` |
| **地址类型** | RIP 相对（位置无关） | 绝对地址（编译期固定） |
| **表大小限制** | 受 `CodeGenerator` 构造时指定的缓冲区大小约束 | 实际上无限制 |
| **写表时机** | JIT `generate()` 执行时由 `db/dd/dq` 动态填入 | C++ 编译期确定 |
| **运行时是否可变** | 可在 `generate()` 时按参数计算后写入 | 不可变（`const`） |
| **内存局部性** | 代码与表相邻，缓存命中好 | 代码与表在不同缓存行 |
| **多核安全** | 每个 JIT 实例独立，无共享 | 多实例共享同一份只读表 |
| **适合场景** | 表内容在 JIT 时才能确定；或表较大（几 KB 的 SIMD 数据） | 表是编译期常量；表较小（几十字节）；需要与现有 C++ 数据结构共享 |

---

## 选型指南

```
表是编译期常量？
    ├─ 是 → 表小（≤ 1 KiB）？
    │       ├─ 是 → 方案 B（static const + mov 地址）最简洁
    │       └─ 否 → 方案 A 更节省寄存器，且代码/数据局部性更好
    └─ 否（取决于运行时参数） → 必须用方案 A（generate() 时填表）
```

附加考量：
- **pshufb 类字节混洗**：表大小恰好是 16/32/64 字节，天然适合方案 A，对齐用 `align()` 一行解决。
- **vpermd / float 值映射**：表元素是 `float`，用方案 B 可以用 C++ 浮点字面量直接初始化，不易出错。
- **表被多个 JIT 类共用**：方案 B 只需一份 `.rodata`，方案 A 每个实例各有一份副本。

---

## 两个完整 sample

下面给两个最小完整示例。目的不是追求最强性能，而是把两种 table 的写法完整写出来。

### Sample 1：嵌入式 table，做 16 字节 nibble 字符映射

这个 sample 展示 `Label + db` 的标准写法：表跟在 JIT 代码后面，用 `lea(reg, ptr[rip + label])` 取地址，再用 `pshufb` 查表。

```cpp
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>

#include <xbyak/xbyak.h>

class JitEmbeddedNibbleLut : public Xbyak::CodeGenerator {
public:
    using fn_t = void (*)(uint8_t* dst, const uint8_t* src);

    JitEmbeddedNibbleLut() : Xbyak::CodeGenerator(4096) {
        generate();
        ready();
        fn_ = getCode<fn_t>();
    }

    void operator()(uint8_t* dst, const uint8_t* src) const {
        fn_(dst, src);
    }

private:
    fn_t fn_ = nullptr;

    void generate() {
        using namespace Xbyak;

        const Reg64& reg_dst = rdi;
        const Reg64& reg_src = rsi;
        const Reg64& reg_table = rax;

        const Xmm& xmm_idx = xmm0;
        const Xmm& xmm_table = xmm1;
        const Xmm& xmm_mask = xmm2;

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
        db('0'); db('1'); db('2'); db('3');
        db('4'); db('5'); db('6'); db('7');
        db('8'); db('9'); db('A'); db('B');
        db('C'); db('D'); db('E'); db('F');

        align(16);
        L(mask);
        for (int i = 0; i < 16; ++i) db(0x0F);
    }
};

int main() {
    alignas(16) uint8_t src[16] = {
        0x00, 0x01, 0x02, 0x03,
        0x04, 0x05, 0x06, 0x07,
        0x08, 0x09, 0x0A, 0x0B,
        0x0C, 0x0D, 0x0E, 0x0F,
    };
    alignas(16) uint8_t dst[16] = {};

    JitEmbeddedNibbleLut jit;
    jit(dst, src);

    std::cout.write(reinterpret_cast<const char*>(dst), 16);
    std::cout << '\n';
    return 0;
}
```

运行结果应该是：

```text
0123456789ABCDEF
```

这个 sample 的关键点：
- table 用 `db(...)` 直接写进 JIT 缓冲区。
- 地址不是来自全局变量，而是来自 `rip + lut`。
- 这种方式特别适合 `pshufb` 这种 16 字节小表。

### Sample 2：外部静态 table，做 float 权重查表

这个 sample 展示 `static const + mov(reg, (size_t)table)` 的标准写法：表由 C++ 放到 `.rodata`，JIT 只负责取地址和访问。

```cpp
#include <cstddef>
#include <cstdint>
#include <iostream>

#include <xbyak/xbyak.h>

static const float kWeightLut[4] = {0.125f, 0.25f, 0.5f, 1.0f};

class JitStaticFloatLut : public Xbyak::CodeGenerator {
public:
    using fn_t = float (*)(uint8_t index);

    JitStaticFloatLut() : Xbyak::CodeGenerator(4096) {
        generate();
        ready();
        fn_ = getCode<fn_t>();
    }

    float operator()(uint8_t index) const {
        return fn_(index);
    }

private:
    fn_t fn_ = nullptr;

    void generate() {
        using namespace Xbyak;

        const Reg64& reg_table = rax;
        const Reg32& reg_index = edi;

        mov(reg_table, (size_t)kWeightLut);
        and_(reg_index, 0x03);
        movss(xmm0, ptr[reg_table + rdi * 4]);
        ret();
    }
};

int main() {
    JitStaticFloatLut jit;

    for (uint8_t index = 0; index < 4; ++index) {
        std::cout << "index=" << static_cast<int>(index)
                  << " value=" << jit(index) << '\n';
    }
    return 0;
}
```

运行结果应该类似：

```text
index=0 value=0.125
index=1 value=0.25
index=2 value=0.5
index=3 value=1
```

这个 sample 的关键点：
- table 是普通 C++ `static const` 数组。
- JIT 里直接 `mov(reg_table, (size_t)kWeightLut)`。
- 这种方式特别适合固定不变、可读性优先的数值型常量表。

如果只记一句话：
- 表要跟 JIT 实例一起生成、或者天然就是 16/32 字节 shuffle 表，用嵌入式 table。
- 表本来就是固定常量、希望直接复用 C++ 数组定义，用外部静态 table。

---