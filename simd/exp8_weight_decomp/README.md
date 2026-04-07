# 权重解压缩 JIT 内核 (Weight Decompression Kernel)

## 概述

本项目将 oneDNN 中的 `jit_brgemm_weights_decompression_kernel_t` 内核提取出来，
脱离 oneDNN 框架依赖，改写为基于 **Xbyak** 的独立 JIT 内核，支持 **AVX-512** 和 **AVX2** 两种 ISA。

该内核用于将低精度压缩权重实时解压缩为 `f32`（或 `f16`），
并可选地执行反量化操作（scale + zero_point），核心公式为：

$$
\text{output} = (\text{weight} - \text{zero\_point}) \times \text{scale}
$$

### 3 行速记（M/K/N 视角）

- 标准 GEMM：`C[M,N] = A[M,K] * B[K,N]`，本内核只处理权重 `B[K,N]`
- 映射关系：`oc <-> N`，`(ig, ic) <-> K`
- 合并公式：`k = ig * ic_internal_size + ic`，因此 `B[k,n] == B[ig,ic,oc]`

---

## 文件结构

```
exp8_weight_decomp/
├── weight_decomp_types.hpp    # 数据类型枚举、编译/运行时参数结构体定义
├── weight_decomp_kernel.hpp   # 模板化 JIT 内核 (AVX-512 + AVX2 统一实现)
├── test_weight_decomp.cpp     # 测试程序（23个测试用例 + 参考实现）
└── README.md                  # 本文档
```

---

## 数据类型

### 权重数据类型（输入）

内核支持以下压缩权重格式：

| 类型 | 位宽 | 每byte元素数 | `ic_internal_size` | 说明 |
|------|------|-------------|-------------------|------|
| `u8` | 8-bit | 1 | 1 | 无符号 8-bit 整数 |
| `s8` | 8-bit | 1 | 1 | 有符号 8-bit 整数 |
| `u4` | 4-bit | 2 | 2 | 无符号 4-bit 整数，每 byte 存 2 个值 |
| `s4` | 4-bit | 2 | 2 | 有符号 4-bit 整数，每 byte 存 2 个值 |
| `u2` | 2-bit | 4 | 4 | 无符号 2-bit 整数，每 byte 存 4 个值 |
| `nf4` | 4-bit | 2 | 2 | NormalFloat4（QLoRA），4-bit 索引映射到查找表 |
| `f4_e2m1` | 4-bit | 2 | 2 | FP4 E2M1 格式（1符号+2指数+1尾数），查找表映射 |
| `f16` | 16-bit | — | 1 | IEEE 754 半精度浮点 |
| `bf16` | 16-bit | — | 1 | Brain Float 16 |

### Sub-byte 打包布局

对于 sub-byte 类型（u4/s4/nf4/f4_e2m1/u2），多个值共享一个 byte：

**u4/s4/nf4/f4_e2m1（每 byte 2 个值）：**
```
┌─────────┬─────────┐
│ 高4位    │ 低4位    │   1 byte
│ ic=0    │ ic=1    │
└─────────┴─────────┘
 bit7..4    bit3..0
```

**u2（每 byte 4 个值）：**
```
┌────┬────┬────┬────┐
│ic=0│ic=1│ic=2│ic=3│   1 byte
└────┴────┴────┴────┘
 7:6  5:4  3:2  1:0
```

### 输出数据类型

| 类型 | 字节数 | 说明 |
|------|--------|------|
| `f32` | 4 | 32-bit 单精度浮点（默认输出格式） |
| `f16` | 2 | IEEE 754 半精度浮点（通过 `vcvtps2ph` 从 f32 转换） |

> **注意：** bf16 交织输出仅在 oneDNN 原版中使用（用于 brgemm 布局），本独立实现不包含 bf16 输出路径。

### Scale 数据类型

| 类型 | 说明 |
|------|------|
| `f32` | 32-bit 浮点 scale |
| `e8m0` | 8-bit 纯指数格式，值 $v$ 表示 $2^{v-127}$，例如 `127` → `1.0`，`128` → `2.0`，`126` → `0.5` |

### Zero Point 数据类型

| 类型 | 说明 |
|------|------|
| `f32` | 32-bit 浮点 zero point |
| `u8` | 无符号 8-bit 整数 zero point |
| `u2` | 无符号 2-bit 整数 zero point（仅广播模式） |

---

## 输入与输出

### 编译时参数 (`compile_params_t`)

在创建内核时指定，决定 JIT 生成的代码逻辑：

```cpp
struct compile_params_t {
    bool with_scales;            // 是否使用 scale
    bool with_zero_points;       // 是否使用 zero point
    bool broadcast_scales;       // scale 是否为标量广播（per-tensor）
    bool broadcast_zero_points;  // zero point 是否为标量广播（per-tensor）
    size_t oc_size;              // 输出通道数 OC
    size_t ic_internal_size;     // IC 内部分组大小（由权重类型决定）
    data_type_t weights_dt;      // 权重数据类型
    data_type_t decomp_buffer_dt;// 输出数据类型（f32 / bf16）
    data_type_t scales_dt;       // scale 数据类型
    data_type_t zero_points_dt;  // zero point 数据类型
};
```

### 运行时参数 (`runtime_params_t`)

每次执行时传入，通过寄存器传递给 JIT 代码：

```cpp
struct runtime_params_t {
    const void *weights_ptr;       // 压缩权重缓冲区
    const void *decomp_buffer_ptr; // 解压缩输出缓冲区
    const void *scales_ptr;        // scale 数组（可选）
    const void *zero_points_ptr;   // zero point 数组（可选）
    size_t ic_size;                // IC 分组数（循环次数）also named ig
};
```

### 内存布局

### 默认权重

- 输入布局为 **IC group-major**，即权重按 IC group 分块存储，每个 group 内包含 `oc_size × ic_internal_size` 个逻辑元素。
- 输出布局为 **IC-major (Row-major)**, 即先按 IC group 存储，IC 内部再按 OC 存储。
- scale/zero_point 按 OC 维度索引, 在所有 IC group 之间共享（不随 IC 变化）。

```
# 举例说明
假设: 
- oc_size=128 (8 x AVX-512 向量宽度, 即N=128)
- ic_internal_size=2 (u4 权重)
则每个 IC group 包含 256 个逻辑元素 (128 oc × 2 ic)，
对应 128 bytes 的压缩权重（每 byte 存 2 个 u4 值）。
权重内存布局 (weights_ptr):
IC group 0: [byte0][byte1]...[byte127]   (128 bytes = 256 个 u4 值)
IC group 1: [byte128][byte129]...[byte255]   (128 bytes = 256 个 u4 值)
...
IC group ig: [byte(ig*128)]...[byte((ig+1)*128-1)]
输出内存布局 (decomp_buffer_ptr):
IC group 0:
  ic=0: [oc0_ic0_f32][oc1_ic0_f32]...[oc127_ic0_f32]   (512 bytes)
  ic=1: [oc0_ic1_f32][oc1_ic1_f32]...[oc127_ic1_f32]   (512 bytes)
IC group 1:
  ic=0: [oc0_ic0_f32][oc1_ic0_f32]...[oc127_ic0_f32]   (512 bytes)
  ic=1: [oc0_ic1_f32][oc1_ic1_f32]...[oc127_ic1_f32]   (512 bytes)
...
IC group ig:
  ic=0: [oc0_ic0_f32][oc1_ic0_f32]...[oc127_ic0_f32]   (512 bytes)
  ic=1: [oc0_ic1_f32][oc1_ic1_f32]...[oc127_ic1_f32]   (512 bytes)
```

#### 维度约定

```
OC        = oc_size                    # 输出通道总数
ic_groups = runtime_params_t::ic_size  # IC 分组数（外层循环迭代次数）
ic_int    = ic_internal_size           # IC 内部大小（由权重类型决定: u8/s8=1, u4/s4/nf4/f4_e2m1=2, u2=4）
vec       = vec_size                   # 向量宽度（AVX-512=16, AVX2=8）
oc_blocks = ceil(OC / vec)             # OC 分块数
```

#### `ic_group` 和 `ic_internal_size` 到底是什么？

先只看这两个量：

- `ic_group`（代码中常写 `ig`，运行时来自 `runtime_params_t::ic_size`）
  含义：外层 K 维分组编号，也就是“第几组输入通道块”。
- `ic_internal_size`（代码中常写 `ic_int`）
  含义：每个 `ic_group` 内有多少个组内槽位（`ic`）。

两者关系：

- 外层循环跑 `ic_group` 次
- 每个 `ic_group` 里内层再跑 `ic_internal_size` 次
- 所以总 K 方向逻辑元素数是：`K = ic_group * ic_internal_size`

组内类型对应：

- `u8/s8/f16/bf16`：`ic_internal_size = 1`
- `u4/s4/nf4/f4_e2m1`：`ic_internal_size = 2`（一个 byte 有两个 4-bit 槽位）
- `u2`：`ic_internal_size = 4`（一个 byte 有四个 2-bit 槽位）

把它写成 K 索引公式就是：

`k = ig * ic_internal_size + ic`

示例（u4）：若 `ic_group=3`、`ic_internal_size=2`，则 `K=6`，映射为：

- `ig=0 -> k={0,1}`
- `ig=1 -> k={2,3}`
- `ig=2 -> k={4,5}`

#### 结合 tensor rank 来看（更直观）

同一份权重可以用两种等价视角表示：

1. GEMM 视角（2D，rank-2）

`B[K, N]`

- `K`：输入通道方向
- `N`：输出通道方向

2. 解压实现视角（3D，rank-3）

`W[ig, ic, oc]`

- `ig`：K 方向外层组
- `ic`：组内槽位
- `oc`：输出通道

二者的坐标换算：

- `k = ig * ic_internal_size + ic`
- `n = oc`
- 所以 `B[k, n] == W[ig, ic, oc]`

可把它理解成把 rank-2 的第 0 维 `K` 拆成两维：

`K -> [ig, ic]`，其中 `ig` 是 coarse 维，`ic` 是 fine 维。

具体例子：

- 假设 `K=6, N=4`，并且 `ic_internal_size=2`
- 那么 `ig = K / ic_internal_size = 3`，可写成 `W[3, 2, 4]`
- 例如 `B[5, 2]`：
  - `ig = 5 / 2 = 2`
  - `ic = 5 % 2 = 1`
  - `oc = 2`
  - 对应 `W[2, 1, 2]`

如果从卷积 rank-4 权重看（如 `OIHW`），在固定 `o/h/w` 后，沿 `i` 方向展开得到 GEMM 的 `K`，
本内核做的就是把这条 `K` 轴按 `ic_internal_size` 重新分组后解压。

#### OC/IC 与 weight tensor 的对应关系（总结）

可将权重视为逻辑张量：`W[ig][ic][oc]`。

- `oc` (`0..oc_size-1`)：输出通道索引（某个输入位置连向哪个输出通道）
- `ig` (`0..ic_size-1`)：外层 IC 分组索引（`runtime_params_t::ic_size` 的循环变量）
- `ic` (`0..ic_internal_size-1`)：组内 IC 子索引（由打包位宽决定）

总逻辑元素数：`ic_size * ic_internal_size * oc_size`。

#### 与 GEMM 的 `M/K/N` 一一对应

若用标准 GEMM 记号：

`C[M, N] = A[M, K] * B[K, N]`

本解压缩内核处理的是权重矩阵 `B[K, N]`，因此核心是 `K/N` 两个维度：

- `oc` 对应 `N`（输出通道维，`B` 的列）
- `ig + ic` 共同对应 `K`（输入通道维，`B` 的行）

把 `K` 拆成两级索引：

`k = ig * ic_internal_size + ic`

所以可写成：

`B[k, n] == B[ig, ic, oc]`，其中 `n = oc`。

说明：

- 这里的 `M` 来自激活侧（batch/空间位置），不在本解压 kernel 中处理
- 你看到的 `ocb` 是 `oc` 的向量分块索引：`oc = ocb * vec_size + lane`
- 若看到 `io`，通常是笔误，实际应为 `oc`

对 byte 类型（u8/s8/f16/bf16），线性索引为：

`idx = ig * (ic_internal_size * oc_size) + ic * oc_size + oc`

对 sub-byte 类型（u4/s4/nf4/f4_e2m1/u2），逻辑索引仍然相同，
但物理存储会按打包比例压缩：

- u4/s4/nf4/f4_e2m1：2 值/byte（`ic_internal_size=2`）
- u2：4 值/byte（`ic_internal_size=4`）

因此物理地址可理解为：

`byte_index = idx / pack_scale`，`pack_scale ∈ {1, 2, 4}`

其中 `idx` 对应“逻辑上的第几个权重值”，`pack_scale` 决定每个 byte 可容纳多少值。

向量化执行时再把 `oc` 分解为：

- `ocb = oc / vec_size`（第几个 OC 向量块）
- `lane = oc % vec_size`（块内 lane）

即：`oc = ocb * vec_size + lane`。

#### 从 `(ig, ic, oc)` 到 `byte + bit` 的对照表与算例

先给统一步骤（sub-byte 类型通用）：

1. 逻辑线性索引：`idx = ig * (ic_internal_size * OC) + ic * OC + oc`
2. 物理 byte：`byte_index = idx / pack_scale`
3. byte 内子槽位：`slot = idx % pack_scale`

其中：

- u4: `pack_scale=2`，`slot=0` 对应高 4 bit，`slot=1` 对应低 4 bit
- u2: `pack_scale=4`，`slot=0/1/2/3` 分别对应 `bits[7:6]/[5:4]/[3:2]/[1:0]`

对照表：

| 类型 | `ic_internal_size` | `pack_scale` | `slot` 到 bit 段映射 |
|------|---------------------|--------------|----------------------|
| u4/s4/nf4/f4_e2m1 | 2 | 2 | `slot=0 -> bits[7:4]`, `slot=1 -> bits[3:0]` |
| u2 | 4 | 4 | `slot=0 -> bits[7:6]`, `slot=1 -> bits[5:4]`, `slot=2 -> bits[3:2]`, `slot=3 -> bits[1:0]` |

##### 完整算例 A（u4）

设：`OC=16`, `ic_internal_size=2`, `pack_scale=2`，求 `(ig=1, ic=0, oc=5)` 对应位置。

1. `idx = 1 * (2 * 16) + 0 * 16 + 5 = 37`
2. `byte_index = 37 / 2 = 18`
3. `slot = 37 % 2 = 1`

结论：该元素在 `weights_ptr[18]` 的 `bits[3:0]`（低 4 bit）。

再看同 byte 的另一个值 `(ig=1, ic=0, oc=4)`：

- `idx=36`, `byte_index=18`, `slot=0` -> `bits[7:4]`

因此 `weights_ptr[18] = [oc4 的值(高4位) | oc5 的值(低4位)]`。

##### 完整算例 B（u2）

设：`OC=16`, `ic_internal_size=4`, `pack_scale=4`，求 `(ig=0, ic=2, oc=7)` 对应位置。

1. `idx = 0 * (4 * 16) + 2 * 16 + 7 = 39`
2. `byte_index = 39 / 4 = 9`
3. `slot = 39 % 4 = 3`

结论：该元素在 `weights_ptr[9]` 的 `bits[1:0]`。

同一个 `weights_ptr[9]` 的 4 个 2-bit 槽位对应：

- `slot=0` -> `bits[7:6]`
- `slot=1` -> `bits[5:4]`
- `slot=2` -> `bits[3:2]`
- `slot=3` -> `bits[1:0]`

也就是该 byte 恰好可容纳 4 个逻辑元素（按索引连续排列）。

#### 从 `byte + bit` 反推 `(ig, ic, oc)`（逆映射）

调试时常见输入是：`byte_index` 与 byte 内槽位 `slot`。可按下面步骤反推：

1. 先恢复逻辑线性索引：`idx = byte_index * pack_scale + slot`
2. 记 `group_span = ic_internal_size * OC`
3. 计算：

  - `ig = idx / group_span`
  - `rem = idx % group_span`
  - `ic = rem / OC`
  - `oc = rem % OC`

即：

`(ig, ic, oc) = unravel(idx, [ic_internal_size, OC])`，其中 `idx = byte_index * pack_scale + slot`。

u4 专用（`pack_scale=2, ic_internal_size=2`）：

- `slot=0` 对应 `bits[7:4]`
- `slot=1` 对应 `bits[3:0]`

u2 专用（`pack_scale=4, ic_internal_size=4`）：

- `slot=0/1/2/3` 对应 `bits[7:6]/[5:4]/[3:2]/[1:0]`

快速校验示例：若 `OC=16`，`u2`，`byte_index=9`，`slot=3`：

- `idx = 9 * 4 + 3 = 39`
- `group_span = 4 * 16 = 64`
- `ig = 39 / 64 = 0`
- `rem = 39`
- `ic = 39 / 16 = 2`
- `oc = 39 % 16 = 7`

得到 `(ig, ic, oc) = (0, 2, 7)`，与正向算例一致。

可直接复制到测试代码中的 6 行 `inline` 调试函数：

```cpp
inline void byte_to_ig_ic_oc(size_t b, size_t s, size_t p, size_t ic_int, size_t OC,
               size_t &ig, size_t &ic, size_t &oc) {
  const size_t idx = b * p + s, span = ic_int * OC, rem = idx % span;
  ig = idx / span;
  ic = rem / OC;
  oc = rem % OC;
}
```

#### 权重输入布局 (weights_ptr)

权重在内存中按 **IC group-major** 排列，每个 IC group 内包含 `oc_size × ic_int` 个逻辑值。

**对于 byte 类型（u8, s8, f16, bf16）**，每个值占 1 或 2 bytes，布局为：

```
IC group 0                          IC group 1
┌─────────────────────────┐         ┌─────────────────────────┐
│ oc0  oc1  oc2 ... ocN-1 │  ic=0   │ oc0  oc1  oc2 ... ocN-1 │  ic=0
└─────────────────────────┘         └─────────────────────────┘
```

偏移公式: `offset(ig, ocb, ic) = (ig * OC * ic_int + ic * OC + ocb * vec) * dt_size`

**对于 sub-byte 类型（u4, s4, nf4, f4_e2m1）**，2 个值每 byte，ic_int=2：

```
IC group 0                                    IC group 1
┌──────────────────────────────────────┐      ┌──────────────────────────────────────┐
│ [oc0:ic0|ic1] [oc1:ic0|ic1] ... OC-1 │      │ [oc0:ic0|ic1] [oc1:ic0|ic1] ... OC-1 │
└──────────────────────────────────────┘      └──────────────────────────────────────┘
  每 byte = 2 个 4-bit 值 (高nibble=ic0, 低nibble=ic1)
```

每 byte 存储: `bits[7:4]` = ic=0 的值, `bits[3:0]` = ic=1 的值

偏移公式: `byte_offset(ig, ocb) = ig * OC * ic_int / 2 + ocb * vec * ic_int / 2`

**对于 u2 类型**，4 个值每 byte，ic_int=4：

```
IC group 0                                              IC group 1
┌───────────────────────────────────────────────────┐   ┌───────────────...
│ [oc0: ic0|ic1|ic2|ic3] [oc1: ic0|ic1|ic2|ic3] ...│   │ ...
└───────────────────────────────────────────────────┘   └───────────────...
  每 byte = 4 个 2-bit 值
  bits[7:6]=ic0, bits[5:4]=ic1, bits[3:2]=ic2, bits[1:0]=ic3
```

偏移公式: `byte_offset(ig, ocb) = ig * OC * ic_int / 4 + ocb * vec * ic_int / 4`

#### 输出布局 (decomp_buffer_ptr)

**f32 输出**使用 **IC-major (Row-major)** 布局:

```
IC group 0:
  ic=0: [oc0, oc1, ..., ocN-1]     ← contiguous OC
  ic=1: [oc0, oc1, ..., ocN-1]     ← next ic (仅 sub-byte 类型有 ic>0)
  ...
IC group 1:
  ic=0: [oc0, oc1, ..., ocN-1]
  ic=1: [oc0, oc1, ..., ocN-1]
  ...
```

偏移公式: `offset(ig, ocb, ic) = (ig * ic_int * OC + ic * OC + ocb * vec) * sizeof(f32)`

每次 IC group 迭代后，输出指针推进 `ic_int * OC * sizeof(output_dt)`。

**f16 输出**与 f32 同布局，仅存储为 16-bit（通过 `vcvtps2ph` 转换后存储）。

#### Scale / Zero Point 布局

Scale 和 zero point 按 OC 维度索引，在所有 IC group 之间共享（不随 IC 变化）：

```
Per-channel:  [s_oc0, s_oc1, s_oc2, ..., s_ocN-1]   (OC 个元素)
Broadcast:    [s]                                     (1 个标量)
```

偏移: `offset(ocb) = ocb * vec * sizeof(scale_dt)`

#### 完整示例

以 **u4 权重, oc_size=16 (AVX-512), ic_groups=2** 为例:

```
                    权重内存 (packed)
                    ═══════════════
  IC group 0:  [byte0][byte1]...[byte15]   (16 bytes = 32 个 u4 值 = 16 oc × 2 ic)
  IC group 1:  [byte16][byte17]...[byte31]

  byte0 = [oc0_ic0 (4bit)] [oc0_ic1 (4bit)]
  byte1 = [oc1_ic0 (4bit)] [oc1_ic1 (4bit)]
  ...

                    输出内存 (f32)
                    ═══════════════
  IG0/ic0: [oc0_ic0_f32][oc1_ic0_f32]...[oc15_ic0_f32]   (64 bytes)
  IG0/ic1: [oc0_ic1_f32][oc1_ic1_f32]...[oc15_ic1_f32]   (64 bytes)
  IG1/ic0: [oc0_ic0_f32][oc1_ic0_f32]...[oc15_ic0_f32]   (64 bytes)
  IG1/ic1: [oc0_ic1_f32][oc1_ic1_f32]...[oc15_ic1_f32]   (64 bytes)

  总计: 2 × 2 × 16 × 4 = 256 bytes 输出
```

#### oneDNN bf16 交织输出（仅原版）

oneDNN 中 bf16 输出使用 **OC-major 交织** 布局以满足 brgemm 的数据排列要求：

```
相邻两个 ic 的 bf16 结果交织存储:
  ymm_store0 = vcvtneps2bf16(zmm_ic0)  →  [ic0_oc0, ic0_oc1, ..., ic0_oc7]
  ymm_store1 = vcvtneps2bf16(zmm_ic1)  →  [ic1_oc0, ic1_oc1, ..., ic1_oc7]

  vpunpcklwd → [ic0_oc0, ic1_oc0, ic0_oc1, ic1_oc1, ...]  交织低半部分
  vpunpckhwd → [ic0_oc4, ic1_oc4, ic0_oc5, ic1_oc5, ...]  交织高半部分
  vperm2i128 → 重排 128-bit lane 得到最终布局
```

本独立实现不包含此 bf16 交织路径。

---

## 计算过程

### 总体流程

```
┌──────────────────────────────────────────────────────────────┐
│                    JIT Kernel Entry                          │
│                                                              │
│  1. 保存 callee-saved 寄存器 (preamble)                       │
│  2. 从 runtime_params_t 加载指针到 GPR 寄存器                   │
│  3. 加载查找表常量到 ZMM 寄存器 (nf4/f4_e2m1)                  │
│  4. 预加载 scale/zero_point 到 ZMM 寄存器                      │
│  5. IC 分组循环 ─┐                                            │
│                  │  for each oc_block:                        │
│                  │    for each ic in ic_internal:             │
│                  │      ① 加载压缩权重 → f32 向量              │
│                  │      ② 减去 zero_point（可选）              │
│                  │      ③ 乘以 scale（可选）                   │
│                  │      ④ 存储到解压缩缓冲区                    │
│                  │  推进指针, ic_size--                        │
│                  └──────────────────────                      │
│  6. 恢复寄存器, ret (postamble)                               │
└──────────────────────────────────────────────────────────────┘
```

### 各数据类型的解压缩方法

#### u8 → f32

```
内存: [b0][b1]...[b15]     (16 bytes)
  ↓ vpmovzxbd             zero-extend byte → dword (16个)
ZMM: [d0][d1]...[d15]     (16 × 32-bit int)
  ↓ vcvtdq2ps             int32 → float32
ZMM: [f0][f1]...[f15]     (16 × float)
```

#### s8 → f32

```
内存: [b0][b1]...[b15]     (16 bytes, signed)
  ↓ vpmovsxbd             sign-extend byte → dword
ZMM: [d0][d1]...[d15]     (16 × 32-bit int, signed)
  ↓ vcvtdq2ps
ZMM: [f0][f1]...[f15]     (16 × float)
```

#### u4 → f32

每个 byte 包含 2 个 4-bit 值。需要两次解压（`ic=0` 和 `ic=1`）：

```
ic=0 (高 nibble):                    ic=1 (低 nibble):
  vpmovzxbd  → [0xAB → 0x000000AB]    vpmovzxbd  → [0xAB → 0x000000AB]
  vpsrld 4   → [0x0000000A]           vpslld 28  → [0xB0000000]
                                       vpsrld 28  → [0x0000000B]
  vcvtdq2ps  → [10.0f]                vcvtdq2ps  → [11.0f]
```

#### s4 → f32

与 u4 类似，但使用 **算术右移** (`vpsrad`) 保留符号位：

```
ic=0 (高 nibble):                    ic=1 (低 nibble):
  vpmovsxbd  → sign-extend             vpmovsxbd
  vpsrad 4   → 算术右移保留符号          vpslld 28 → vpsrad 28
  vcvtdq2ps                            vcvtdq2ps
```

#### u2 → f32

每个 byte 包含 4 个 2-bit 值，需要四次解压：

```
Byte = [ic0:ic1:ic2:ic3] = [bit7:6 | bit5:4 | bit3:2 | bit1:0]

ic=0: vpmovzxbd → vpsrld 6            → 取 bit[7:6]
ic=1: vpmovzxbd → vpslld 26 → vpsrld 30  → 取 bit[5:4]
ic=2: vpmovzxbd → vpslld 28 → vpsrld 30  → 取 bit[3:2]
ic=3: vpmovzxbd → vpslld 30 → vpsrld 30  → 取 bit[1:0]
```

#### nf4 → f32（NormalFloat4 查找表）

NF4 是 QLoRA 论文定义的 4-bit 正态分布量化格式，16 个值近似覆盖标准正态分布的分位点：

```
Index:  0      1      2      3      4      5      6      7
Value: -1.000 -0.696 -0.525 -0.395 -0.284 -0.185 -0.091  0.000

Index:  8      9     10     11     12     13     14     15
Value:  0.080  0.161  0.246  0.338  0.441  0.563  0.723  1.000
```

解压缩过程：

```
① 提取 4-bit 索引（同 u4）
② vpermd: 使用索引向量在查找表 ZMM 中查找   ← AVX-512 的优势：
   vmm_lookup = [lut0, lut1, ..., lut15]        单个 ZMM 放下全部 16 项
   vmm_index  = [idx0, idx1, ..., idx15]        一条指令完成 16 路查表
   result[i]  = vmm_lookup[vmm_index[i]]
```

#### f4_e2m1 → f32（FP4 查找表）

F4 E2M1 格式：1 位符号 + 2 位指数 + 1 位尾数：

```
Index: 0     1     2     3     4     5     6     7
Value: 0.0   0.5   1.0   1.5   2.0   3.0   4.0   6.0

Index: 8     9    10    11    12    13    14    15
Value:-0.0  -0.5  -1.0  -1.5  -2.0  -3.0  -4.0  -6.0
```

解压同 nf4：提取索引 → `vpermd` 查表。

#### f16 → f32

```
vcvtph2ps: 硬件指令直接将 16 个 f16 转为 16 个 f32
```

#### bf16 → f32

```
vpmovzxwd: 16-bit → 32-bit (zero-extend)
vpslld 16: 左移 16 位，将 bf16 的位模式放到 f32 的高 16 位
           （bf16 本质上就是 f32 截断低 16 位尾数）
```

### 反量化

加载并转为 f32 后，执行反量化：

```
步骤 ②: vsubps  vmm_weight, vmm_weight, vmm_zero_point   // weight -= zp
步骤 ③: vmulps  vmm_weight, vmm_weight, vmm_scale         // weight *= scale
```
### 输出存储

反量化后的 f32 结果根据输出类型存储：

| 输出类型 | 存储指令 | 说明 |
|----------|----------|------|
| `f32` | `vmovups` | 直接存储 16 个 float (64 bytes) |
| `f16` | `vcvtps2ph` + `vmovdqu16` | f32→16个f16 (32 bytes)，imm8=4 表示 round-to-nearest-even |
| `bf16` | `vcvtneps2bf16` + `vmovdqu16` | f32→16个bf16 (32 bytes)，带交织存储 |

```
f32 路径:   ZMM(16×f32)  ── vmovups ──→  内存 (64 bytes)

f16 路径:   ZMM(16×f32)  ── vcvtps2ph ──→  YMM(16×f16)  ── vmovdqu16 ──→  内存 (32 bytes)

bf16 路径:  ZMM(16×f32)  ── vcvtneps2bf16 ─→  YMM(16×bf16) ── vmovdqu16 ──→  内存 (32 bytes)
```
Scale 和 zero point 支持两种模式：

- **广播模式 (broadcast)**：所有 OC 共享同一个 scalar 值
  - `vbroadcastss` 将标量广播到整个 ZMM
- **Per-channel 模式**：每个 OC 通道有独立的值
  - 直接 `vmovups` 加载 16 个 f32 值到 ZMM

### E8M0 Scale 的特殊处理

E8M0 是一种纯指数格式，将 8-bit 值 $v$ 解释为 IEEE 754 float 的指数部分：

$$
\text{scale} = 2^{v - 127}
$$

实现方式：将 byte 值左移 23 位，直接构造出 float 的位模式：

```
byte value = v
vpslld v, v, 23   →  IEEE754: [0][v(8bit)][00...0(23bit)]  =  2^(v-127)
```

### ic循环

关键在于 `weights_offset` **不随 `ic` 变化**，而不同 `ic` 的数据是通过 `load_weights()` 内部的**位移操作**从同一地址提取的。

分两种情况：

**1. byte+ 类型 (u8/s8/f16/bf16)：`ic_internal_size == 1`**

内层循环只执行一次（`ic=0`），不存在遍历问题。每个通道独占一个完整字节/word。

**2. sub-byte 类型 (u4/nf4/u2 等)：`ic_internal_size > 1`**

多个 IC 值打包在同一字节中，所以 `weights_offset` 对所有 `ic` 都相同——它们读的是**同一块内存**。区分靠 `load_weights(vmm, addr, ic)` 的第三个参数 `ic`：

以 **u4** 为例（`ic_internal_size=2`），一个字节 `[high_nibble | low_nibble]`：
- `ic=0`（偶数）→ `vpsrld(vmm, vmm, 4)` → 提取高 4 位
- `ic=1`（奇数）→ `vpslld 28` + `vpsrld 28` → 提取低 4 位

以 **u2** 为例（`ic_internal_size=4`），一个字节 `[b7b6 | b5b4 | b3b2 | b1b0]`：
- `ic=0` → `vpsrld(vmm, vmm, 6)` → 提取 bit[7:6]
- `ic=1` → `vpslld 26` + `vpsrld 30` → 提取 bit[5:4]
- `ic=2` → `vpslld 28` + `vpsrld 30` → 提取 bit[3:2]
- `ic=3` → `vpslld 30` + `vpsrld 30` → 提取 bit[1:0]

所以遍历逻辑是：

```
外层 ocb:  选择哪一块 vec_size 个 OC 通道 (地址偏移)
内层 ic:   从同一地址的同一字节中，用不同位移提取不同 IC 的值
```

**同一地址，不同位移** —— 这就是 sub-byte packing 的核心设计。

### 输出buffer



输出 buffer 的布局是 **行主序 (row-major)**，形状为 `[ic_internal_size, oc_size]`：

```
ic=0:  [ oc_0, oc_1, oc_2, ..., oc_{oc_size-1} ]
ic=1:  [ oc_0, oc_1, oc_2, ..., oc_{oc_size-1} ]
...
```

公式拆解：

- `jcp_.oc_size * ic`：跳过前 `ic` 行，每行 `oc_size` 个元素 → **行偏移**
- `ocb * vec_size`：在当前行内，跳到第 `ocb` 块的起始位置 → **列偏移**
- 两者相加得到元素索引，再乘 `decomp_dt_size` 转为字节偏移

以 AVX-512 + u4 为例（`oc_size=32, vec_size=16, ic_internal_size=2`）：

```
迭代 (ocb=0, ic=0): offset = (0*16 + 32*0) * 4 = 0     → 写入 [0][0..15]
迭代 (ocb=0, ic=1): offset = (0*16 + 32*1) * 4 = 128   → 写入 [1][0..15]
迭代 (ocb=1, ic=0): offset = (1*16 + 32*0) * 4 = 64    → 写入 [0][16..31]
迭代 (ocb=1, ic=1): offset = (1*16 + 32*1) * 4 = 192   → 写入 [1][16..31]
```

最终输出：
```
行0: [oc_0..oc_15] [oc_16..oc_31]   ← ic=0 的全部 OC
行1: [oc_0..oc_15] [oc_16..oc_31]   ← ic=1 的全部 OC
```

就是标准的二维数组寻址：`output[ic][ocb * vec_size]`。

### 输入buffer

输入（压缩权重）buffer 的布局是**列主序 (column-major)**，形状为 `[oc_size, ic_internal_size]`，但由于 sub-byte packing，多个 ic_internal 会被打包进同一个字节。

具体展开如下：

- **每个 oc block（vec_size 个通道）为一组，连续存储**
- **每个 oc block 内，所有 ic_internal 的数据打包在一起**

以 AVX-512 + u4 为例（`oc_size=32, vec_size=16, ic_internal_size=2`）：

- 总共有 $32/16=2$ 个 oc block
- 每个 oc block 占 $vec\_size \times weights\_dt\_size = 16 \times 1 = 16$ 字节（u4 实际上 2 个通道打包 1 字节，但这里每次只读 16 字节，后续用位移提取）

内存布局如下：

```
[ocb=0]  ic=0,1 的数据（16字节，打包）
[ocb=1]  ic=0,1 的数据（16字节，打包）
```

更细致地看，**每个 oc block 的 1 字节，包含了 ic=0 和 ic=1 的权重**：

- 第 0 字节：oc0, ic0/ic1
- 第 1 字节：oc1, ic0/ic1
- ...
- 第 15 字节：oc15, ic0/ic1

**总结：**

- 外层 oc block，内层 ic_internal
- 每个 oc block 的所有 ic_internal 权重打包在一起，按字节顺序排列
- sub-byte 类型（u4/u2/nf4等）多个 ic_internal 共用同一字节，通过位移提取

公式化描述：

$$
\text{input}[ocb][byte] = \text{packed}(ocb \times vec\_size + byte, ic=0..ic\_internal\_size-1)
$$

即：**每个 oc block 的每个通道，其 ic_internal 个权重被打包在同一字节/连续字节中**。

---

## 寄存器分配

### GPR（通用寄存器）— AVX-512 / AVX2 共用

| 寄存器 | 用途 |
|--------|------|
| `rdi` | 函数参数指针（Linux System V ABI 第一参数） |
| `r8` | 压缩权重指针 `weights_ptr` |
| `r9` | 解压缩输出指针 `decomp_buffer_ptr` |
| `r10` | scale 指针 `scales_ptr` |
| `r11` | zero point 指针 `zero_points_ptr` |
| `r12` | IC 循环计数器 `ic_size` |
| `r13` | 临时寄存器 |

### ZMM（AVX-512 向量寄存器，512-bit = 16 × float，32 个）

```
zmm0  ~ zmm3  : 权重向量（unroll_factor = 4，最多 4 个 OC block 同时处理）
zmm4  ~ zmm7  : scale 向量（per oc_block 预加载）
zmm8  ~ zmm11 : zero point 向量（per oc_block 预加载）
zmm12 ~ zmm27 : 可用
zmm28         : 辅助寄存器
zmm29         : 辅助寄存器
zmm30         : 辅助寄存器
zmm31         : 查找表（nf4 或 f4_e2m1 的 16-entry LUT）
```

### YMM（AVX2 向量寄存器，256-bit = 8 × float，16 个）

由于 AVX2 只有 16 个 YMM 寄存器，分配更紧凑：

```
ymm0  ~ ymm3  : 权重向量（最多 4 个 OC block）
ymm4  ~ ymm5  : scale 向量
ymm6  ~ ymm7  : zero point 向量
ymm8  ~ ymm11 : 可用
ymm12         : 辅助掩码 / nf4 用 mask7 (值>7 的比较结果)
ymm13         : 辅助寄存器 / f4_e2m1 用 sign_mask
ymm14         : 查找表低 8 项（nf4 low table）/ f4_e2m1 的 8-entry abs LUT
ymm15         : 查找表高 8 项（nf4 high table）/ 辅助
```

---

## 编译 & 运行

```bash
cd exp8_weight_decomp

# 编译（同时启用 AVX2 和 AVX-512）
g++ -O2 -mavx2 -mf16c -mavx512f -mavx512bw -mavx512vl -mavx512dq -std=c++17 \
    -I../../3rdparty/xbyak test_weight_decomp.cpp -o test_weight_decomp

# 运行
./test_weight_decomp
```

预期输出：

```
============================================================
  Weight Decompression JIT Kernel Tests
============================================================

CPU: AVX-512F supported
CPU: AVX2     supported

--- AVX-512 Tests ---
[PASS] u8->f32 no_scale (elems=32, max_diff=0)
[PASS] u8->f32 broadcast_scale (elems=32, max_diff=0)
[PASS] u8->f32 scale+zp (elems=32, max_diff=0)
[PASS] s8->f32 broadcast_scale (elems=32, max_diff=0)
[PASS] u4->f32 scale+zp (elems=64, max_diff=0)
[PASS] s4->f32 no_scale (elems=32, max_diff=0)
[PASS] u2->f32 broadcast_scale (elems=64, max_diff=0)
[PASS] nf4->f32 broadcast_scale (elems=64, max_diff=0)
[PASS] f4_e2m1->f32 no_scale (elems=32, max_diff=0)
[PASS] u8->f32 u8_zp+broadcast_scale (elems=16, max_diff=0)
[PASS] u4->f32 e8m0_scale (elems=32, max_diff=0)
[PASS] u8->f32 multi_oc_block (elems=64, max_diff=0)
[PASS] u8->f16 no_scale (elems=32, max_diff=0)
[PASS] u4->f16 scale+zp (elems=64, max_diff=0)
[PASS] nf4->f16 broadcast_scale (elems=32, max_diff=0.000902)

--- AVX2 Tests ---
[PASS] [AVX2] u8->f32 scale+zp (elems=16, max_diff=0)
[PASS] [AVX2] u4->f32 scale+zp (elems=32, max_diff=0)
[PASS] [AVX2] s4->f32 no_scale (elems=16, max_diff=0)
[PASS] [AVX2] u2->f32 broadcast_scale (elems=32, max_diff=0)
[PASS] [AVX2] nf4->f32 broadcast_scale (elems=32, max_diff=0)
[PASS] [AVX2] f4_e2m1->f32 no_scale (elems=16, max_diff=0)
[PASS] [AVX2] u8->f32 multi_oc_block (elems=32, max_diff=0)
[PASS] [AVX2] f16->f32 broadcast_scale (elems=16, max_diff=0)

============================================================
  Results: 23/23 passed
============================================================
```

---

## 测试用例说明

| # | ISA | 测试名 | 权重类型 | Scale | Zero Point | 说明 |
|---|-----|--------|---------|-------|------------|------|
| 1 | 512 | `u8->f32 no_scale` | u8 | 无 | 无 | 最基础：直接 u8→f32 |
| 2 | 512 | `u8->f32 broadcast_scale` | u8 | f32 广播 | 无 | 标量 scale |
| 3 | 512 | `u8->f32 scale+zp` | u8 | f32 per-ch | f32 广播 | scale + zero point |
| 4 | 512 | `s8->f32 broadcast_scale` | s8 | f32 广播 | 无 | 有符号权重 |
| 5 | 512 | `u4->f32 scale+zp` | u4 | f32 广播 | f32 广播 | 4-bit sub-byte |
| 6 | 512 | `s4->f32 no_scale` | s4 | 无 | 无 | 有符号 4-bit |
| 7 | 512 | `u2->f32 broadcast_scale` | u2 | f32 广播 | 无 | 2-bit（4值/byte） |
| 8 | 512 | `nf4->f32 broadcast_scale` | nf4 | f32 广播 | 无 | NormalFloat4 查找表 |
| 9 | 512 | `f4_e2m1->f32 no_scale` | f4_e2m1 | 无 | 无 | FP4 查找表 |
| 10 | 512 | `u8->f32 u8_zp+broadcast_scale` | u8 | f32 广播 | u8 per-ch | u8 类型 zero point |
| 11 | 512 | `u4->f32 e8m0_scale` | u4 | e8m0 广播 | 无 | E8M0 指数型 scale |
| 12 | 512 | `u8->f32 multi_oc_block` | u8 | f32 per-ch | f32 per-ch | 多 OC block（oc=32） |
| 13 | 512 | `u8->f16 no_scale` | u8 | 无 | 无 | f16 输出基本流程 |
| 14 | 512 | `u4->f16 scale+zp` | u4 | f32 广播 | f32 广播 | sub-byte + 反量化 + f16 输出 |
| 15 | 512 | `nf4->f16 broadcast_scale` | nf4 | f32 广播 | 无 | 查找表 + f16 输出 |
| 16 | AVX2 | `u8->f32 scale+zp` | u8 | f32 per-ch | f32 广播 | AVX2 基本 u8 |
| 17 | AVX2 | `u4->f32 scale+zp` | u4 | f32 广播 | f32 广播 | AVX2 4-bit |
| 18 | AVX2 | `s4->f32 no_scale` | s4 | 无 | 无 | AVX2 有符号 4-bit |
| 19 | AVX2 | `u2->f32 broadcast_scale` | u2 | f32 广播 | 无 | AVX2 2-bit |
| 20 | AVX2 | `nf4->f32 broadcast_scale` | nf4 | f32 广播 | 无 | 拆分LUT + vblendvps |
| 21 | AVX2 | `f4_e2m1->f32 no_scale` | f4_e2m1 | 无 | 无 | 符号分离 + 8项LUT |
| 22 | AVX2 | `u8->f32 multi_oc_block` | u8 | f32 per-ch | f32 per-ch | 多OC block（oc=16） |
| 23 | AVX2 | `f16->f32 broadcast_scale` | f16 | f32 广播 | 无 | F16C: vcvtph2ps |

---

## 模板化 ISA 设计 (参考 oneDNN)

参考 oneDNN 的 `template <cpu_isa_t isa>` 模式，使用单个模板类统一 AVX-512 和 AVX2:

```cpp
enum class isa_t { avx2, avx512 };

template <isa_t isa>
class WeightDecompKernel : public Xbyak::CodeGenerator {
    // Vmm: 编译时选择 ZMM 或 YMM
    using Vmm = std::conditional_t<isa == isa_t::avx512, Xbyak::Zmm, Xbyak::Ymm>;
    static constexpr size_t vec_size = (isa == isa_t::avx512) ? 16 : 8;
    static constexpr int    n_vregs  = (isa == isa_t::avx512) ? 32 : 16;
    ...
};

// 使用:
WeightDecompKernel<isa_t::avx512> kernel_512(params);
WeightDecompKernel<isa_t::avx2>   kernel_avx2(params);
```

ISA 差异通过 `if constexpr` 在编译时消除，最终生成无分支的 JIT 代码。
差异点集中在 `load_lookup_tables()` 和 `load_weights()` 的 nf4/f4_e2m1 分支。

### AVX-512 vs AVX2 对比

| 方面 | AVX-512 | AVX2 |
|------|---------|------|
| 向量寄存器 | ZMM (512-bit, 16×f32) | YMM (256-bit, 8×f32) |
| 寄存器数量 | 32 个 | 16 个 |
| vec_size | 16 | 8 |
| 输出类型 | f32, f16 | 仅 f32 |

### nf4 查找表策略

**AVX-512（单条指令）：** 16-entry LUT 正好放入一个 ZMM，`vpermd` 一步完成。

```
zmm31 = [lut0, lut1, ..., lut15]   ← 16 个 float，完整 LUT
vpermd zmm_out, zmm_index, zmm31   ← 16 路并行查表
```

**AVX2（拆分LUT + blend）：** YMM 只能放 8 个 float，需要拆成 low (0-7) 和 high (8-15) 两张表：

```
ymm15 = [lut0, lut1, ..., lut7]    ← 低 8 项
ymm14 = [lut8, lut9, ..., lut15]   ← 高 8 项

① vpcmpgtd ymm_mask, ymm_index, ymm_seven   // mask = (index > 7)
② vpermd   ymm_lo_res, ymm_index, ymm15     // 查低表
③ vpsubd   ymm_hi_idx, ymm_index, ymm_eight // hi_idx = index - 8
④ vpermd   ymm_hi_res, ymm_hi_idx, ymm14    // 查高表
⑤ vblendvps ymm_out, ymm_lo_res, ymm_hi_res, ymm_mask  // 合并
```

### f4_e2m1 查找表策略

**AVX-512：** 同 nf4，16-entry LUT + `vpermd`。

**AVX2（符号分离 + 8项LUT）：** F4 E2M1 的高 8 项是低 8 项的取反，因此：

```
ymm15 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]   ← 仅 8 项绝对值

① vpmovsxbd  ymm_idx, [weights]     // 符号扩展
② vpsrad     ymm_idx, ymm_idx, 4   // 算术右移保留符号
③ vpand      ymm_sign, ymm_idx, ymm_mask  // sign = val & 0x80000000
④ vpermd     ymm_abs, ymm_idx, ymm15      // 查 8 项 abs LUT
⑤ vorps      ymm_out, ymm_abs, ymm_sign   // 合并符号位
```

### AVX2 限制

| 不支持 | 原因 |
|--------|------|
| bf16 输出 | 无 `vcvtneps2bf16` 指令 |
| f16 输出 | 无 `vcvtps2ph`（需 F16C，本实现仅支持 f32 输出） |
| bf16 权重输入 | 无高效 16-bit zero-extend+shift |

---

## 与 oneDNN 原版的区别

| 方面 | oneDNN 原版 | 本实现 |
|------|------------|--------|
| 框架依赖 | 深度依赖 oneDNN 基类、类型系统 | 完全独立，仅依赖 Xbyak |
| ISA 支持 | SSE4.1 / AVX2 / AVX-512 三种模板 | AVX-512 + AVX2 模板 |
| 基类 | `jit_generator_t`（含 `uni_*` 封装层） | `Xbyak::CodeGenerator`（`if constexpr` 适配） |
| 指令封装 | `uni_vmovups` 等统一接口 | `make_vmm_addr` 等地址适配 + `if constexpr` |
| nf4 查找 | AVX2 需拆分为 2×8 + blend | 同 |
| 输出格式 | f32 / bf16(交织) / f16 | f32 / f16（AVX-512），仅 f32（AVX2） |
| 代码生成 | 编译时模板 + 运行时 JIT | 编译时模板 + 运行时 JIT |
