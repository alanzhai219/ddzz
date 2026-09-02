# FP32 x64 Transpose 实验

这是一个独立、仅支持 FP32 的实验代码，从 oneDNN x64 JIT `reorder` 实现中提取与数据搬运相关的思路。它刻意保持比 oneDNN 更小的规模，且不是 ABI 兼容的替代实现。

## 已包含的优化

- 针对完整 tile 的 AVX2 `$8\times8$` 寄存器内转置。
- `$64\times64$` cache tile，彼此独立的 tile 可以并行处理。
- 在可用时采用 OpenMP 静态调度。
- 对行或列长度不是 $8$ 的倍数时处理 scalar tail。
- 面向任意逻辑坐标映射的通用 FP32 stride-to-stride `reorder`。
- 显式提供连续存储 3D、4D 张量的 transpose API，接受通用的轴排列。
- `[D0,R,C] -> [D0,C,R]` 与 `[D0,D1,R,C] -> [D0,D1,C,R]` 的快速路径：每个外层切片均使用 AVX2 分块矩阵转置。

通用接口以输出的逻辑坐标系描述问题。对于每一个坐标 $i$，计算：

$$
source = \sum_i coordinate_i \cdot sourceStride_i,
\qquad
output = \sum_i coordinate_i \cdot outputStride_i
$$

测试包含常见的 `NCHW -> NWCH` 张量轴排列映射；它等价于将源数据按 `acdb` 的轴顺序解释，并按 `abcd` 的连续顺序写入目标数据。

## 快速路径如何优化

优化路径面向连续存储的 FP32 矩阵转置。形状为 `[R, C]` 的矩阵满足：

$$
dst[c \cdot R + r] = src[r \cdot C + c]
$$

实现避免将每个元素都当成相互无关的跨步复制：

1. **AVX2 寄存器内转置。** 完整的 `$8\times8$` 数据块被加载为八个 `__m256` 向量。通过 `unpack`、`shuffle` 和 `permute2f128` 指令，在寄存器内重排全部 64 个 FP32 元素，随后通过八次连续向量存储写出输出列。这消除了矩阵主体区域逐元素 scalar 搬运的开销。
2. **Cache blocking。** 矩阵被切分为 `$64\times64$` tile。每个 tile 限制了活跃源数据与目标数据的工作集，降低转置时跨步写入造成的 cache miss。tile 之间没有数据依赖，因此无需同步。
3. **单层并行。** 公共 2D API 通过 OpenMP 静态调度分发 cache tile。专用 3D `{0,2,1}` 与 4D `{0,1,3,2}` 路径则分发彼此独立的 2D plane，再在每个 worker 内串行执行分块 kernel。这样可避免嵌套 OpenMP region 带来的线程过度订阅和同步开销。
4. **Tail 处理。** AVX2 kernel 仅在完整 `$8\times8$` 块可用时执行；剩余行或列采用 scalar 复制。因此，`65x67` 这类形状仍可让大部分数据走向量化快速路径。

对于任意 3D/4D 轴排列，`transpose_3d_fp32()` 与 `transpose_4d_fp32()` 会构造通用 stride-mapping 问题，再进入通用 `reorder` 循环。该路径支持每一种合法排列，但不会假设可以套用连续 `$8\times8$` 矩阵 kernel。专用的末两轴交换能够保留 AVX2/cache-tiled 路径，因为每个外层切片恰好就是一个连续矩阵。

## AVX2 寄存器内部如何完成 `$8\times8$` 转置

寄存器内重排的核心在 `transpose_8x8_avx2()`：一次读取 $8\times8=64$ 个 FP32 元素，在寄存器中完成转置，再写回内存。这样避免了对矩阵主体逐元素执行 load/store。

设输入的 $8\times8$ 矩阵为：

$$
A=
\begin{bmatrix}
a_{00}&a_{01}&a_{02}&a_{03}&a_{04}&a_{05}&a_{06}&a_{07}\\
a_{10}&a_{11}&a_{12}&a_{13}&a_{14}&a_{15}&a_{16}&a_{17}\\
\vdots&&&&&&&\vdots\\
a_{70}&a_{71}&a_{72}&a_{73}&a_{74}&a_{75}&a_{76}&a_{77}
\end{bmatrix}
$$

目标是：

$$
A^T=
\begin{bmatrix}
a_{00}&a_{10}&a_{20}&a_{30}&a_{40}&a_{50}&a_{60}&a_{70}\\
a_{01}&a_{11}&a_{21}&a_{31}&a_{41}&a_{51}&a_{61}&a_{71}\\
\vdots&&&&&&&\vdots\\
a_{07}&a_{17}&a_{27}&a_{37}&a_{47}&a_{57}&a_{67}&a_{77}
\end{bmatrix}
$$

对应的地址映射为：

$$
dst[c \cdot 8 + r] = src[r \cdot 8 + c]
$$

### 1. 载入：一个 `__m256` 对应输入的一行

`r0` 到 `r7` 分别加载输入矩阵的第 0 到第 7 行。每个 `__m256` 保存八个 FP32 值，但 AVX2 寄存器内部由两个独立的 128-bit lane 组成，每个 lane 含四个 FP32。例如：

$$
r_0=[a_{00},a_{01},a_{02},a_{03}\mid a_{04},a_{05},a_{06},a_{07}]
$$

竖线表示两个 128-bit lane。许多 shuffle 指令只能分别在 lane 内工作，因此最后还需要跨 lane 的合并操作。

### 2. `unpack`：两行交错，形成 `$2\times2$` 局部转置

对 `r0` 与 `r1` 执行 `_mm256_unpacklo_ps` 后得到：

$$
t_0=[a_{00},a_{10},a_{01},a_{11}\mid a_{04},a_{14},a_{05},a_{15}]
$$

`_mm256_unpackhi_ps(r0, r1)` 则得到：

$$
t_1=[a_{02},a_{12},a_{03},a_{13}\mid a_{06},a_{16},a_{07},a_{17}]
$$

同样处理 `(r2,r3)`、`(r4,r5)`、`(r6,r7)`，得到 `t2` 到 `t7`。此时数据已经按列方向交错，但每个寄存器只包含两行输入的信息。

### 3. `shuffle`：合并四行，形成 `$4\times4$` 子块的列片段

将 `t0` 与 `t2` 使用 `_mm256_shuffle_ps` 合并：

$$
u_0=[a_{00},a_{10},a_{20},a_{30}\mid a_{04},a_{14},a_{24},a_{34}]
$$

$$
\begin{aligned}
u_1&=[a_{01},a_{11},a_{21},a_{31}\mid a_{05},a_{15},a_{25},a_{35}]\\
u_2&=[a_{02},a_{12},a_{22},a_{32}\mid a_{06},a_{16},a_{26},a_{36}]\\
u_3&=[a_{03},a_{13},a_{23},a_{33}\mid a_{07},a_{17},a_{27},a_{37}]
\end{aligned}
$$

`u4` 到 `u7` 以同样方式表示输入行 `4..7` 的列片段。例如：

$$
u_4=[a_{40},a_{50},a_{60},a_{70}\mid a_{44},a_{54},a_{64},a_{74}]
$$

此时一个输出行的前四项和后四项仍位于不同寄存器中。

### 4. `permute2f128`：跨 128-bit lane 拼成完整输出行

`_mm256_permute2f128_ps(u0, u4, 0x20)` 将 `u0` 和 `u4` 的低 128-bit lane 合并：

$$
[a_{00},a_{10},a_{20},a_{30}\mid a_{40},a_{50},a_{60},a_{70}]
$$

它正好是转置后矩阵的第 0 行。`0x31` 选择两个寄存器的高 lane，得到转置后的第 4 行：

$$
[a_{04},a_{14},a_{24},a_{34}\mid a_{44},a_{54},a_{64},a_{74}]
$$

八次 store 的寄存器来源如下：

| 输出行 | 合并来源 |
|---:|---|
| 0 | `u0`、`u4` 的低 lane |
| 1 | `u1`、`u5` 的低 lane |
| 2 | `u2`、`u6` 的低 lane |
| 3 | `u3`、`u7` 的低 lane |
| 4 | `u0`、`u4` 的高 lane |
| 5 | `u1`、`u5` 的高 lane |
| 6 | `u2`、`u6` 的高 lane |
| 7 | `u3`、`u7` 的高 lane |

### 重排层次总结

整个过程等价于逐步扩大已完成转置的子块：

$$
1\times1 \rightarrow 2\times2 \rightarrow 4\times4 \rightarrow 8\times8
$$

| 阶段 | 指令 | 完成的操作 |
|---|---|---|
| 读取 | `_mm256_loadu_ps` | 读取 8 行，每行 8 个 FP32 |
| 行对交错 | `_mm256_unpacklo_ps`、`_mm256_unpackhi_ps` | 组合两行的数据 |
| 四行重组 | `_mm256_shuffle_ps` | 得到 `$4\times4$` 子块的列片段 |
| 跨 lane 合并 | `_mm256_permute2f128_ps` | 拼接成完整的 8 元素输出行 |
| 写回 | `_mm256_storeu_ps` | 连续写出转置后的列 |

因此，完整 tile 只需要八次向量读取、寄存器 shuffle/permute，以及八次向量写入。中间转置阶段不再访问内存；只有边界不足 `$8\times8$` 的 tail 才回退到 scalar 复制。`transpose_tile()` 将该 kernel 应用于每个完整 `$8\times8$` tile。

## 构建与运行

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/exp17_transpose
```

需要使用支持 AVX2 的编译器和 CPU。项目以 `-mavx2` 编译，不提供非 AVX2 的二进制回退实现。

## 相对 oneDNN reorder 刻意移除的部分

- 所有非 FP32 源/目标数据类型及其转换逻辑。
- scale、zero point、饱和处理、sum（`beta`）及量化补偿。
- oneDNN 的 primitive descriptor、scratchpad、运行时维度和 blocked layout 基础设施。
- oneDNN 完全动态化的 JIT 循环生成器。

保留的实现只专注于普通 FP32 transpose/reorder 的数据搬运。
