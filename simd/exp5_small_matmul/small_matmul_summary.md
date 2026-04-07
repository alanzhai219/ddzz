# small_matmul 总结

## 这段代码在做什么

这份 `small_matmul.cpp` 生成了一个 JIT kernel，用来重复计算很多个 `2x2 x 2x2` 的小矩阵乘。

- 输入：`A(2x2)`、`B(2x2)`，以及 batch 次数 `B`
- 输出：`C(2x2)`
- 外层循环：每次处理 1 个 batch，直到 `reg_work_amount == 0`

数学形式：

```text
C[m, n] = sum(A[m, k] * B[k, n]), k = 0..1
```

## 参数是怎么进来的

JIT 函数签名等价于：

```cpp
void kernel(jit_matmul_small_call_args *args)
```

在 Linux x86-64 System V ABI 下，第 1 个整数/指针参数放在 `RDI`，所以：

- `reg_params = rdi`
- `reg_params` 指向 `args`

然后从 `args` 结构体中取出字段，装到工作寄存器：

| 字段 | 含义 | 工作寄存器 |
|---|---|---|
| `input1` | A 的当前地址 | `r8` |
| `input2` | B 的当前地址 | `r9` |
| `output` | C 的当前地址 | `r10` |
| `B` | 剩余 batch 次数 | `r11` |

可以把入口阶段理解成：

```cpp
r8  = args->input1;
r9  = args->input2;
r10 = args->output;
r11 = args->B;
```

## 寄存器级别是怎么计算的

### 1. 通用寄存器负责地址和循环

- `r8`：遍历 A
- `r9`：遍历 B
- `r10`：写回 C
- `r11`：batch 循环计数器

### 2. 向量寄存器里放的是 2x2 的 4 个元素

对 `M = 2, K = 2, N = 2`：

- `vmm_input1[0..3]` 存 A 的 4 个元素
- `vmm_input2[0..3]` 存 B 的 4 个元素
- `vmm_output[0..3]` 存 C 的 4 个累加器

对应关系可以记成：

| 寄存器 | 含义 |
|---|---|
| `vmm_input1[0]` | `A[0,0]` |
| `vmm_input1[1]` | `A[0,1]` |
| `vmm_input1[2]` | `A[1,0]` |
| `vmm_input1[3]` | `A[1,1]` |
| `vmm_input2[0]` | `B[0,0]` |
| `vmm_input2[1]` | `B[0,1]` |
| `vmm_input2[2]` | `B[1,0]` |
| `vmm_input2[3]` | `B[1,1]` |
| `vmm_output[0]` | `C[0,0]` |
| `vmm_output[1]` | `C[0,1]` |
| `vmm_output[2]` | `C[1,0]` |
| `vmm_output[3]` | `C[1,1]` |

### 3. 真正执行的乘加

这段 kernel 本质上做的是：

```text
C00 += A00 * B00
C01 += A00 * B01
C10 += A10 * B00
C11 += A10 * B01

C00 += A01 * B10
C01 += A01 * B11
C10 += A11 * B10
C11 += A11 * B11
```

也就是普通的 `2x2` 矩阵乘，只不过中间值都先留在寄存器里，最后再写回内存。

## 为什么它“用了向量寄存器”，但本质还是标量计算

关键点是这段代码的装载方式是 `vmovss`。

- `vmovss` 每次只读 1 个 `float`
- 真正有效的是寄存器最低的 1 个 lane
- 没有把一个 `ymm` 的 8 个 `float lane` 都装满有效数据

所以虽然代码使用了：

- `Ymm/Zmm` 类型寄存器
- `vfmadd231ps` 这种 packed 指令

但当前版本更准确的描述是：

> 用向量寄存器装标量，并用向量指令形式做标量级乘加。

它不是典型意义上的“AVX2 一条指令并行算 8 个 FP32”。

## 为什么还值得这样写

即使当前不是满宽 SIMD，这种写法也有价值：

- 把中间结果留在寄存器，减少内存往返
- 先建立 micro-kernel 的寄存器 blocking 框架
- 后续改成真正 SIMD 时，寄存器组织方式可以复用

所以这版可以看成：

- 不是最终优化版
- 但已经是一个“寄存器级小核”的雏形

## 怎样改成真正的 AVX2 并行

对这个 `2x2` 小矩阵场景，最合适的方向不是在单个矩阵内部挤 8 路，而是沿 batch 维做 8 路并行。

原因很直接：

- 单个矩阵只有 `2x2`
- `N = 2`，根本吃不满 AVX2 的 8 个 `FP32 lane`

更合理的做法是一次处理 8 个 batch：

- 一个 `ymm` 里放 8 个 batch 的 `A[0,0]`
- 另一个 `ymm` 里放 8 个 batch 的 `A[0,1]`
- B 和 C 也按同样方式组织

这样一条 `vfmadd231ps` 才会真正同时更新 8 个 batch 的结果。

### 目标寄存器组织

| 寄存器 | 内容 |
|---|---|
| `ymm0` | 8 个 batch 的 `A00` |
| `ymm1` | 8 个 batch 的 `A01` |
| `ymm2` | 8 个 batch 的 `A10` |
| `ymm3` | 8 个 batch 的 `A11` |
| `ymm4` | 8 个 batch 的 `B00` |
| `ymm5` | 8 个 batch 的 `B01` |
| `ymm6` | 8 个 batch 的 `B10` |
| `ymm7` | 8 个 batch 的 `B11` |
| `ymm8` | 8 个 batch 的 `C00` |
| `ymm9` | 8 个 batch 的 `C01` |
| `ymm10` | 8 个 batch 的 `C10` |
| `ymm11` | 8 个 batch 的 `C11` |

然后计算变成：

```text
ymm8  = ymm0 * ymm4 + ymm1 * ymm6
ymm9  = ymm0 * ymm5 + ymm1 * ymm7
ymm10 = ymm2 * ymm4 + ymm3 * ymm6
ymm11 = ymm2 * ymm5 + ymm3 * ymm7
```

这时每条 FMA 才是真正 8 路并行。

## 真 AVX2 的前提：数据布局也要配合

如果内存还是当前这种按单个 batch 连续摆放的布局，那么“跨 8 个 batch 取同一位置元素”并不连续，装载成本会很高。

所以要想让 AVX2 真正高效，通常要把数据改成更适合 batch 向量化的布局，比如把同一元素位置在 8 个 batch 上打包存放。

一句话记忆：

> 想让 AVX2 真并行，不能只换寄存器类型，必须同时让数据布局和寄存器布局都服务于 8 个 lane 的并行计算。
