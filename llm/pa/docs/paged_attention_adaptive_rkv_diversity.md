# CPU PagedAttention 中的 Adaptive RKV Diversity

## 概述

本文档说明 CPU plugin 中 PagedAttention 的 Adaptive RKV diversity 生成功能是如何实现的。

该功能为 `PagedAttentionExtension` 增加了第三个输出，用来对运行时选定的一组 KV-cache block 生成 diversity
score。上层组件可以使用这些分数来做 cache eviction 决策。

在 CPU 侧，这部分实现位于 paged-attention executor 中，数学计算部分复用了 OpenVINO Core 中共享的
reference 算法实现。


## 功能目标

Adaptive RKV diversity 的目标，是对候选淘汰区域中的 block 进行打分，度量这些 block 中的 key vector 与其它候选
token 之间的相似程度。

高层逻辑如下：

1. 选出一段可能被淘汰的 key-cache token 区域。
2. 对每个 key vector 做归一化。
3. 计算候选 token 两两之间的余弦相似度。
4. 通过按行均值阈值过滤掉低相似度值。
5. 将 token 级别的相似度累加为 block 级别的 diversity 分数，并取相反数。

负值绝对值越大，表示该 block 与淘汰集合中其它 token 的耦合越强。这个输出的消费者可以结合更高层的保留规则，
继续做进一步聚合或筛选。


## 运行时接口约定

在 executor 侧实现补齐之前，CPU plugin 实际上已经支持了 Adaptive RKV 的输入定义以及输出 shape 推导。

相关输入索引定义在：

- `src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/executor_pa_common.hpp`

相关输出内存大小计算在：

- `src/plugins/intel_cpu/src/nodes/paged_attn.cpp`

这个功能依赖以下 PagedAttention 输入：

1. `adaptive_rkv_start_size`
2. `adaptive_rkv_evictable_sizes`
3. `adaptive_rkv_diversity_block_set_indices`
4. `adaptive_rkv_diversity_block_set_indices_begins`

并写入：

1. 输出索引 `2`，本文称为 Adaptive RKV diversity output


## 输出语义

对于每个 sequence，输出 shape 为：

$$
[\text{evictable\_size} / \text{block\_size},\ \text{evictable\_size}]
$$

多个 sequence 的结果会被展平成一个连续的 `f32` 输出缓冲区。

如果一次请求包含多个 subsequence，CPU 实现会按 subsequence 顺序依次把每段结果追加到输出中。


## 实现位置

CPU 侧实现位于：

- `src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/executor_pa.cpp`

核心新增方法是：

- `AttentionExecutor<...>::compute_adaptive_rkv_diversity(...)`

这个方法在 `AttentionExecutor<...>::execute(...)` 的尾部被调用，也就是常规 paged-attention kernel 已经消费完
最新 KV cache 状态之后。


## 为什么放在 `_kernel(...)` 之后执行

`execute(...)` 中的执行顺序是：

1. 初始化并校验输入输出 tensor
2. 通过 `concat_pastkv(...)` 更新 KV cache
3. 执行主 attention kernel `_kernel(...)`
4. 基于更新后的 key cache 计算 Adaptive RKV diversity

diversity 计算依赖的就是 attention kernel 使用的那一份逻辑 KV cache 状态。把它放在 `concat_pastkv(...)`
之后，可以保证最新写入的 key 已经进入 paged cache。

把它放在 `_kernel(...)` 之后还有一个好处，就是把新功能和 attention 主路径隔离开。attention 的主输出不变，
第三个输出只是附加计算结果。


## `compute_adaptive_rkv_diversity` 内部数据流

对于每个 subsequence：

1. 从 `adaptive_rkv_evictable_sizes` 读取 `evictable_size`。
2. 如果 `evictable_size == 0`，则跳过该 subsequence。
3. 从 `adaptive_rkv_diversity_block_set_indices_begins` 读取 block set 范围。
4. 通过 `adaptive_rkv_diversity_block_set_indices` 将这个范围映射成物理 KV-cache block 列表。
5. 将选中的 key vector 物化成一个稠密临时 tensor，其 shape 为：

$$
[H_k,\ token\_count,\ S]
$$

其中：

- $H_k$ 表示 KV head 数量
- `token_count = block_count * block_size`
- $S$ 表示 key head size

6. 将这个稠密 tensor 传给 `ov::reference::AdaptiveRKVDiversityCalculator<float>`。
7. 将返回的 block diversity 矩阵展平后写入最终输出缓冲区。


## 稠密临时 Tensor 的重建过程

reference 算法要求输入是一个规则张量：

$$
[num\_heads,\ num\_tokens,\ head\_size]
$$

而 paged KV cache 并不是按这种形式存储的。它是按 block 存储的，而且还可能是压缩格式。

因此，CPU executor 首先要把数据重建成一个稠密 float tensor。

对于每个选中的 block 和每个 KV head：

1. 找到 paged cache 中的物理 block 编号
2. 找到这个 block 内对应 head 的数据区域
3. 对该 block 做反量化或类型转换，得到 float 数据
4. 将解码后的向量写入稠密 `[Hk, token_count, S]` buffer

这个重建阶段就是 paged 运行时格式与共享数学 reference 实现之间的关键桥梁。


## 支持的 Cache 布局

CPU 实现支持 paged-attention executor 当前已经支持的同一组 key-cache 形式。

### 非压缩 cache

对于 `f32`、`f16`、`bf16` 这类非压缩 key cache，executor 使用 `cvt_copy(...)` 将 block 内容转换成 `float`。

### `u8` 和 `u4` 压缩 cache

对于 `u8` 和 `u4` key cache，executor 复用了已有 helper：

- `dequant<float, KEY_PREC>(...)`

这个 helper 已经能够处理：

1. by-token quantization
2. by-channel quantization

这样可以避免在新功能里重复实现一遍 cache-layout 解码逻辑。

### `i8` SageAttention cache

对于 `i8` key cache，executor 使用已有的底层 helper：

- `attn_dequant_kernel<float, ov::element::i8>(...)`

之所以单独处理这一路，是因为它的打包行布局和 `u8`、`u4` 的布局不一样。

实现里会先根据每个 group 的 payload 大小计算单行跨度：

$$
row\_stride = \sum (group\_size + sizeof(float))
$$

这里的求和范围是一个 token 行中全部 group。

然后再按 group 逐段把每个 token 行反量化到稠密 float buffer 里。


## 为什么复用现有反量化 Helper

新功能刻意复用了 CPU plugin 现有的反量化 helper，而不是再单独实现一套新的解码路径。

这样做有几个明显好处：

1. cache-layout 相关知识点只保留在一处
2. 新功能与主 attention 路径采用相同的数值约定
3. 已有量化模式的支持几乎可以直接继承
4. 将来 cache-layout 调整时，不容易出现 attention 路径和 diversity 路径各自漂移的问题


## CPU 侧使用的 Reference 算法

在稠密 key tensor 重建完成后，CPU 实现将数学计算委托给：

- `openvino/reference/adaptive_rkv_diversity.hpp`

reference calculator 内部执行以下步骤。

### 1. 对 key 做归一化

每个 key vector 都会沿着 head-size 维度做 L2 normalization。

### 2. 计算余弦相似度矩阵

归一化后的 tensor 与自身转置相乘，得到：

$$
[num\_heads,\ num\_tokens,\ num\_tokens]
$$

这就是所有候选 token 的两两余弦相似度。

### 3. 切出 eviction 区域

只有如下区间对应的方形子矩阵会被用于最终打分：

- token 区间 `[start_size, start_size + eviction_size)`

### 4. 将对角线置零

通过把对角线设为零，去掉 token 对自身的相似度。

### 5. 按行均值阈值过滤

对每个 head、每一行，先计算该行平均相似度，再把低于该均值的值全部清零。

这样能保留每个 token 相对更强的相似关系。

### 6. 在 head 维度做均值

经过过滤后的相似度会在 KV head 维度上做平均。

### 7. 按 block 做求和归约

剩余的 token 级别相似度会进一步按 block 聚合：

$$
diversity(block, token) = -\sum_{t \in block} similarity(t, token)
$$

负号本身就是 reference 定义的一部分，CPU 实现保持了这个约定。


## 为什么 CPU 代码直接调用 Reference Calculator

理论上 CPU plugin 也可以自行实现整套数学计算，但当前设计刻意保持简单。

新的 CPU 代码只做两件事：

1. 将运行时 cache 数据解码并重排成稠密 tensor
2. 调用共享 reference 算法完成数学计算

这样可以带来：

1. 与共享实现保持一致的正确性
2. 更低的维护成本
3. 更小的 plugin 侧代码改动规模
4. 更容易与其它 backend 做行为对齐


## 多 Sequence 打包与 Flatten 输出

paged-attention 节点一次调用可以处理多个 subsequence，而每个 subsequence 的 `evictable_size` 可能不同。

因此 executor 无法为整次请求写出一个固定二维 shape 的输出，而是要写成展平的一维缓冲区。

输出缓冲区布局如下：

1. subsequence 0 的 diversity 结果
2. subsequence 1 的 diversity 结果
3. subsequence 2 的 diversity 结果
4. 依此类推

`paged_attn.cpp` 中的节点级输出 shape 逻辑，会按下面的公式预先算出总输出元素数：

$$
\sum_i \frac{evictable\_size_i^2}{block\_size}
$$

executor 的实际写入顺序与这个 flatten 约定严格一致。


## 安全检查

CPU 实现会校验选中的 block set 与请求打分区域之间的关系是否合法。

对每个 subsequence 都会检查：

$$
token\_count \ge start\_size + evictable\_size
$$

如果条件不成立，就会触发断言，因为这意味着请求的 eviction 窗口超出了当前物化出来的 token 子集范围。


## 与现有 PagedAttention 输出的关系

这个功能不会改变以下两个输出的语义：

1. output 0：attention 主输出
2. output 1：可选的 score 输出

它只是在以下条件满足时为 output 2 增加计算逻辑：

1. 图中请求了 Adaptive RKV diversity output
2. `adaptive_rkv_evictable_sizes` 非空
3. 提供了 `adaptive_rkv_diversity_block_set_indices`

如果功能未启用，常规 attention 执行路径保持不变。


## 测试策略

CPU 侧回归测试添加在：

- `src/plugins/intel_cpu/tests/functional/custom/subgraph_tests/src/x64/paged_attn_token_type.cpp`

测试内容如下：

1. 构造一个启用了 output 2 的小型 `PagedAttentionExtension` 图
2. 使用确定性的 `f32` KV-cache 内容
3. 构造一个 subsequence，并设置：
   - `start_size = 32`
   - `evictable_size = 32`
   - block set 为 `{0, 1}`
4. 执行 CPU inference
5. 在测试代码中独立计算 expected diversity 值
6. 用 `EXPECT_NEAR` 对比 plugin 输出与预期结果

这个测试覆盖了端到端链路：

1. 节点输出大小计算
2. executor 调度
3. block gathering
4. 稠密 tensor 重建
5. diversity 输出 flatten


## 与 GPU 实现的关系

commit `083dcb639be7e1646fa85cb6d0fb088f33f8c060` 中的 GPU 实现，使用了专门的 OpenCL kernel pipeline 以及若干
中间 GPU buffer 来解决同一个问题。

CPU 端刻意采用了不同的实现策略。

GPU 侧：

1. 计算始终保留在设备端
2. 为归一化、相似度、阈值过滤和 block 聚合分别引入了专用 kernel
3. 重点围绕 GPU 执行模型和显存搬运做优化

CPU 侧：

1. 先在 host memory 中重建稠密 key 数据
2. 数学部分复用共享 reference 实现
3. 复用现有 CPU cache-dequant helper
4. 优先保证正确性、可维护性和低实现风险，而不是立刻引入一套新的 SIMD 专用 diversity kernel

尽管实现策略不同，但两个 backend 在 op 边界上遵循的是同一个逻辑契约。


## 性能考量

当前 CPU 实现并不在 attention 主 kernel 的关键路径上。它是一个辅助功能，只在特定运行模式下请求第三个输出时
才会执行。

主要开销来自：

1. 将 paged KV cache 解码成稠密 float buffer
2. 物化 reference calculator 需要的相似度矩阵

在当前设计下，这样的代价是可以接受的，因为：

1. 功能本身是可选的
2. 初版 CPU port 的首要目标是正确性与跨 backend 一致性
3. 实现保持紧凑，并且与 OpenVINO 共享 reference 契约保持一致

如果后续 profiling 发现这部分在 CPU 上变成热点，可以在不修改公开运行时接口的前提下，再引入更内核化的优化
实现。


## 总结

CPU 上的 Adaptive RKV diversity 实现采用了两阶段设计：

1. 将运行时 paged KV-cache block 解码并重建成稠密 float tensor
2. 调用共享 reference diversity 算法，并将结果展平后写到 output 2

这种设计把新增代码局限在较小范围内，复用了已经验证过的 helper，支持现有 CPU cache 格式，并且能够与上层所
期望的、backend 无关的数学行为保持一致。