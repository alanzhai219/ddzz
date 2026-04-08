# Paged Attention 与 Adaptive RKV Diversity 详解

这份文档把前面两次说明合并到一起，目标是从零开始讲清楚两件事：

1. Paged Attention 的 KV Cache 是怎么设计和管理的。
2. Adaptive RKV Diversity 是在这个分页 KV Cache 之上，如何计算候选 KV 的冗余度与多样性的。

文中会先讲原理，再给一组可以手算的具体数字，把“写入 block -> 读出 attention -> 计算 diversity”完整串起来。

---

## 1. 为什么需要 Paged Attention

在自回归推理中，当前 token 的 query 需要和历史所有 token 的 key 做相似度计算，再用这些分数去加权历史所有 value：

$$
\mathrm{Attn}(q, K, V) = \mathrm{softmax}\left(\frac{qK^T}{\sqrt{d}}\right)V
$$

为了避免每次都重新计算历史 token 的 K/V，系统会把历史 token 的 K 和 V 存起来，这就是 KV Cache。

传统 KV Cache 如果要求“每个序列一整块连续内存”，会带来几个问题：

1. 序列长度动态增长，连续内存难管理。
2. 多个序列并发时容易产生碎片。
3. 回收、复用、beam search 或请求混合调度时，整块搬移代价很高。

Paged Attention 的核心思路，就是把 KV Cache 像分页内存一样管理：

1. 把 KV Cache 切成固定大小的 block。
2. 不再要求一个序列的 KV 在物理内存上连续。
3. 每个序列只维护“逻辑 block -> 物理 block”的映射表。

也就是：逻辑上连续，物理上离散。

---

## 2. OpenVINO 中 Paged Attention 的关键输入

在 OpenVINO CPU 实现里，Paged Attention 的关键输入定义可以在：

- `src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/executor_pa_common.hpp`
- `src/core/src/op/paged_attention.cpp`

最重要的输入是：

1. `query`, `key`, `value`
   当前这一步新增的 token 对应的 Q/K/V。

2. `key_cache`, `value_cache`
   全局物理 KV Cache。典型布局是：

   $$
   [\text{NUM\_BLOCKS}, H, \text{block\_size}, S]
   $$

3. `past_lens`
   每个序列当前已有多少历史 token。

4. `subsequence_begins`
   当前这批输入 token 在扁平数组里，按序列切分时的起止位置。

5. `block_indices`
   每个序列的逻辑 block 对应到哪些物理 block 的扁平映射表。

6. `block_indices_begins`
   每个序列在 `block_indices` 中的起始偏移。

可以把它想成：

- `key_cache` / `value_cache` 是全局物理仓库
- `block_indices` 是页表
- `past_lens` 是逻辑长度
- `subsequence_begins` 是这轮输入 token 的分段信息

---

## 3. KV Cache 的分页管理直觉

假设 `block_size = 32`。

如果某个序列已经有 70 个历史 token，那么逻辑上它需要 3 个 block：

1. 逻辑 block 0：token 0 到 31
2. 逻辑 block 1：token 32 到 63
3. 逻辑 block 2：token 64 到 95，其中只有前 6 个位置有效

但物理上这三个 block 不一定连续，比如：

1. 逻辑 block 0 -> 物理 block 17
2. 逻辑 block 1 -> 物理 block 4
3. 逻辑 block 2 -> 物理 block 29

所以逻辑序列仍然是连续的 `0..69`，只是物理存储被分页到了不同地方。

---

## 4. 新 token 如何写入分页 KV Cache

CPU 侧追加新 token 的关键路径在 `executor_pa.cpp` 中的 `concat_pastkv()`。

它的核心工作是计算 `slot_mapping`。

`slot_mapping` 的含义是：

当前这一批新增 token 中，第 `i` 个 token 应该写到全局 KV Cache 的哪个物理槽位。

逻辑过程如下：

1. 对每个序列，已知 `past_lens` 和本轮新增长度 `q_len`。
2. 计算新增 token 写入后的逻辑位置。
3. 用 `block_indices` 和 `block_indices_begins` 找到对应的物理 block。
4. 再算出 block 内 offset。
5. 最终得到：

   $$
   \text{slot} = \text{block\_number} \times \text{block\_size} + \text{offset\_in\_block}
   $$

然后：

1. 如果 cache 是浮点，就直接拷贝。
2. 如果 cache 是量化格式，就先量化后写入。

所以 Paged Attention 的“写入”本质不是往一个连续数组尾部 append，而是：

先用逻辑位置找 block table，再把数据写到某个物理 block 的某个槽位。

---

## 5. Attention 计算时如何从分页 Cache 中读取

当要执行 attention 时，算子并不会假设历史 KV 是一整块连续内存。

它的做法是：

1. 根据当前序列找到自己的 block table。
2. 按逻辑 block 顺序遍历。
3. 每访问一个逻辑 block，就通过 `block_indices + block_indices_begins` 找到对应的物理 block。
4. 再从那个物理 block 中读取 key/value。

所以从数学上看，它仍然是在对一条长度为 `cur_kv_len` 的历史序列做 attention；只是从实现上，这条序列是通过页表从离散 block 拼回来的。

---

## 6. 为什么会引出 Adaptive RKV

如果 KV Cache 可以无限增长，当然最好。但真实推理服务里，内存是有限的。

所以必须回答一个问题：

哪些 KV block 值得保留，哪些可以优先淘汰？

最简单的策略是按时间淘汰，但这很粗糙。因为：

1. 有些旧 token 虽然早，但信息很独特。
2. 有些 token 虽然新，但和其他 token 高度重复。

Adaptive RKV 要解决的就是这个问题：

不要只看时间位置，还要看缓存内容是不是冗余。

它在这里选择使用 key 来衡量多样性。直觉是：

如果两个 token 的 key 很像，那么未来 query 很可能会以相似方式匹配它们，它们在检索意义上就更冗余。

---

## 7. AdaptiveRKVDiversityCalculator 在算什么

参考实现位于：

- `src/core/reference/include/openvino/reference/adaptive_rkv_diversity.hpp`

它的输入是：

1. `key_data`，形状为：

   $$
   [\text{num\_heads}, \text{num\_key\_tokens}, \text{head\_size}]
   $$

2. `start_size`
   从序列开头开始，有多少 token 不参与这次 diversity 计算。

3. `eviction_size`
   从 `start_size` 之后开始，有多少 token 属于 eviction area，要参与多样性评估。

4. `block_size`
   最终结果要按 block 聚合。

这里的区间语义是：

- start area: `[0, start_size)`
- eviction area: `[start_size, start_size + eviction_size)`

---

## 8. Diversity 计算流程

它的计算过程可以拆成 6 步。

### 8.1 对每个 token 的 key 做 L2 归一化

$$
\hat{k}_i = \frac{k_i}{\|k_i\|_2 + \epsilon}
$$

这样后续 token 间的点积就变成余弦相似度。

### 8.2 计算所有 token 两两之间的余弦相似度矩阵

$$
C_{ij} = \hat{k}_i^\top \hat{k}_j
$$

会得到一个形状为：

$$
[\text{num\_heads}, \text{num\_key\_tokens}, \text{num\_key\_tokens}]
$$

的相似度张量。

### 8.3 只切出 eviction area 对 eviction area 的子矩阵

保留的就是：

$$
C_{\text{evict}, \text{evict}}
$$

形状变成：

$$
[\text{num\_heads}, \text{eviction\_size}, \text{eviction\_size}]
$$

### 8.4 把对角线清零

因为 token 和自己总是最相似，这个值没有实际意义，会污染统计。

### 8.5 对每一行求均值，把小于均值的相似度置零

这一步的含义不是“所有相似都算冗余”，而是：

只保留一个 token 与其他 token 之间那些显著高于其平均水平的相似关系。

### 8.6 对 head 平均，再按 block 聚合并取负号

先对 head 维做平均，得到：

$$
[\text{eviction\_size}, \text{eviction\_size}]
$$

再按 block 把每 `block_size` 行分成一组求和，并取负号：

$$
D_{b,j} = - \sum_{i \in \text{block } b} \tilde{C}_{ij}
$$

于是：

1. 值越负，通常意味着这个 block 越冗余。
2. 值越接近 0，通常表示它与其他 token 的显著相似关系更少。

---

## 9. 为什么输出不是“每个 block 一个标量”

参考实现返回的不是“每个 block 一个最终分数”，而是：

$$
[\text{eviction\_size}/\text{block\_size}, \text{eviction\_size}]
$$

也就是“每个 eviction block 对 eviction area 中每个 token 的原始多样性值”。

原因是：

1. 最终做淘汰时，系统可能有一部分 block 必须保留。
2. 这个“必须保留的子集”在 Paged Attention kernel 执行时还不知道。
3. 所以 kernel 只能先输出较原始的中间结果。
4. 最后的过滤和再聚合要交给更上层完成。

---

## 10. CPU 后端如何接上 Adaptive RKV

CPU 后端新增的逻辑在：

- `src/plugins/intel_cpu/src/nodes/kernels/scaled_attn/executor_pa.cpp`

总体过程是：

1. 根据 `adaptive_rkv_diversity_block_set_indices` 和对应的 `begins`，找出每个序列要参与 diversity 计算的 block 集合。
2. 从 paged `key_cache` 中把这些 block 对应的 key 数据取出来。
3. 如果 cache 是量化格式，先解量化到 float。
4. 组织成连续的：

   $$
   [H_k, \text{token\_count}, S]
   $$

5. 调用 `AdaptiveRKVDiversityCalculator<float>`。
6. 把结果写到第三个输出。

所以 CPU backend 在这里并没有重新发明一套数学，而是复用了 reference 实现。它真正额外负责的是：

1. 从分页 KV Cache 中收集数据
2. 必要时解量化
3. 把离散物理块整理成 calculator 需要的连续张量

### 10.1 `compute_adaptive_rkv_diversity()` 的输入与输出

`compute_adaptive_rkv_diversity()` 的作用可以概括成一句话：

它从 paged `k_cache` 中，把指定的一组物理 block 抽出来，整理成连续的 `key_data`，然后调用 `AdaptiveRKVDiversityCalculator<float>` 计算可驱逐区域的 diversity 分数，最后写到第三个输出里。

这个函数的关键输入有 5 个：

1. `k_cache`
   分页后的物理 key cache，本质是全局物理仓库。

2. `adaptive_rkv_start_size`
   从逻辑 token 序列开头算起，有多少 token 属于 `start area`，这部分不参与本次 eviction diversity 打分。

3. `adaptive_rkv_evictable_sizes`
   每个序列的 eviction 区长度，形状是 `[B_seq]`，单位是 token。

4. `adaptive_rkv_diversity_block_set_indices`
   一个物理 block 索引表，告诉函数当前序列要从哪些物理 block 中抽 key。

5. `adaptive_rkv_diversity_block_set_indices_begins`
   每个序列在 `adaptive_rkv_diversity_block_set_indices` 中的起止边界，形状是 `[B_seq + 1]`。

输出是：

1. `output_arkv_similarity`
   对每个序列，输出一个形状为：

   $$
   [\text{eviction\_size} / \text{block\_size}, \text{eviction\_size}]
   $$

   的二维矩阵。每一行对应一个 eviction block，每一列对应 eviction 区中的一个 token。

值的直觉是：

1. 越负，通常说明这个 block 和其它 eviction token 越相似，也就是越冗余。
2. 越接近 0，通常说明这个 block 和别人越不像，也就是越多样。

### 10.2 `compute_adaptive_rkv_diversity()` 的逐步计算原理

先看 executor 这层做什么，再看 reference calculator 内部做什么。

#### A. Executor 层的数据准备

对每个序列，CPU 侧做下面几步：

1. 读出 `evictable_size = adaptive_rkv_evictable_sizes[seq_idx]`。
2. 用 `adaptive_rkv_diversity_block_set_indices_begins` 切出当前序列对应的 block 集合。
3. 计算：

   $$
   token\_count = block\_count \times block\_size
   $$

4. 检查 `token_count >= start_size + evictable_size`，确保 block 集足够覆盖 start area 和 eviction area。
5. 分配连续缓冲：

   $$
   key\_data \in [H_k, token\_count, S]
   $$

6. 遍历每个 `hk` 和每个 `block_idx`，按 `adaptive_rkv_diversity_block_set_indices` 指定的顺序，从 paged `k_cache` 中取出物理 block。
7. 如果 cache 是量化格式，先反量化到 float；如果已经是浮点，就直接拷贝。

所以 executor 这层真正做的是：

1. 从离散物理 block 中“重组”出逻辑连续 token 序列。
2. 把这条连续序列整理成 reference 算法想要的 `key_data`。

#### B. Reference calculator 的数学流程

`AdaptiveRKVDiversityCalculator<float>` 的核心流程如下：

1. 对每个 token 的 key 向量做 L2 归一化：

   $$
   \hat{k}_i = \frac{k_i}{\|k_i\|_2 + \epsilon}
   $$

   这样后续点积就等于余弦相似度。

2. 计算所有 token 两两之间的余弦相似度矩阵：

   $$
   C_{ij} = \hat{k}_i^T \hat{k}_j
   $$

   得到形状：

   $$
   [H_k, token\_count, token\_count]
   $$

3. 从完整矩阵中只切出 eviction area 对 eviction area 的子矩阵，形状变成：

   $$
   [H_k, eviction\_size, eviction\_size]
   $$

4. 把对角线清零，因为 token 和自己最相似，这个值没有分析意义。

5. 对每一行求均值，把小于该行均值的相似度置 0。

   这一步的直觉是：
   只保留“显著高于本行平均水平”的强相似关系，把弱相关去掉。

6. 对 head 维做平均，得到：

   $$
   [eviction\_size, eviction\_size]
   $$

7. 按 block 做聚合，并取负号：

   $$
   D_{b,j} = - \sum_{i \in block\ b} \tilde{C}_{ij}
   $$

   于是输出形状变成：

   $$
   [eviction\_size / block\_size, eviction\_size]
   $$

最终，executor 把这个二维结果按行平铺写入第三个输出张量。

### 10.3 一个最小例子：5 个输入如何映射到 `key_data` 和最终输出

下面用一个极小例子，把 5 个输入怎样映射到 `key_data`，再怎样得到最终输出矩阵，完整展开。

设：

1. `B_seq = 1`
2. `Hk = 1`
3. `S = 2`
4. `block_size = 2`
5. `adaptive_rkv_start_size = 2`
6. `adaptive_rkv_evictable_sizes = [2]`
7. `adaptive_rkv_diversity_block_set_indices = [5, 2]`
8. `adaptive_rkv_diversity_block_set_indices_begins = [0, 2]`

含义是：

1. 当前只处理 1 条序列。
2. 每个 block 里有 2 个 token。
3. 要从物理 block `5` 和物理 block `2` 中取 key。
4. 总共得到 4 个逻辑 token。
5. 前 2 个 token 是 `start area`，后 2 个 token 是 `eviction area`。

假设 `k_cache` 中这两个物理 block 的内容是：

```text
Physical Block 5
+---------+--------+
| token 0 | [1, 0] |
| token 1 | [0, 1] |
+---------+--------+

Physical Block 2
+---------+---------+
| token 0 | [1, 1]  |
| token 1 | [1, -1] |
+---------+---------+
```

因为 `adaptive_rkv_diversity_block_set_indices = [5, 2]`，所以逻辑 token 顺序会被重组为：

```text
+---------------+---------------------------+--------+
| logical token | 来源                      | value  |
+---------------+---------------------------+--------+
| token 0       | physical block 5, token 0 | [1, 0] |
| token 1       | physical block 5, token 1 | [0, 1] |
| token 2       | physical block 2, token 0 | [1, 1] |
| token 3       | physical block 2, token 1 | [1,-1] |
+---------------+---------------------------+--------+
```

于是：

$$
key\_data =
\begin{bmatrix}
[1,0] \\
[0,1] \\
[1,1] \\
[1,-1]
\end{bmatrix}
$$

接下来：

1. `start_size = 2`，所以 token 0、1 不参与本次 eviction 打分。
2. `evictable_size = 2`，所以 token 2、3 是 eviction area。
3. 归一化后：

   - token 0: `[1, 0]`
   - token 1: `[0, 1]`
   - token 2: `[0.707, 0.707]`
   - token 3: `[0.707, -0.707]`

4. 完整 4x4 cosine similarity 矩阵是：

```text
                col
            t0      t1      t2      t3
         +-------+-------+-------+-------+
row  t0  | 1.000 | 0.000 | 0.707 | 0.707 |
         +-------+-------+-------+-------+
     t1  | 0.000 | 1.000 | 0.707 |-0.707 |
         +-------+-------+-------+-------+
     t2  | 0.707 | 0.707 | 1.000 | 0.000 |
         +-------+-------+-------+-------+
     t3  | 0.707 |-0.707 | 0.000 | 1.000 |
         +-------+-------+-------+-------+
```

5. 只切 eviction area，即 token 2、3，对应的 2x2 子矩阵：

```text
                col
            t2      t3
         +-------+-------+
row  t2  | 1.000 | 0.000 |
         +-------+-------+
     t3  | 0.000 | 1.000 |
         +-------+-------+
```

6. 对角线清零后：

```text
                col
            t2      t3
         +-------+-------+
row  t2  | 0.000 | 0.000 |
         +-------+-------+
     t3  | 0.000 | 0.000 |
         +-------+-------+
```

7. 每行均值都是 0，阈值过滤后矩阵不变。

8. `Hk = 1`，所以 head 平均后仍然不变。

9. `block_size = 2`，因此整个 eviction area 只对应 1 个 block。按 block 聚合并取负号：

```text
                col
            t2      t3
         +-------+-------+
block 0  | 0.000 | 0.000 |
         +-------+-------+
```

最终输出 shape 为：

$$
[eviction\_size / block\_size, eviction\_size] = [1, 2]
$$

如果把 eviction 区改成两个完全一样的 token，例如：

1. token 2 = `[1,1]`
2. token 3 = `[1,1]`

那么最后输出会变成：

$$
\begin{bmatrix}
-1 & -1
\end{bmatrix}
$$

这更能体现这个值的意义：

1. 越负，表示这个 eviction block 和其它 eviction token 越相似。
2. 也就是越冗余，diversity 越低。

### 10.4 同一个例子的分格图

下面把同一个最小例子直接画成“物理 block -> 逻辑 token -> 相似度矩阵 -> eviction 子矩阵 -> 最终输出”的分格图。

```text
Step 1. 物理 block 里的 key 向量

Physical Block 5
+---------+--------+
| token 0 | [1, 0] |
| token 1 | [0, 1] |
+---------+--------+

Physical Block 2
+---------+---------+
| token 0 | [1, 1]  |
| token 1 | [1, -1] |
+---------+---------+


Step 2. 按 block_set_indices = [5, 2] 重新拼成逻辑 token 序列

block_set_indices_begins = [0, 2]
表示当前 sequence 用 indices[0:2] = [5, 2]

Logical token order
+---------------+---------------------------+--------+
| logical token | 来源                      | value  |
+---------------+---------------------------+--------+
| token 0       | physical block 5, token 0 | [1, 0] |
| token 1       | physical block 5, token 1 | [0, 1] |
| token 2       | physical block 2, token 0 | [1, 1] |
| token 3       | physical block 2, token 1 | [1,-1] |
+---------------+---------------------------+--------+

所以 key_data = [Hk, token_count, S] = [1, 4, 2]

key_data[0] =
[
  [1, 0],
  [0, 1],
  [1, 1],
  [1,-1]
]


Step 3. 划分 start area 和 eviction area

adaptive_rkv_start_size = 2
adaptive_rkv_evictable_size = 2

+---------------+-------------+------------------+
| logical token | 区域        | 是否参与评分     |
+---------------+-------------+------------------+
| token 0       | start area  | 否               |
| token 1       | start area  | 否               |
| token 2       | eviction    | 是               |
| token 3       | eviction    | 是               |
+---------------+-------------+------------------+


Step 4. 归一化后得到 token 向量

token 0: [1, 0]
token 1: [0, 1]
token 2: [1, 1]  / sqrt(2) = [0.707,  0.707]
token 3: [1,-1]  / sqrt(2) = [0.707, -0.707]


Step 5. 计算完整 4x4 cosine similarity 矩阵

                col
            t0      t1      t2      t3
         +-------+-------+-------+-------+
row  t0  | 1.000 | 0.000 | 0.707 | 0.707 |
         +-------+-------+-------+-------+
     t1  | 0.000 | 1.000 | 0.707 |-0.707 |
         +-------+-------+-------+-------+
     t2  | 0.707 | 0.707 | 1.000 | 0.000 |
         +-------+-------+-------+-------+
     t3  | 0.707 |-0.707 | 0.000 | 1.000 |
         +-------+-------+-------+-------+


Step 6. 只切 eviction area 的 2x2 子矩阵
因为 start_size = 2，所以只保留 token 2 和 token 3

                col
            t2      t3
         +-------+-------+
row  t2  | 1.000 | 0.000 |
         +-------+-------+
     t3  | 0.000 | 1.000 |
         +-------+-------+


Step 7. 对角线清零

                col
            t2      t3
         +-------+-------+
row  t2  | 0.000 | 0.000 |
         +-------+-------+
     t3  | 0.000 | 0.000 |
         +-------+-------+


Step 8. 每行求均值，再做阈值过滤

row t2 mean = (0 + 0) / 2 = 0
row t3 mean = (0 + 0) / 2 = 0

规则：保留 >= mean 的值，否则置 0

过滤后不变：

                col
            t2      t3
         +-------+-------+
row  t2  | 0.000 | 0.000 |
         +-------+-------+
     t3  | 0.000 | 0.000 |
         +-------+-------+


Step 9. 对 head 维求均值
因为 Hk = 1，所以结果不变

aggregated_token_similarities =
                col
            t2      t3
         +-------+-------+
row  t2  | 0.000 | 0.000 |
         +-------+-------+
     t3  | 0.000 | 0.000 |
         +-------+-------+


Step 10. 按 block 聚合，得到最终输出
eviction_size = 2, block_size = 2
所以 eviction_blocks = 1

唯一这个 eviction block 包含：
- token 2
- token 3

对每一列做 block 内行求和，再取负号：

output[0, t2] = -(0 + 0) = 0
output[0, t3] = -(0 + 0) = 0

最终输出矩阵：

                col
            t2      t3
         +-------+-------+
block 0  | 0.000 | 0.000 |
         +-------+-------+

shape = [eviction_size / block_size, eviction_size] = [1, 2]
```

---

# 11. 用一组具体数字，手工推演完整流程

下面给一个最小但完整的例子，把：

1. 新 token 写入 block
2. 按 block 读出 attention
3. 再算 diversity

完整走一遍。

为了手算简单，缩小配置如下：

1. `num_heads = 1`
2. `head_size = 2`
3. `block_size = 2`
4. 单个序列
5. 不考虑量化、不考虑 alibi、不考虑 sliding window

真实 OpenVINO CPU 实现中 `block_size` 固定是 32，但机制完全一样。

---

## 12. 定义物理 KV Cache 和 block table

我们准备一个全局物理 `key_cache` / `value_cache`，有 4 个物理 block：

$$
[\text{NUM\_BLOCKS}, H, \text{block\_size}, S] = [4, 1, 2, 2]
$$

假设这个序列逻辑上使用两个 block，但映射到物理 block 2 和物理 block 0：

1. 逻辑 block 0 -> 物理 block 2
2. 逻辑 block 1 -> 物理 block 0

那么这个序列的 block table 可以写成：

- `block_indices = [2, 0]`
- `block_indices_begins = [0, 2]`

含义是：这个序列在 `block_indices` 里的区间是 `[0, 2)`。

---

## 13. 当前这轮新增 token

假设当前序列原来已经有：

- `past_lens = [1]`

这轮又来了 3 个新 token，于是：

- `subsequence_begins = [0, 3]`

表示扁平输入中，这个序列占用 token 区间 `[0, 3)`。

新增 token 的 key/value 定义成：

### 13.1 新增 key

- `k_0 = [1, 0]`
- `k_1 = [0, 1]`
- `k_2 = [1, 1]`

### 13.2 新增 value

- `v_0 = [10, 0]`
- `v_1 = [0, 20]`
- `v_2 = [30, 30]`

---

## 14. 手工计算 slot_mapping

由于历史长度 `past_len = 1`，这 3 个新增 token 写入后的逻辑位置分别是：

1. token 0 -> 逻辑位置 1
2. token 1 -> 逻辑位置 2
3. token 2 -> 逻辑位置 3

因为 `block_size = 2`：

1. 逻辑位置 1 在逻辑 block 0，offset 1
2. 逻辑位置 2 在逻辑 block 1，offset 0
3. 逻辑位置 3 在逻辑 block 1，offset 1

再结合 block table：

1. 逻辑 block 0 -> 物理 block 2
2. 逻辑 block 1 -> 物理 block 0

所以物理写入位置是：

1. token 0 -> 物理 block 2, offset 1
2. token 1 -> 物理 block 0, offset 0
3. token 2 -> 物理 block 0, offset 1

如果写成线性 slot 编号：

$$
\text{slot} = \text{block\_number} \times \text{block\_size} + \text{offset}
$$

就得到：

1. token 0 -> `2 * 2 + 1 = 5`
2. token 1 -> `0 * 2 + 0 = 0`
3. token 2 -> `0 * 2 + 1 = 1`

因此：

$$
\text{slot\_mapping} = [5, 0, 1]
$$

---

## 15. 写入后的物理 Cache 内容

因为 `past_lens = 1`，说明逻辑位置 0 早已存在。假设旧 token 是：

- `k_{past} = [1, 0]`
- `v_{past} = [5, 5]`

逻辑位置 0 属于逻辑 block 0 的 offset 0，也就是物理 block 2 的 offset 0。

写完之后，这个序列逻辑上的 4 个 token 是：

1. 位置 0: `k=[1,0]`, `v=[5,5]`
2. 位置 1: `k=[1,0]`, `v=[10,0]`
3. 位置 2: `k=[0,1]`, `v=[0,20]`
4. 位置 3: `k=[1,1]`, `v=[30,30]`

但物理存储分布是：

### 15.1 物理 block 2

1. offset 0 -> 逻辑 0 -> `k=[1,0]`, `v=[5,5]`
2. offset 1 -> 逻辑 1 -> `k=[1,0]`, `v=[10,0]`

### 15.2 物理 block 0

1. offset 0 -> 逻辑 2 -> `k=[0,1]`, `v=[0,20]`
2. offset 1 -> 逻辑 3 -> `k=[1,1]`, `v=[30,30]`

这就是 paged attention 的本质：

1. 逻辑历史序列是连续的 `[0,1,2,3]`
2. 物理存储散落在不同 block 中

---

## 16. 手工计算一次 Attention

现在算最新 token，也就是逻辑位置 3 的 attention。

设当前 query：

$$
q = [1, 1]
$$

历史 key 矩阵：

$$
K =
\begin{bmatrix}
1 & 0 \\
1 & 0 \\
0 & 1 \\
1 & 1
\end{bmatrix}
$$

历史 value 矩阵：

$$
V =
\begin{bmatrix}
5 & 5 \\
10 & 0 \\
0 & 20 \\
30 & 30
\end{bmatrix}
$$

---

## 17. Attention 读取分页 Cache 的过程

逻辑上它要读取两个逻辑 block：

1. 逻辑 block 0 -> 物理 block 2
2. 逻辑 block 1 -> 物理 block 0

也就是：

1. 先到物理 block 2 取出两个 token
2. 再到物理 block 0 取出两个 token

虽然物理地址不连续，但逻辑上恢复出的历史序列还是：

$$
[k_0, k_1, k_2, k_3]
$$

---

## 18. 计算 Attention Score

点积分别是：

1. $q \cdot k_0 = [1,1] \cdot [1,0] = 1$
2. $q \cdot k_1 = [1,1] \cdot [1,0] = 1$
3. $q \cdot k_2 = [1,1] \cdot [0,1] = 1$
4. $q \cdot k_3 = [1,1] \cdot [1,1] = 2$

所以：

$$
s = [1, 1, 1, 2]
$$

如果按标准缩放，`head_size = 2`，则：

$$
\frac{1}{\sqrt{2}} \approx 0.707
$$

缩放后：

$$
s' \approx [0.707, 0.707, 0.707, 1.414]
$$

记：

1. $e^{0.707} \approx 2.028$
2. $e^{1.414} \approx 4.113$

归一化和：

$$
Z = 2.028 + 2.028 + 2.028 + 4.113 = 10.197
$$

softmax 权重约为：

1. $w_0 = 2.028 / 10.197 \approx 0.199$
2. $w_1 \approx 0.199$
3. $w_2 \approx 0.199$
4. $w_3 = 4.113 / 10.197 \approx 0.403$

---

## 19. 计算最终 Attention 输出

输出为：

$$
o = \sum_i w_i v_i
$$

第一维：

$$
0.199 \cdot 5 + 0.199 \cdot 10 + 0.199 \cdot 0 + 0.403 \cdot 30
$$

$$
= 0.995 + 1.99 + 0 + 12.09 = 15.075
$$

第二维：

$$
0.199 \cdot 5 + 0.199 \cdot 0 + 0.199 \cdot 20 + 0.403 \cdot 30
$$

$$
= 0.995 + 0 + 3.98 + 12.09 = 17.065
$$

因此最终输出大约是：

$$
o \approx [15.08, 17.07]
$$

这说明：虽然 KV Cache 物理上是分页存储的，但只要 block table 正确，attention 的逻辑结果和“连续存储”没有本质区别。

---

## 20. 继续用同一组 key 计算 Adaptive RKV Diversity

现在我们继续用这条历史序列的 key 来算 diversity。

设：

1. `start_size = 2`
2. `eviction_size = 2`
3. `block_size = 2`

这表示：

1. 前 2 个 token 是 start area，不参与这次 diversity 计算。
2. 后 2 个 token 是 eviction area，需要评估冗余度。

所以 eviction area 中的两个 token 是：

1. token 2: `[0,1]`
2. token 3: `[1,1]`

---

## 21. 第一步：归一化 eviction key

### 21.1 token 2

$$
k_2 = [0,1], \quad \|k_2\| = 1
$$

$$
\hat{k}_2 = [0,1]
$$

### 21.2 token 3

$$
k_3 = [1,1], \quad \|k_3\| = \sqrt{2}
$$

$$
\hat{k}_3 = [1/\sqrt{2}, 1/\sqrt{2}] \approx [0.707, 0.707]
$$

---

## 22. 第二步：计算 eviction 区内部的余弦相似度矩阵

两两点积：

1. $\hat{k}_2 \cdot \hat{k}_2 = 1$
2. $\hat{k}_2 \cdot \hat{k}_3 = 0.707$
3. $\hat{k}_3 \cdot \hat{k}_2 = 0.707$
4. $\hat{k}_3 \cdot \hat{k}_3 = 1$

因此：

$$
C =
\begin{bmatrix}
1 & 0.707 \\
0.707 & 1
\end{bmatrix}
$$

---

## 23. 第三步：对角线清零

对角线表示 token 与自身的相似度，清零后：

$$
C' =
\begin{bmatrix}
0 & 0.707 \\
0.707 & 0
\end{bmatrix}
$$

---

## 24. 第四步：按行求均值并过滤

第一行均值：

$$
(0 + 0.707) / 2 = 0.3535
$$

第二行均值同样是：

$$
0.3535
$$

逐元素比较是否大于等于本行均值：

1. 对角线上的 `0` 小于均值，保留为 0
2. 非对角线上的 `0.707` 大于均值，保留

因此过滤后矩阵不变：

$$
\tilde{C} =
\begin{bmatrix}
0 & 0.707 \\
0.707 & 0
\end{bmatrix}
$$

---

## 25. 第五步：对 head 平均

这里只有 1 个 head，所以平均后仍然是：

$$
A =
\begin{bmatrix}
0 & 0.707 \\
0.707 & 0
\end{bmatrix}
$$

---

## 26. 第六步：按 block 聚合，并取负号

因为 `block_size = 2`，而 eviction_size 也是 2，所以 eviction area 中只有 1 个 block。

按列把这一整个 block 的两行加起来：

1. 第 0 列：`0 + 0.707 = 0.707`
2. 第 1 列：`0.707 + 0 = 0.707`

再取负号：

$$
D = [-0.707, -0.707]
$$

因此最终 diversity 输出矩阵是：

$$
\begin{bmatrix}
-0.707 & -0.707
\end{bmatrix}
$$

它的形状是：

$$
[1, 2]
$$

即：

$$
[\text{num\_eviction\_blocks}, \text{eviction\_size}]
$$

---

## 27. 这个 Diversity 输出怎么理解

这个结果不是最终“淘汰决策”，而是更原始的 block-to-token 冗余信息。

它可以理解为：

1. eviction block 中的 token 与 eviction 区其他 token 之间存在显著相似性。
2. 这些相似性累加后得到正值 `0.707`。
3. 再取负号后，输出为 `-0.707`。

所以：

1. 值越负，通常说明这个 block 越冗余。
2. 值越接近 0，通常说明这个 block 与其他 token 的显著相似关系越少。

---

## 28. 把整条链路用一句话串起来

这个例子完整发生了三件事：

1. 新 token 通过 `slot_mapping` 被写入分页 KV Cache 的不同物理 block。
2. attention 计算时，又通过 block table 从离散物理 block 中恢复逻辑历史序列，最终得到输出：

   $$
   o \approx [15.08, 17.07]
   $$

3. Adaptive RKV Diversity 再从同一份 key cache 中抽出 eviction area，计算 token 间余弦相似度并聚合，最终输出：

   $$
   [-0.707, -0.707]
   $$

这说明 Adaptive RKV Diversity 并不是一个脱离 Paged Attention 的独立机制，而是建立在分页 KV Cache 管理之上的“内容冗余评估器”。

---

## 29. 最后总结

如果只记住三句话，可以记下面三句：

1. Paged Attention 解决的是 KV Cache 的物理存储与调度问题，让逻辑连续的历史序列可以分散存放在多个物理 block 中。
2. `block_indices` 这类输入，本质上就是分页 KV Cache 的页表。
3. `AdaptiveRKVDiversityCalculator` 解决的是“这些已经存下来的 KV 哪些更冗余”的问题，它在分页 KV Cache 之上做内容层面的淘汰辅助判断。

---

## 30. 对比表：有无 Adaptive RKV Diversity

下面这张表把“普通 Paged Attention”和“Paged Attention + Adaptive RKV Diversity”按几个关键维度并列对比。

结论先说：KV Cache 的分页存储机制本身不变，变化的是是否会在 attention 之后再利用 key cache 做一轮冗余度分析，并输出给上层做缓存管理。

| 维度 | 只有 Paged Attention | Paged Attention + Adaptive RKV Diversity |
| --- | --- | --- |
| 输入 | 基础输入即可：`q`、`k`、`v`、`k_cache`、`v_cache`、`past_lens`、`subsequence_begins`、`block_indices`、`block_indices_begins`，以及可选的 `scale`、`sliding_window`、`alibi`、`sinks` 等 | 在基础输入上，额外需要 `adaptive_rkv_start_size`、`adaptive_rkv_evictable_sizes`、`adaptive_rkv_diversity_block_set_indices`、`adaptive_rkv_diversity_block_set_indices_begins` |
| 输出 | 主输出是 attention 结果 `output_emb`；可选第二输出 `output_score` | 除 `output_emb`、可选 `output_score` 外，还会有第三输出 `output_arkv_similarity` |
| KV cache 行为 | KV cache 作为历史上下文存储；新 token 通过 `slot_mapping` 写入 paged block；attention 时按 `block_indices` 从离散物理块读回逻辑序列 | KV cache 的写入和读取行为与左侧相同，没有换布局、没有换写法；区别只是 attention 结束后，`k_cache` 还会被再次遍历，作为 diversity 分析的数据源 |
| KV cache 布局 | 典型布局仍是 `[NUM_BLOCKS, H, block_size, S]` 或其量化变体 | 完全相同；adaptive RKV 不要求另一份特殊 cache，也不改变分页组织 |
| 新 token 写入 | 调用 `concat_pastkv()`，根据 `past_lens`、`subsequence_begins`、`block_indices` 计算 `_slot_mapping`，再把 k/v 写进对应物理 block | 完全相同；adaptive RKV 不参与写入逻辑 |
| Attention 主计算 | 只做 paged attention 本身：QK、softmax、再乘 V | 先做同样的 paged attention 主计算，再追加 diversity 分析 |
| 额外计算 | 没有与缓存冗余度相关的额外计算 | 需要从 `k_cache` 中根据 block set 收集 key，必要时解量化，再调用 `AdaptiveRKVDiversityCalculator<float>` 计算冗余度 |
| 对量化 cache 的影响 | 只要 attention 主路径支持对应量化格式即可 | 除主路径外，diversity 还要额外支持从量化 key cache 解码到 float |
| 运行时开销 | 较低，只做 attention 所需计算 | 更高，增加了一次 key 收集、可能的解量化、归一化、相似度矩阵、reduce 等计算 |
| 上层可见信息 | 上层只能拿到 attention 输出，若要做 eviction，通常靠位置、时间、窗口等规则 | 上层除了 attention 输出，还能拿到“候选 KV block 的冗余度/多样性中间结果”，从而做更内容感知的 eviction |
| 上层收益 | 实现简单，开销更低，但 cache 管理策略信息较少 | 可以更智能地判断哪些 KV block 更冗余、更适合淘汰，提升有限 cache 预算下的保留质量 |
| 本质定位 | KV cache 只是“被读写的历史状态” | KV cache 既是“被读写的历史状态”，又是“被分析的内容数据源” |