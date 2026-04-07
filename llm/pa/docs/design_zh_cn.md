# Standalone Paged Attention 教学项目中文说明

本文档面向想理解 paged attention 运行时核心机制的读者，重点解释这个教学项目的流程、设计原理，以及各层之间如何配合工作。

这份实现不是为了追求最高性能，而是为了把下面这条主线讲清楚：

1. 序列如何映射到物理 KV block
2. block table 如何驱动 KV 写入和历史读取
3. prefill、decode、beam fork、beam merge、finish 这些动作如何改变运行时状态
4. 为什么运行时要把“调度层”和“执行层”拆开

## 1. 这个项目在解决什么问题

在大模型推理里，attention 需要读取历史 token 的 K/V。一个直接但低效的做法是为每个序列分配一整段连续内存，然后随着序列增长不断扩容或搬运数据。

Paged attention 的核心思想是：

1. 不要求一个序列的 KV 在物理内存中连续存放
2. 把 KV cache 切成固定大小的 block
3. 每个序列只维护“逻辑顺序上的 block 列表”
4. 执行时通过 block table 把逻辑 token 位置翻译成物理 block + offset

这样做的直接收益是：

1. 追加 token 时通常只需要申请新 block，而不是整体搬迁
2. 多个 beam 可以共享同一段历史 KV，直到真正发生写入时再 copy-on-write
3. block 可以在序列结束后被回收，重新分配给新序列

从抽象上看，这和操作系统里的分页机制很像：

1. 逻辑地址不直接等于物理地址
2. 需要一层映射表做翻译
3. 物理资源可以共享、回收、再分配

## 2. 整体架构分层

这个教学项目把系统分成三层，再加一个多层 LLM 包装层。

### 2.1 调度层：KVBlockManager

职责是管理“谁拥有哪些 block”。它不做 attention 计算，只关心资源和映射关系。

它负责：

1. 维护空闲 block 列表 `free_blocks`
2. 维护每个物理 block 的引用计数 `block_ref_counts`
3. 维护每个序列的状态 `SequenceState`
4. 为 prefill / decode 预留可写空间
5. 在 beam 共享尾块且即将写入时触发 copy-on-write
6. 在 sequence finish 时回收 block
7. 构建执行层需要的 metadata

### 2.2 地址翻译层：ExecutorPACommon

职责是把“逻辑 token 位置”翻译成“物理 KV 位置”。

它负责：

1. 为新 token 的 KV 写入构建 slot mapping
2. 为 attention 读取历史上下文构建 `(block, offset)` 列表

这层本身不存数据，也不做矩阵计算。它只是把调度层产出的 block table 解释成执行层可直接使用的地址。

### 2.3 执行层：PagedAttentionExecutor

职责是对某一层 attention 的 KV cache 做真实读写，并执行 attention。

它负责：

1. 把当前 step 产生的 K/V 写入 paged KV cache
2. 按 block table 读取历史 K/V
3. 执行 prefill attention
4. 执行 decode attention
5. 在启用压缩时做 int8 量化和反量化
6. 在 copy-on-write 时复制物理 block

### 2.4 多层封装：ToyLayer 和 ToyLLMRuntime

这是为了模拟真实 LLM 会有多层 attention，而不是单层。

其中：

1. `ToyLayer` 表示一层 attention 模块，有自己的投影权重和自己的 KV cache
2. `ToyLLMRuntime` 表示整个推理运行时，内部持有一个全局调度器和多个 layer

关键点是：

1. 所有 layer 共享同一份调度元数据
2. 但每一层都有自己独立的 KV cache
3. 如果调度层要求 copy 某个 block，就要对所有 layer 同步执行同样的 block copy

这正是“逻辑调度全局共享、物理 cache 分层独立”的核心设计。
### 2.5 一个 LLM 有多少个 paged attention 实例

在真实 transformer 模型中，每一层 attention 都需要自己的 KV cache。因此，如果模型有 `num_layers` 层（demo 中 `num_layers = 4`），就会有 `num_layers` 个 `PagedAttentionExecutor` 实例，每个实例持有独立的 K cache 和 V cache 张量。

它们之间的关系是：

1. **共享的部分**：所有层共用同一个 `KVBlockManager` 调度器。因为 token 的逻辑位置在每一层都相同——第 5 个 token 永远是第 5 个 token，不会因为层不同而改变位置。所以 block table、引用计数、copy-on-write 决策都只需要维护一份。

2. **独立的部分**：每层的 KV cache 内容完全不同。同一个 token 在第 0 层产生的 K/V 向量和在第 3 层产生的完全不同（因为投影权重不同，输入 hidden state 也不同）。所以每一层都必须有自己的 `k_cache` 和 `v_cache` 张量。

3. **同步操作**：当调度层决定做 copy-on-write 时，**所有** 层都要把对应的物理 block 复制一份。因为逻辑映射是共享的，如果只复制部分层，剩下的层就会出现"映射指向新块，但数据还在旧块"的失配。

用 demo 参数来说：

- 系统中有 4 个 `PagedAttentionExecutor`
- 每个持有自己的 `k_cache[64, 4, 4, 8]` 和 `v_cache[64, 4, 4, 8]`
- 但只有 1 个 `KVBlockManager`，所有 4 层都使用它产出的 metadata
- 一次 block copy 要在 4 层上各执行一次，总搬运量是单层的 4 倍

这也解释了为什么 `ToyLLMRuntime` 里的 copy 循环是 `for layer in self.layers: layer.copy_block(...)`——每层都不能漏。
## 3. 关键数据结构

### 3.1 SequenceState

每个序列保存三类信息：

1. `seq_id`：序列 ID
2. `logical_blocks`：这个序列当前拥有的逻辑 block 列表
3. `past_len`：已经提交到 KV cache 的 token 数

其中 `logical_blocks` 非常关键。它表示：

1. 第 0 个逻辑 block 对应哪个物理 block
2. 第 1 个逻辑 block 对应哪个物理 block
3. 以此类推

注意，这里保存的是“按逻辑顺序排列的物理 block 列表”，而不是连续物理地址。

### 3.2 BatchMetadata

这是调度层传给执行层的桥梁结构。

字段含义如下：

1. `past_lens`
   表示 batch 中每个序列在当前 step 开始前已经有多少历史 token

2. `subsequence_begins`
   表示 batch 输入里每个子序列在拼接后的 token 轴上的起始位置

3. `block_indices`
   把当前 batch 中所有序列需要使用的物理 block 串起来，形成一张平铺后的 block table

4. `block_indices_begins`
   表示每个序列在 `block_indices` 中的起始偏移

可以把它理解成一种“压平后的稀疏映射表”。执行层不用知道序列对象内部结构，只要拿到这四个数组，就能完成地址翻译。

### 3.3 BlockCopyPlan

这是 copy-on-write 的执行计划。

它只包含两项：

1. `src_block`
2. `dst_block`

调度层先决定“哪个共享尾块需要复制成哪个新块”，执行层只负责照这个计划在每层 cache 上把数据复制过去。

这个设计把“策略决策”和“数据搬运”拆开了。
### 3.4 `num_heads`、`hidden_size` 和多头注意力

要理解 KV cache 的形状，需要先理解 `num_heads` 和 `hidden_size` 这两个模型架构参数。

**`hidden_size`：模型的主干宽度**

`hidden_size` 是每个 token 在层与层之间传递的向量长度。它是 LLM 的"对外接口宽度"——每一层的输入和输出都是 `[tokens, hidden_size]` 形状的张量。

在代码里它决定了投影权重的输入维度和输出投影的目标维度：

```python
# ToyLayer.__init__
proj_size = num_heads * head_size                              # 4 × 8 = 32
self.wq = torch.randn((hidden_size, proj_size), ...) * scale   # [32, 32]
self.wo = torch.randn((proj_size, hidden_size), ...) * scale   # [32, 32]
```

demo 中 `hidden_size = 32`，每个 token 是一个 32 维向量。

**`num_heads`：把向量切成几份做独立 attention**

这里需要先理解单头 attention 和多头 attention 的区别。

**单头 attention** 用整个 `hidden_size` 维向量做一次 attention：

```python
import torch, math

def single_head_attention(x: torch.Tensor) -> torch.Tensor:
    """
    x: [seq_len, hidden_size]，比如 [10, 32]
    """
    seq_len, hidden_size = x.shape

    # 投影：hidden_size → hidden_size，一整条向量
    Wq = torch.randn(hidden_size, hidden_size) * 0.1
    Wk = torch.randn(hidden_size, hidden_size) * 0.1
    Wv = torch.randn(hidden_size, hidden_size) * 0.1

    Q = torch.matmul(x, Wq)   # [seq_len, hidden_size] = [10, 32]
    K = torch.matmul(x, Wk)   # [seq_len, hidden_size] = [10, 32]
    V = torch.matmul(x, Wv)   # [seq_len, hidden_size] = [10, 32]

    # attention: 用完整的 32 维向量算点积 → 只产生一组注意力权重
    score = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(hidden_size)  # [10, 10]

    # causal mask: token i 只能看到 token 0..i，未来位置填 -inf
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    score = score.masked_fill(causal_mask, float('-inf'))

    prob = torch.softmax(score, dim=-1)          # [10, 10]
    out = torch.matmul(prob, V)                    # [10, 32]
    return out
```

问题在于：32 维向量只产生**一组**注意力权重，只能学到一种"关注模式"——要么关注语法关系，要么关注语义相似度，没法同时兼顾。

**多头 attention** 把向量切成 `num_heads` 份，每份独立做 attention：

```python
def multi_head_attention(x: torch.Tensor, num_heads: int = 4) -> torch.Tensor:
    """
    x: [seq_len, hidden_size]，比如 [10, 32]
    num_heads = 4 → head_size = 32 / 4 = 8
    """
    seq_len, hidden_size = x.shape
    head_size = hidden_size // num_heads  # 32 // 4 = 8

    Wq = torch.randn(hidden_size, hidden_size) * 0.1
    Wk = torch.randn(hidden_size, hidden_size) * 0.1
    Wv = torch.randn(hidden_size, hidden_size) * 0.1
    Wo = torch.randn(hidden_size, hidden_size) * 0.1

    Q = torch.matmul(x, Wq)   # [10, 32]
    K = torch.matmul(x, Wk)   # [10, 32]
    V = torch.matmul(x, Wv)   # [10, 32]

    # 关键步骤：reshape 成 [seq_len, num_heads, head_size]，再转成 [num_heads, seq_len, head_size]
    Q = Q.reshape(seq_len, num_heads, head_size).transpose(0, 1)  # [4, 10, 8]
    K = K.reshape(seq_len, num_heads, head_size).transpose(0, 1)  # [4, 10, 8]
    V = V.reshape(seq_len, num_heads, head_size).transpose(0, 1)  # [4, 10, 8]

    # 每个 head 独立算 attention（用 8 维向量，而不是 32 维）
    score = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(head_size)  # [4, 10, 10]

    # causal mask: token i 只能看到 token 0..i，未来位置填 -inf
    # unsqueeze(0) 让 [seq_len, seq_len] 广播到 [num_heads, seq_len, seq_len]
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    score = score.masked_fill(causal_mask.unsqueeze(0), float('-inf'))

    prob = torch.softmax(score, dim=-1)                               # [4, 10, 10]
    head_out = torch.bmm(prob, V)                                     # [4, 10, 8]

    # 拼回去
    out = head_out.transpose(0, 1).reshape(seq_len, hidden_size)  # [10, 32]
    out = torch.matmul(out, Wo)                                    # [10, 32]
    return out
```

两者的核心区别：

| | 单头 | 多头（`num_heads=4`） |
|---|---|---|
| Q/K/V 做点积的维度 | 32（整个 hidden） | 8（`head_size = hidden_size / num_heads`） |
| 产生几组注意力权重 | 1 组 `[10, 10]` | 4 组 `[10, 10]` |
| 能学到的关注模式 | 1 种 | 4 种（语法、语义、位置…各学各的） |
| 计算量 | 基本相同 | 基本相同（总维度没变） |
| matmul 次数 | 5 次 | 6 次 |

**两个函数各有几次矩阵乘法**

逐行数代码中的 `torch.matmul` / `torch.bmm`，可以精确统计：

**`single_head_attention`：5 次**

| # | 代码 | 作用 | 形状变化 |
|---|---|---|---|
| 1 | `torch.matmul(x, Wq)` | 投影得 Q | `[10,32] × [32,32] → [10,32]` |
| 2 | `torch.matmul(x, Wk)` | 投影得 K | `[10,32] × [32,32] → [10,32]` |
| 3 | `torch.matmul(x, Wv)` | 投影得 V | `[10,32] × [32,32] → [10,32]` |
| 4 | `torch.matmul(Q, K.transpose(0,1))` | 算注意力分数 | `[10,32] × [32,10] → [10,10]` |
| 5 | `torch.matmul(prob, V)` | 加权求和 | `[10,10] × [10,32] → [10,32]` |

没有输出投影矩阵 $W_o$，所以比多头少一次。

**`multi_head_attention`：6 次**

| # | 代码 | 作用 | 形状变化 |
|---|---|---|---|
| 1 | `torch.matmul(x, Wq)` | 投影得 Q | `[10,32] × [32,32] → [10,32]` |
| 2 | `torch.matmul(x, Wk)` | 投影得 K | `[10,32] × [32,32] → [10,32]` |
| 3 | `torch.matmul(x, Wv)` | 投影得 V | `[10,32] × [32,32] → [10,32]` |
| 4 | `torch.bmm(Q, K.transpose(1,2))` | 每个 head 算分数 | `[4,10,8] × [4,8,10] → [4,10,10]` |
| 5 | `torch.bmm(prob, V)` | 每个 head 加权求和 | `[4,10,10] × [4,10,8] → [4,10,8]` |
| 6 | `torch.matmul(out, Wo)` | 输出投影 | `[10,32] × [32,32] → [10,32]` |

多出的第 6 次就是 $W_o$。多头 attention 需要 $W_o$ 把拼接后的多头输出重新混合——如果不加这一步，各 head 的结果只是简单拼接，head 之间无法交换信息。单头不需要这一步，因为只有一个 head，拼接就是自身。

**计算量（FLOPs）对比**

统一符号：$S$ = seq_len, $H$ = hidden_size, $N$ = num_heads, $D$ = head_size = $H/N$。一次矩阵乘 $[M, K] \times [K, N]$ 的 FLOPs 为 $2MKN$。

投影阶段：

| | 单头 | 多头 | 说明 |
|---|---|---|---|
| $W_q, W_k, W_v$ | $3 \times 2SH^2 = 6SH^2$ | $6SH^2$ | 相同 |
| $W_o$ | 无 | $2SH^2$ | 多头多一次 |
| **投影小计** | $6SH^2$ | $8SH^2$ | |

Attention 阶段：

- 单头用完整 $H$ 维做点积：$QK^T$ 为 $2S^2H$，$\text{prob} \cdot V$ 为 $2S^2H$，合计 $4S^2H$
- 多头做 $N$ 次、每次 $D$ 维：$QK^T$ 为 $N \times 2S^2D = 2S^2H$，$\text{prob} \cdot V$ 为 $2S^2H$，合计 $4S^2H$

**Attention 部分的 FLOPs 完全相同**——因为 $N \times D = H$，切分只是重新分配维度，总计算量不变。

总计对比：

| | 投影 | Attention | **总计** |
|---|---|---|---|
| 单头 | $6SH^2$ | $4S^2H$ | $6SH^2 + 4S^2H$ |
| 多头 | $8SH^2$ | $4S^2H$ | $8SH^2 + 4S^2H$ |
| **差额** | $+2SH^2$ | $0$ | $+2SH^2$ |

多头比单头多出的计算量**只有 $W_o$ 这一次矩阵乘法**。

用 demo 参数代入（$S=10, H=32$）：单头 74240 FLOPs，多头 94720 FLOPs，多头多约 28%，全部来自 $W_o$。

实际模型中哪项占主导取决于 $S$ 和 $H$ 的关系：

| 场景 | 条件 | 主导项 | 单头 vs 多头 |
|---|---|---|---|
| 短序列 | $S \ll H$（如 LLaMA-7B: $H=4096, S=128$） | 投影 $SH^2$ | 多头多 $\frac{8}{6} \approx 1.33\times$ 投影开销 |
| 长序列 | $S \gg H$（如 $S=32768$） | Attention $S^2H$ | **几乎没有差别** |
| 均衡点 | $S \approx H$ | 两项相当 | 多头多约 20-25% |

**内存和带宽对比**

*参数内存（权重）*

单头 $3H^2$（$W_q, W_k, W_v$），多头 $4H^2$（多一个 $W_o$），多 33%。用 demo 参数（$H=32$, float32）：单头 12 KB，多头 16 KB。实际模型如 LLaMA-7B（$H=4096$）：单头每层约 192 MB，多头每层约 256 MB。

*中间激活内存（推理时峰值）*

关键差异在 score/prob 矩阵：

| 张量 | 单头 | 多头 |
|---|---|---|
| Q, K, V | $3SH$ | $3SH$（总量不变） |
| score + prob | $2S^2$ | $2NS^2$ |
| 输出 | $SH$ | $2SH$（head_out + 拼接后 out） |
| **合计** | $4SH + 2S^2$ | $5SH + 2NS^2$ |

多头的注意力矩阵内存是单头的 $N$ 倍。以 LLaMA-7B（$N=32, S=4096$）为例：

$$\text{单头 score+prob}: 2 \times 4096^2 \times 4\text{B} \approx 128\text{MB}$$

$$\text{多头 score+prob}: 32 \times 128\text{MB} \approx 4\text{GB}$$

这也是 FlashAttention 要做**分块计算**的直接原因——避免一次性实例化完整的 $[N, S, S]$ 注意力矩阵。

*内存带宽*

投影阶段的带宽（读权重 + 读写激活）：单头 $3H^2 + 6SH$，多头 $4H^2 + 8SH$，多头多 33%。Attention 阶段的带宽（读写 Q/K/V/score/prob/out）：单头 $4SH + 2S^2$，多头 $4SH + 2NS^2$，差异同样来自 $N$ 倍的注意力矩阵。

*与 KV cache 的关系*

每个 token 的 KV cache 大小：单头存 $[H]$，多头存 $[N, D]$，但 $N \times D = H$，所以 **KV cache 大小完全相同**。但多头在 decode 阶段有一个带宽优势：每个 head 的 attention 可以独立流水线化，GPU 可以在计算 head 0 时预取 head 1 的 KV 数据。单头只有一大块 $[L, H]$ 的 KV，无法拆分流水。

*全维度汇总*

| 维度 | 单头 vs 多头 | 根本原因 |
|---|---|---|
| 权重内存 | 多头多 33% | 多一个 $W_o$ |
| 注意力矩阵内存 | 多头多 $N$ 倍 | $[S,S]$ → $[N,S,S]$ |
| KV cache 大小 | **完全相同** | 投影后总维度都是 $H$ |
| 投影带宽 | 多头多 33% | 多读写一次 $W_o$ |
| Attention 带宽 | 多头多 $N$ 倍（score/prob） | 同注意力矩阵 |
| decode 带宽利用率 | **多头更优** | 多 head 可流水线重叠 |
| 总 FLOPs | 多头仅多 $2SH^2$ | 仅 $W_o$ 一项 |

**核心结论**：多头 attention 用几乎相同的计算量（仅多一个 $W_o$），换来 $N$ 种不同的关注模式。真正的额外代价是注意力矩阵的 $N$ 倍内存膨胀，而 KV cache 大小完全不变。

**在教学项目中的代码体现**

对应 `ToyLayer` 中的 `_split_heads` 和 `_merge_heads`：

```python
def _split_heads(self, x):
    return x.reshape(x.shape[0], self.num_heads, self.head_size)
    # [tokens, 32] → [tokens, 4, 8]

def _merge_heads(self, x):
    return x.reshape(x.shape[0], self.num_heads * self.head_size)
    # [tokens, 4, 8] → [tokens, 32]
```

完整的前向流程是：

$$
[tokens, \underbrace{hidden\_size}_{32}] \xrightarrow{W_q} [tokens, \underbrace{num\_heads \times head\_size}_{4 \times 8 = 32}] \xrightarrow{reshape} [tokens, \underbrace{num\_heads}_{4}, \underbrace{head\_size}_{8}]
$$

**`hidden_size` 与 `num_heads × head_size` 的关系**

在 demo 中 $num\_heads \times head\_size = 4 \times 8 = 32 = hidden\_size$，投影前后维度恰好一样。这是最常见的设计（如 LLaMA-7B：$32 \times 128 = 4096 = hidden\_size$），但不是必须的——有些 GQA/MQA 架构的乘积可能小于 `hidden_size`。

**和 KV cache 的关联**

`hidden_size` **不出现在 cache shape 中**。cache 只存投影之后的 head 空间：

```
cache: [num_blocks, num_heads, block_size, head_size]
                    ↑                       ↑
              head 空间的两个维度，与 hidden_size 无关
```

`hidden_size` 只存在于投影权重矩阵里，是 attention 的"上游接口"。所以 `hidden_size` 是"对外接口宽度"，`num_heads` 是"内部并行 attention 的路数"，`head_size` 是"每路 attention 的向量长度"。

### 3.5 `num_blocks` 和 `block_size` 的含义

这两个参数共同定义了 KV cache 的"分页内存模型"：

- `num_blocks`：整个系统可用的物理 KV block 总数
- `block_size`：每个物理 block 最多能存多少个 token 的 KV

两者合起来决定 KV cache 的总容量上限：

$$
max\_tokens\_in\_cache = num\_blocks \times block\_size
$$

以 demo 参数为例，`num_blocks = 64`，`block_size = 4`，总容量是 $64 \times 4 = 256$ 个 token 位置（所有活跃序列合计）。

**`block_size` 如何影响地址映射**

某个 token 的逻辑位置 `logical_pos` 映射为：

$$
逻辑块号 = logical\_pos \;/\; block\_size
$$

$$
块内偏移 = logical\_pos \;\%\; block\_size
$$

例如 `block_size = 4` 时：

- token 0,1,2,3 在第 0 个逻辑块
- token 4,5,6,7 在第 1 个逻辑块
- token 8,9,10,11 在第 2 个逻辑块

**`num_blocks` 如何约束调度**

`num_blocks = 64` 表示调度器一开始有物理 block `0..63` 可用。随着序列增长：

- 新序列 prefill 时分一些 block
- decode 时如果尾块写满，再追加新 block
- beam fork 时可以共享已有 block（不消耗新 block）
- finish 时把不再使用的 block 放回 free list

当 free list 耗尽时，无法再接纳新 token，系统会报错。

**block_size 越大越好吗**

不一定。`block_size` 越大：

- 好处：需要的 block 更少，调度开销更低
- 坏处：尾块浪费更多（比如序列只有 1 个 token 也要占一整个 block）
- 坏处：copy-on-write 时要复制的数据量更大

`block_size` 越小则相反。生产实现通常取 16 或 32 作为折衷。

**这四个参数的性质分类**

`num_blocks`、`block_size`、`num_heads`、`head_size` 虽然都出现在 cache shape `[num_blocks, num_heads, block_size, head_size]` 中，但它们的来源完全不同：

| 参数 | 性质 | 谁决定 | 能否运行时调整 |
|---|---|---|---|
| `num_heads` | 模型架构参数 | 模型设计者 | 不能，训练时固定 |
| `head_size` | 模型架构参数 | 模型设计者 | 不能，训练时固定 |
| `num_blocks` | 运行时部署参数 | 部署工程师 | 能，取决于可用内存 |
| `block_size` | 运行时部署参数 | 部署工程师 | 能，是性能调优旋钮 |

换句话说：

- `num_heads` 和 `head_size` 是"模型说了算"的，加载一个预训练模型后就不能改。例如 LLaMA-7B 的 `num_heads = 32`，`head_size = 128`，这是模型权重决定的。
- `num_blocks` 和 `block_size` 是"部署环境说了算"的。同一个模型，在 GPU 显存多的机器上可以设更大的 `num_blocks`；`block_size` 则根据吞吐量和内存碎片做权衡。

所以 cache 形状可以理解为：

$$
[\underbrace{num\_blocks}_{运行时} \;,\; \underbrace{num\_heads}_{模型} \;,\; \underbrace{block\_size}_{运行时} \;,\; \underbrace{head\_size}_{模型}]
$$

前两个维度由运行时和模型各提供一个，后两个维度也如此。这也是为什么调度层只需要知道 `num_blocks` 和 `block_size`，不需要知道 `num_heads` 和 `head_size`——后者是执行层内部的事。

### 3.6 物理 block 的数据类型和内存布局

理解调度层的 `num_blocks` / `block_size` 之后，下面看执行层如何用它们构建真实的 KV cache 张量。

#### 3.6.1 整体 cache 形状

在 `PagedAttentionExecutor.__init__` 里，cache 是一个四维张量：

```python
cache_shape = (num_blocks, num_heads, block_size, head_size)
```

四个维度的含义：

| 维度 | 名称 | 含义 |
|---:|---|---|
| 0 | `num_blocks` | 第几个物理 block |
| 1 | `num_heads` | 第几个 attention head |
| 2 | `block_size` | block 内第几个 token 槽位（offset） |
| 3 | `head_size` | 该 head 的向量维度 |

用 demo 参数代入：

```
k_cache shape = [64, 4, 4, 8]
v_cache shape = [64, 4, 4, 8]
```

即 64 个物理 block，每个 block 有 4 个 head，每个 head 有 4 个 token 槽位，每个槽位存一个长度为 8 的向量。

#### 3.6.2 单个 block 的内部结构

用 `cache[block_id]` 取出一个 block 时，得到一个三维张量：

$$
[num\_heads, block\_size, head\_size] = [4, 4, 8]
$$

画成表格（以 K cache 为例）：

```
block N:
         offset 0       offset 1       offset 2       offset 3
head 0:  [d0..d7]       [d0..d7]       [d0..d7]       [d0..d7]
head 1:  [d0..d7]       [d0..d7]       [d0..d7]       [d0..d7]
head 2:  [d0..d7]       [d0..d7]       [d0..d7]       [d0..d7]
head 3:  [d0..d7]       [d0..d7]       [d0..d7]       [d0..d7]
```

每个 `[d0..d7]` 是一个 token 在某个 head 上的 K（或 V）向量，长度为 `head_size = 8`。

#### 3.6.3 float 模式 vs int8 模式

**float 模式**（`use_int8_cache = False`）：

```
k_cache: torch.float32, shape [num_blocks, num_heads, block_size, head_size]
v_cache: torch.float32, shape [num_blocks, num_heads, block_size, head_size]
```

每个元素是原始 float32 值，没有额外结构。

**int8 模式**（`use_int8_cache = True`，demo 默认开启）：

```
k_cache: torch.int8,    shape [num_blocks, num_heads, block_size, head_size]
v_cache: torch.int8,    shape [num_blocks, num_heads, block_size, head_size]
k_scale: torch.float32, shape [num_blocks, num_heads, block_size]
v_scale: torch.float32, shape [num_blocks, num_heads, block_size]
```

注意 scale 比 cache 少一个维度。这是因为量化粒度是**逐 token、逐 head**：

- 一个 token 在一个 head 上的向量（长度 `head_size`）共享同一个 scale
- 同一个 block 里不同 token、不同 head 的 scale 各不相同

#### 3.6.4 写入路径

以写入某个 token 为例，追踪 `_write_one_token`：

```
输入: slot = physical_block * block_size + offset
      k: shape [num_heads, head_size]    # 该 token 在所有 head 上的 K 向量
      v: shape [num_heads, head_size]    # 该 token 在所有 head 上的 V 向量
```

int8 模式下的写入过程：

1. 从 slot 算出 block 和 offset
2. 对 k 做逐 head 量化，得到 `qk: [num_heads, head_size] int8`，`sk: [num_heads] float32`
3. 写入 cache：`k_cache[block, :, offset, :] = qk`，`k_scale[block, :, offset] = sk`

用具体数字看，假设写入 seq=100 的 token 2，它的 slot = 2（物理 block 0，offset 2）：

```
k_cache[0, :, 2, :] ← int8 量化后的 K 向量，shape [4, 8]
k_scale[0, :, 2]    ← 4 个 head 各自的 scale，shape [4]
```

写完之后，物理 block 0 在 K cache 中的状态（值为示意）：

```
block 0:
         offset 0          offset 1          offset 2          offset 3
head 0:  [int8 × 8] s=0.03 [int8 × 8] s=0.05 [int8 × 8] s=0.04 [空]
head 1:  [int8 × 8] s=0.02 [int8 × 8] s=0.06 [int8 × 8] s=0.03 [空]
head 2:  [int8 × 8] s=0.04 [int8 × 8] s=0.03 [int8 × 8] s=0.05 [空]
head 3:  [int8 × 8] s=0.01 [int8 × 8] s=0.04 [int8 × 8] s=0.02 [空]
```

#### 3.6.5 读取路径

追踪 `_read_token_kv`：

```
输入: block, offset
输出: k [num_heads, head_size] float32
      v [num_heads, head_size] float32
```

int8 模式下：

$$
k_{h,d} = q_{h,d} \times scale_h
$$

每个 head 用自己的 scale 恢复出 float32 向量。

#### 3.6.6 block copy 的搬运量

追踪 `copy_block`：

```python
self.k_cache[dst_block].copy_(self.k_cache[src_block])
self.v_cache[dst_block].copy_(self.v_cache[src_block])
self.k_scale[dst_block].copy_(self.k_scale[src_block])
self.v_scale[dst_block].copy_(self.v_scale[src_block])
```

一次 block copy 搬运的数据量（demo 参数，int8 模式）：

| 数据 | 形状 | 类型 | 大小 |
|---|---|---|---|
| `k_cache[block]` | [4, 4, 8] | int8 | 128 B |
| `v_cache[block]` | [4, 4, 8] | int8 | 128 B |
| `k_scale[block]` | [4, 4] | float32 | 64 B |
| `v_scale[block]` | [4, 4] | float32 | 64 B |
| **合计** | | | **384 B** |

如果用 float 模式（无 scale，cache 为 float32）：

| 数据 | 形状 | 类型 | 大小 |
|---|---|---|---|
| `k_cache[block]` | [4, 4, 8] | float32 | 512 B |
| `v_cache[block]` | [4, 4, 8] | float32 | 512 B |
| **合计** | | | **1024 B** |

int8 模式省了约 2.7 倍空间，代价是量化误差。

这也解释了为什么 copy-on-write 的代价比复制整个序列历史要小得多——只复制一个 block 的数据。

#### 3.6.7 维度顺序为什么是 [block, head, offset, dim]

这个布局的好处是：

1. **block copy 最高效**：`cache[dst].copy_(cache[src])` 按第 0 维切出连续内存块，一次 memcpy
2. **单 token 读取对 attention 友好**：`cache[block, :, offset, :]` 取出所有 head 在某个 token 位置的向量，形状是 `[num_heads, head_size]`，正好是 attention 计算所需的粒度
3. **head 维度连续**：同一个 block 内，同一个 head 的所有 offset 在内存里是连续的，有利于按 head 做批量操作

如果把维度顺序改成 `[block, offset, head, dim]`，block copy 效率一样，但 `cache[block, :, offset, :]` 读取时 head 维度就不连续了，对 attention 计算不友好。
## 4. Prefill 流程详解

Prefill 指的是把 prompt 整段送入模型，第一次建立 KV cache。

以 batch 中两个序列为例，prefill 的完整流程是：

### 4.1 调度层预留 block

`ToyLLMRuntime.prefill()` 会先调用：

1. `KVBlockManager.reserve_for_prefill(seq_id, q_len)`

这里会做两件事：

1. 如果当前序列尾块是共享的且本轮要写新 token，则检查是否需要 copy-on-write
2. 确保追加这批 token 后，逻辑 block 数足够

对于首次 prefill，通常不存在共享尾块，所以主要是按 `q_len` 申请足够多的 block。

### 4.2 生成 copy 计划并应用到所有 layer

如果某个序列在 prefill 前就已经共享尾块，这里会先得到若干 `BlockCopyPlan`。

`ToyLLMRuntime` 会调用：

1. `layer.copy_block(src_block, dst_block)`

并对所有 layer 执行一次。

这样做的原因是：

1. 调度层的映射变了
2. 每层真实 KV 数据也必须同步变过去
3. 否则元数据和物理 cache 会失配

### 4.3 构建 metadata

调度层通过 `build_batch_metadata(seq_ids, q_lens)` 构建出本轮执行所需元数据。

这一步做的不是数值计算，而是把如下信息编码好：

1. 每个序列之前有多长历史
2. 当前 batch 中每个序列写多少 token
3. 每个序列最终会使用哪些物理 block
4. 这些 block 在扁平数组中的切片位置

### 4.4 每层执行 QKV 投影

`ToyLayer.forward_prefill()` 里先把输入 hidden state 投影成：

1. `Q`
2. `K`
3. `V`

PyTorch 版用矩阵乘法：

$$
Q = XW_q, \quad K = XW_k, \quad V = XW_v
$$

然后通过 reshape 把二维张量拆成三维张量：

$$
[tokens, hidden] \rightarrow [tokens, heads, head\_size]
$$

### 4.5 把新 K/V 写入 paged cache

执行层先根据 `build_slot_mapping()` 得到每个新 token 对应的物理 slot。

slot 的定义可以写成：

$$
slot = physical\_block \times block\_size + offset
$$

对于每个 token：

1. 先确定它在逻辑序列中的位置 `logical_pos = past_len + j`
2. 计算它属于第几个逻辑 block
3. 再通过 `block_indices` 找到物理 block
4. 最后把 `K/V` 写入对应位置

### 4.6 执行 attention

写完本轮 K/V 后，才能做 attention，因为当前 token 也属于可见上下文的一部分。

执行过程是：

1. 通过 `collect_context_positions()` 拿到历史上下文所有 `(block, offset)`
2. 从 cache 中把对应的 K/V 读出来
3. 对每个 query token 按因果掩码范围计算分数
4. 对分数做 softmax
5. 用 softmax 权重对 V 做加权求和

数学上，对某个 head 的某个 query，有：

$$
score_i = \frac{q \cdot k_i}{\sqrt{d}}
$$

$$
prob_i = softmax(score)_i
$$

$$
out = \sum_i prob_i v_i
$$

其中 $d = head\_size$。

### 4.7 提交 past_len

所有 layer 都执行完以后，运行时再统一调用：

1. `commit_tokens(seq_id, q_len)`

这一步非常重要，因为它决定下一轮 decode 的逻辑写入位置。

## 5. Decode 流程详解

Decode 表示每个活跃序列只追加一个新 token。

它与 prefill 的区别不是“完全不同的算法”，而是：

1. 每个序列的 `q_len` 固定为 1
2. 只为一个新 token 做 query
3. 但 K/V 读取范围是全部历史加当前 token

流程如下：

### 5.1 为每个序列预留一个新位置

`reserve_for_decode(seq_id)` 本质上就是 `reserve_for_prefill(seq_id, 1)`。

这里可能发生两种事：

1. 如果当前尾块还有空位且没共享，就直接写进去
2. 如果尾块共享且没写满，就先 copy-on-write
3. 如果尾块已满，就追加一个新 block

### 5.2 重新构建 metadata

因为 block table 可能因为扩容或 copy-on-write 而变化，所以 decode 每一步都必须重新构建 metadata。

这也是 paged attention 的一个核心事实：

1. metadata 是运行时状态的快照
2. 每一轮执行都应该使用当前最新的快照

### 5.3 只写一个新 token 的 K/V

对 batch 中每个序列，当前步只会写一个 token，因此 `q_lens = [1, 1, ..., 1]`。

写入路径和 prefill 一样，只是 token 数少很多。

### 5.4 读取完整上下文并计算单 token attention

对某个序列，本轮 decode 的 query 只有一个 token，但它需要看到：

1. 全部历史 token
2. 当前这个新 token

所以总上下文长度是：

$$
total\_kv\_len = past\_len + 1
$$

然后执行和 prefill 相同的 attention 公式。

### 5.5 提交一个新 token

执行完成后，统一把每个序列的 `past_len` 加 1。

## 6. Beam Fork、Copy-on-Write、Merge、Finish 的设计原理

这部分是整个示例最有教学价值的地方，因为它体现了 paged attention 相比“每序列一块连续 buffer”更灵活的本质。

### 6.1 为什么 beam fork 可以共享 block

在 beam search 中，多个候选 beam 往往共享相同的前缀。

例如：

1. beam A 已经生成了 5 个 token
2. 现在分叉出 beam B
3. 在分叉那一刻，A 和 B 的历史完全相同

如果此时复制整段 KV，成本很高，也没有必要。

所以 `fork_sequence(parent, child)` 采用的策略是：

1. child 直接复制 parent 的 `logical_blocks` 列表
2. 对这些物理 block 的引用计数加 1

这意味着：

1. 逻辑上是两个独立序列
2. 物理上暂时共享同一份历史 KV

### 6.2 为什么需要 copy-on-write

共享只适用于“读”，不适用于“其中一个 beam 要继续写”。

如果两个 beam 共享一个未写满的尾块，而其中一个 beam 追加新 token，直接写会污染另一个 beam 的历史。

所以在 `_ensure_writable_tail()` 里会检查：

1. 当前序列是否已有历史
2. 尾块是否没写满
3. 尾块是否被多个序列共享

如果三者都满足，就：

1. 分配一个新块
2. 把当前序列的尾块映射改成新块
3. 给旧块减引用计数
4. 返回 `BlockCopyPlan(old, new)`

随后所有 layer 都会把旧块数据复制到新块。这样新序列就拿到一份独立、但内容一致的尾块副本。

### 6.3 为什么 merge 只需要改映射

`beam_merge(dst, src)` 的语义是：

1. 目标 beam 不再保留原来的假设
2. 它改为指向另一个胜出 beam 的历史

这里不需要搬运 KV 数据，因为源 beam 已经拥有完整有效的 block 列表。做法是：

1. 先释放 dst 当前持有的 block
2. 再把 dst 的 `logical_blocks` 和 `past_len` 改成 src 的内容
3. 对 src 对应的物理 block 增加引用计数

这本质上是一次“逻辑重绑定”，而不是数据重排。

### 6.4 finish 为什么可以做到精确回收

当一个序列结束时，运行时调用 `finish_sequence(seq_id)`：

1. 找到该序列当前使用的所有 block
2. 对每个 block 做引用计数减 1
3. 若某个 block 的引用计数降到 0，则放回 free list

因此：

1. 独占 block 会被立刻回收
2. 共享 block 只有在最后一个引用者离开时才会回收

这让 block 的生命周期和真实使用关系保持一致。

## 7. 执行层为什么要和调度层解耦

这是这个项目最值得保留的设计点之一。

调度层不应该知道：

1. K/V 是 float 还是 int8
2. cache 是连续张量还是别的物理布局
3. attention 核怎么实现

它只需要知道：

1. 某个序列下一步需要多少 token 空间
2. 哪个 block 被谁持有
3. 哪些 block 需要 copy-on-write

反过来，执行层也不应该关心：

1. 为什么要复制某个 block
2. beam 是怎么调度出来的
3. 哪个候选假设赢了 merge

它只需要知道：

1. 本轮有哪些序列
2. 每个序列历史长度是多少
3. 每个逻辑位置映射到哪些物理 block
4. 是否需要执行某些 block copy

这种拆分的好处是：

1. 调度策略可以独立演化
2. attention 内核可以独立替换
3. 同一套调度逻辑可以复用到不同实现语言或不同后端

这也是为什么这个项目同时保留 Python 和 C++ 版本有教学价值。

## 8. int8 KV cache 压缩为什么放在执行层

压缩属于“物理存储格式”的范畴，而不是“逻辑调度策略”的范畴。

调度层只关心某个 token 的 K/V 最终要落到哪个 block 的哪个 offset，不关心这个位置里存的是：

1. fp32
2. fp16
3. int8 + scale

因此压缩逻辑应当放在执行层：

1. 写入时决定如何量化
2. 读取时决定如何反量化
3. block copy 时决定是否连 scale 一起复制

这个教学版本使用的是简化的逐 token、逐 head 对称量化：

$$
scale = \frac{\max(|x|)}{127}
$$

$$
q = round(x / scale)
$$

$$
dequant(x) = q \cdot scale
$$

真实工程里通常会更复杂，例如：

1. 更细粒度的分组
2. 更复杂的 scale 布局
3. 融合反量化和 attention kernel

但这个简化版本已经足够说明“压缩缓存”和“逻辑分页”是两个正交问题。

## 9. Python PyTorch 版本与 C++ 版本如何对应

两个版本保留的是同一套架构，而不是逐行一一对应。

它们的共同点是：

1. 都有相同的调度层结构
2. 都有相同的 metadata 设计
3. 都有相同的 prefill / decode / fork / merge / finish 生命周期
4. 都有每层独立 KV cache、全局共享调度元数据的设计

它们的差异主要在实现手段：

1. Python 版用 PyTorch tensor 做矩阵乘和 attention
2. C++ 版用标准库容器和显式循环表达同样的逻辑

因此：

1. Python 版更适合快速理解整体流程
2. C++ 版更适合理解底层数据搬运和显式内存布局

## 10. 建议的阅读顺序

如果你是第一次接触这份代码，推荐按这个顺序读：

1. 先读 [README.md](../README.md)，建立整体目标
2. 再读 [manual_walkthrough.md](manual_walkthrough.md)，理解 block 如何随着时间变化
3. 再读 [../python/standalone_pa.py](../python/standalone_pa.py)，理解教学主线
4. 最后读 [../cpp/standalone_pa.cpp](../cpp/standalone_pa.cpp)，对照系统语言实现

如果你读代码时容易迷失，可以始终抓住三个问题：

1. 当前这个函数是在改“逻辑映射”，还是在改“物理数据”
2. 当前这个函数操作的是“序列级状态”，还是“单层 attention cache”
3. 当前这个阶段是在“预留空间”，还是在“真正执行 attention”

抓住这三个问题，整套 paged attention 教学实现就不容易混乱。