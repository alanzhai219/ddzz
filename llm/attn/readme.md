# Attention Shapes Notes

本文档总结当前目录中注意力实现的常用术语、张量 shape 和各维度含义。

相关代码：
- [self-head-attn.py](self-head-attn.py)
- [multi-head-attn.py](multi-head-attn.py)

## 1. 常见术语

| 术语 | 常见别名 | 含义 |
|---|---|---|
| embed_dim | d_model, hidden_size | token 的总特征维度 |
| num_heads | n_heads | 多头注意力中的头数 |
| head_dim | head_size | 每个 head 的特征维度 |

多头关系式：

$$
embed\_dim = num\_heads \times head\_dim
$$

---

## 2. Single-Head Attention（self-head-attn.py）

设输入为 `x`，shape 是 `[B, S, D]`。

- `B`: batch size
- `S`: sequence length
- `D`: embed_dim

### 2.1 前向过程 shape 总表

| 步骤 | 表达式 | 输出 shape | 含义 |
|---|---|---|---|
| 输入 | `x` | `[B, S, D]` | 每个 token 一个 `D` 维向量 |
| 线性投影 Q | `q_proj(x)` | `[B, S, D]` | query 向量 |
| 线性投影 K | `k_proj(x)` | `[B, S, D]` | key 向量 |
| 线性投影 V | `v_proj(x)` | `[B, S, D]` | value 向量 |
| 打分矩阵 | `q @ k^T / sqrt(D)` | `[B, S, S]` | 每个 query 位置对每个 key 位置的相关性 |
| 权重归一化 | `softmax(scores, dim=-1)` | `[B, S, S]` | 每行变成概率分布 |
| 上下文聚合 | `attn @ v` | `[B, S, D]` | 按权重对 value 加权求和 |
| 输出映射 | `out_proj(context)` | `[B, S, D]` | 输出特征 |

### 2.2 `scores[b, i, j]` 的语义

| 索引维度 | 含义 |
|---|---|
| `b` | 第 `b` 个样本 |
| `i` | query 位置（当前要更新的 token 位置） |
| `j` | key 位置（被参考的 token 位置） |

解释：`scores[b, i, j]` 越大，表示第 `i` 个 token 越关注第 `j` 个 token。

### 2.3 Q/K/V 参数（权重）shape

在 `self-head-attn.py` 里，`q_proj/k_proj/v_proj` 都是 `nn.Linear(embed_dim, embed_dim)`：

| 模块 | 参数符号 | weight shape | bias shape |
|---|---|---|---|
| q_proj | Wq | `[D, D]` | `[D]` |
| k_proj | Wk | `[D, D]` | `[D]` |
| v_proj | Wv | `[D, D]` | `[D]` |
| out_proj | Wo | `[D, D]` | `[D]` |

说明：PyTorch 的 `nn.Linear(in_features, out_features)` 中，参数 `weight` 形状是 `[out_features, in_features]`。

### 2.4 Mask shape 与语义

| Mask | 输入 shape | 广播后作用位置 | True/False 语义 | 作用 |
|---|---|---|---|---|
| causal mask | `[S, S]` | 对 `scores[B, S, S]` 广播 | True: 可见, False: 屏蔽 | 禁止看到未来 token |
| padding mask | `[B, S]` | key 侧变成 `[B, 1, S]`；query 侧变成 `[B, S, 1]` | True: 有效 token, False: padding | 屏蔽 padding key，并将 padding query 的输出清零 |

补充：
- key 侧屏蔽后，padding token 不会被任何位置关注。
- query 侧清零后，padding 位置不会产生无意义输出。

---

## 3. Multi-Head Attention（multi-head-attn.py）

输入同样是 `[B, S, D]`，但会先拆成多头。

### 3.1 多头 shape 总表

| 步骤 | 输出 shape | 含义 |
|---|---|---|
| 输入 | `[B, S, D]` | 总特征维度 `D` |
| Q/K/V 投影 | `[B, S, D]` | 先在线性层中变换 |
| split heads | `[B, H, S, Hd]` | `H=num_heads`, `Hd=head_dim` |
| 注意力分数 | `[B, H, S, S]` | 每个 head 单独算注意力 |
| 与 V 聚合 | `[B, H, S, Hd]` | 每个 head 得到上下文 |
| merge heads | `[B, S, D]` | 拼回总维度 |
| out_proj | `[B, S, D]` | 输出映射 |

### 3.2 Single-Head 与 Multi-Head 对照

| 项目 | Single-Head | Multi-Head |
|---|---|---|
| Q/K/V shape | `[B, S, D]` | split 后为 `[B, H, S, Hd]` |
| 打分矩阵 shape | `[B, S, S]` | `[B, H, S, S]` |
| 缩放因子 | `sqrt(D)` | `sqrt(Hd)` |
| 表达能力 | 单一子空间 | 多个子空间并行建模 |

### 3.3 Multi-Head 中 Q/K/V 参数（权重）shape

在 `multi-head-attn.py` 里，`q_proj/k_proj/v_proj` 同样是 `nn.Linear(embed_dim, embed_dim)`，因此参数 shape 与 single-head 一致：

| 模块 | 参数符号 | weight shape | bias shape |
|---|---|---|---|
| q_proj | Wq | `[D, D]` | `[D]` |
| k_proj | Wk | `[D, D]` | `[D]` |
| v_proj | Wv | `[D, D]` | `[D]` |
| out_proj | Wo | `[D, D]` | `[D]` |

注意：多头并不是通过给每个 head 单独定义一个线性层来实现，而是先用一个大线性层得到 `[B, S, D]`，再 reshape/split 成 `[B, H, S, Hd]`。

---

## 4. Single-Head vs Multi-Head 对比

下面给出在固定 `B,S,D` 下的对比结论（`H=num_heads`, `Hd=D/H`）。

### 4.1 输入和输出对比

| 项目 | Single-Head | Multi-Head |
|---|---|---|
| 输入 | `[B, S, D]` | `[B, S, D]` |
| 输出 | `[B, S, D]` | `[B, S, D]` |
| Q/K/V 线性层输出 | `[B, S, D]` | `[B, S, D]` |
| 注意力核心计算张量 | `[B, S, S]` | `[B, H, S, S]` |

结论：模块输入输出接口相同，主要区别在内部计算张量和并行子空间数量。

### 4.2 计算过程对比

| 阶段 | Single-Head | Multi-Head |
|---|---|---|
| QKV 投影 | 3 次 `D->D` 线性变换 | 相同 |
| 分头/合头 | 无 | 有（`split` + `merge`） |
| 注意力打分 | 一次 `QK^T` | 每个 head 各算一次（并行） |
| softmax | 在 `[B,S,S]` 上 | 在 `[B,H,S,S]` 的最后一维上 |
| 与 V 聚合 | 一次 `attn @ V` | 每个 head 各算一次后再合并 |

结论：multi-head 比 single-head 多了分头和合头步骤，但核心仍是缩放点积注意力。

### 4.3 计算量（粗略 FLOPs）对比

设忽略常数和低阶项，关注主导项：

| 组成 | Single-Head | Multi-Head |
|---|---|---|
| QKV + Out 投影 | $O(BSD^2)$ | $O(BSD^2)$ |
| 注意力打分 `QK^T` | $O(BS^2D)$ | $O(BHS^2Hd)=O(BS^2D)$ |
| 注意力乘 V | $O(BS^2D)$ | $O(BHS^2Hd)=O(BS^2D)$ |
| 总体量级 | $O(BSD^2 + BS^2D)$ | $O(BSD^2 + BS^2D)$ |

结论：在固定 `D` 下，两者理论总 FLOPs 同阶，multi-head 不是“更高阶复杂度”，而是把通道维计算拆到多个 head。

### 4.4 内存与带宽开销对比

| 维度 | Single-Head | Multi-Head | 说明 |
|---|---|---|---|
| 参数量 | 约 `4*(D*D + D)` | 约 `4*(D*D + D)` | 两个实现都用 4 个 `D->D` 线性层 |
| 激活主峰值 | `scores/attn` 为 `[B,S,S]` | `scores/attn` 为 `[B,H,S,S]` | multi-head 在实现层面更容易显式持有更大注意力张量 |
| 中间重排开销 | 低 | 更高 | 需要 `view/transpose/contiguous` 触发更多访存 |
| 带宽敏感性 | 中 | 更高 | head 维引入更频繁的张量重排和读写 |

补充说明：
- 理论上，若只按元素总数看，`[B,H,S,S]` 与把 `D` 切分前后的主计算总量同阶；
- 工程上，multi-head 往往更依赖高效 kernel（如 fused attention）来降低显存与带宽压力。

---

## 5. 实战检查清单

| 检查项 | 建议 |
|---|---|
| 维度匹配 | 输入最后一维应等于 `embed_dim` |
| 多头整除 | `embed_dim % num_heads == 0` |
| mask 类型 | `padding_mask` 用 `bool`，语义固定为 True=有效 |
| causal 开关 | 自回归任务开启，双向编码任务关闭 |

---

## 6. 代码实现（Single-Head / Multi-Head）

### 6.1 Single-Head Attention（推理版）

```python
class SingleHeadAttentionInference(nn.Module):
	def __init__(self, embed_dim: int, bias: bool = True) -> None:
		super().__init__()
		self.embed_dim = embed_dim
		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

	@torch.inference_mode()
	def forward(self, x, causal=False, padding_mask=None):
		q = self.q_proj(x)                     # [B,S,D]
		k = self.k_proj(x)                     # [B,S,D]
		v = self.v_proj(x)                     # [B,S,D]
		scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.embed_dim)  # [B,S,S]

		if causal:
			s = x.size(1)
			causal_mask = torch.ones(s, s, device=x.device, dtype=torch.bool).tril()
			scores = scores.masked_fill(~causal_mask, float("-inf"))

		if padding_mask is not None:
			key_mask = padding_mask.unsqueeze(1)   # [B,1,S]
			scores = scores.masked_fill(~key_mask, float("-inf"))

		attn = torch.softmax(scores, dim=-1)       # [B,S,S]

		if padding_mask is not None:
			query_mask = padding_mask.unsqueeze(-1)  # [B,S,1]
			attn = attn * query_mask.to(attn.dtype)

		context = attn @ v                         # [B,S,D]
		out = self.out_proj(context)               # [B,S,D]

		if padding_mask is not None:
			out = out * padding_mask.unsqueeze(-1).to(out.dtype)
		return out
```

### 6.2 Multi-Head Attention（推理版）

```python
class MultiHeadAttentionInference(nn.Module):
	def __init__(self, embed_dim: int, num_heads: int, bias: bool = True) -> None:
		super().__init__()
		if embed_dim % num_heads != 0:
			raise ValueError("embed_dim must be divisible by num_heads")
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.head_dim = embed_dim // num_heads
		self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
		self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

	def _split_heads(self, x):                     # [B,S,D] -> [B,H,S,Hd]
		b, s, _ = x.shape
		return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

	def _merge_heads(self, x):                     # [B,H,S,Hd] -> [B,S,D]
		b, _, s, _ = x.shape
		return x.transpose(1, 2).contiguous().view(b, s, self.embed_dim)

	@torch.inference_mode()
	def forward(self, x, causal=False, padding_mask=None):
		q = self._split_heads(self.q_proj(x))      # [B,H,S,Hd]
		k = self._split_heads(self.k_proj(x))      # [B,H,S,Hd]
		v = self._split_heads(self.v_proj(x))      # [B,H,S,Hd]
		scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B,H,S,S]

		if causal:
			s = x.size(1)
			causal_mask = torch.ones(s, s, device=x.device, dtype=torch.bool).tril()
			scores = scores.masked_fill(~causal_mask, float("-inf"))

		if padding_mask is not None:
			key_mask = padding_mask[:, None, None, :]   # [B,1,1,S]
			scores = scores.masked_fill(~key_mask, float("-inf"))

		attn = torch.softmax(scores, dim=-1)       # [B,H,S,S]

		if padding_mask is not None:
			query_mask = padding_mask[:, None, :, None]  # [B,1,S,1]
			attn = attn * query_mask.to(attn.dtype)

		context = attn @ v                          # [B,H,S,Hd]
		context = self._merge_heads(context)        # [B,S,D]
		out = self.out_proj(context)                # [B,S,D]

		if padding_mask is not None:
			out = out * padding_mask.unsqueeze(-1).to(out.dtype)
		return out
```

