# KV Cache Summary

本文档总结 `use_cache` 与无 cache 在当前实现中的差异，覆盖：
- 计算流程
- 内存使用
- 计算量

相关代码：
- [single-head-attn-kv.py](single-head-attn-kv.py)
- [multi-head-attn-kv.py](multi-head-attn-kv.py)

## 1. 计算流程差异

### 1.1 无 cache（每步重算前缀）

1. 第 t 步输入完整前缀（长度 t）。
2. 每步都重新计算该前缀的 Q/K/V。
3. 每步都做完整 attention。
4. 历史 token 的 K/V 被重复计算。

### 1.2 use_cache=True（增量复用）

1. `prefill`：先对已有上下文做一次前向，产出初始 `cache`。
2. `decode`：后续只对新增输入（通常 `T=1`）计算 `k_new/v_new`。
3. 与历史 `past_k/past_v` 在时间维拼接。
4. 当前 query 与拼接后的 K/V 做 attention。
5. 返回更新后的 `new_cache`。

### 1.3 在本仓实现中的具体差异

| 项目 | Single-Head | Multi-Head |
|---|---|---|
| cache `k` shape | `[B, T_cache, D]` | `[B, H, T_cache, Hd]` |
| cache `v` shape | `[B, T_cache, D]` | `[B, H, T_cache, Hd]` |
| 拼接维度 | `dim=1` | `dim=2` |
| 是否 split/merge heads | 否 | 是 |

结论：核心流程一致，multi-head 只是多了 head 维度的重排。

### 1.4 对应流程图（含 shape / 算子 / 输入输出名）

#### Single-Head + KV Cache

```mermaid
flowchart TD
	X[x\nshape: B x T x D]
	C[cache\nkeys: k,v\nshape(k/v): B x T_cache x D]

	Q[q_proj (Linear)\ninput: x\noutput: q\nshape: B x T x D]
	K[k_proj (Linear)\ninput: x\noutput: k_new\nshape: B x T x D]
	V[v_proj (Linear)\ninput: x\noutput: v_new\nshape: B x T x D]

	CATK[cat dim=1\ninputs: past_k, k_new\noutput: k\nshape: B x T_total x D]
	CATV[cat dim=1\ninputs: past_v, v_new\noutput: v\nshape: B x T_total x D]

	S1[transpose\ninput: k\noutput: k^T\nshape: B x D x T_total]
	S2[matmul + scale 1/sqrt(D)\ninputs: q, k^T\noutput: scores\nshape: B x T x T_total]
	S3[softmax dim=-1\ninput: scores\noutput: attn\nshape: B x T x T_total]
	S4[matmul\ninputs: attn, v\noutput: context\nshape: B x T x D]
	O[out_proj (Linear)\ninput: context\noutput: out\nshape: B x T x D]

	NC[new_cache\n{k,v}\nshape(k/v): B x T_total x D]

	X --> Q
	X --> K
	X --> V
	C --> CATK
	C --> CATV
	K --> CATK
	V --> CATV
	CATK --> S1 --> S2 --> S3 --> S4 --> O
	Q --> S2
	CATV --> S4
	CATK --> NC
	CATV --> NC
```

#### Multi-Head + KV Cache

```mermaid
flowchart TD
	X[x\nshape: B x T x D]
	C[cache\nkeys: k,v\nshape(k/v): B x H x T_cache x Hd]

	Q0[q_proj (Linear)\ninput: x\noutput: q0\nshape: B x T x D]
	K0[k_proj (Linear)\ninput: x\noutput: k0\nshape: B x T x D]
	V0[v_proj (Linear)\ninput: x\noutput: v0\nshape: B x T x D]

	QS[split_heads (view+transpose)\ninput: q0\noutput: q\nshape: B x H x T x Hd]
	KS[split_heads (view+transpose)\ninput: k0\noutput: k_new\nshape: B x H x T x Hd]
	VS[split_heads (view+transpose)\ninput: v0\noutput: v_new\nshape: B x H x T x Hd]

	CATK[cat dim=2\ninputs: past_k, k_new\noutput: k\nshape: B x H x T_total x Hd]
	CATV[cat dim=2\ninputs: past_v, v_new\noutput: v\nshape: B x H x T_total x Hd]

	S1[transpose\ninput: k\noutput: k^T\nshape: B x H x Hd x T_total]
	S2[matmul + scale 1/sqrt(Hd)\ninputs: q, k^T\noutput: scores\nshape: B x H x T x T_total]
	S3[softmax dim=-1\ninput: scores\noutput: attn\nshape: B x H x T x T_total]
	S4[matmul\ninputs: attn, v\noutput: context_h\nshape: B x H x T x Hd]
	MG[merge_heads (transpose+view)\ninput: context_h\noutput: context\nshape: B x T x D]
	O[out_proj (Linear)\ninput: context\noutput: out\nshape: B x T x D]

	NC[new_cache\n{k,v}\nshape(k/v): B x H x T_total x Hd]

	X --> Q0 --> QS
	X --> K0 --> KS
	X --> V0 --> VS
	C --> CATK
	C --> CATV
	KS --> CATK
	VS --> CATV
	CATK --> S1 --> S2 --> S3 --> S4 --> MG --> O
	QS --> S2
	CATV --> S4
	CATK --> NC
	CATV --> NC
```

---

## 2. 内存使用差异

| 维度 | 无 cache | use_cache |
|---|---|---|
| 持久内存 | 低（不保存历史 K/V） | 高（保存历史 K/V，随长度线性增长） |
| 临时激活 | 每步可能更大（重算整段） | decode 常见更小（通常 `T=1`） |
| Trade-off | 省显存、费算力 | 费显存、省算力 |

### 2.1 缓存规模（主要项）

1. Single-head：
- `k`: `[B, T_total, D]`
- `v`: `[B, T_total, D]`
- 总元素量约为 `2 * B * T_total * D`

2. Multi-head：
- `k`: `[B, H, T_total, Hd]`
- `v`: `[B, H, T_total, Hd]`
- 因为 `H * Hd = D`，总元素量同阶，约 `2 * B * T_total * D`

---

## 3. 计算量差异

设最终序列长度为 `T`，特征维度为 `D`。

### 3.1 无 cache（逐步重喂完整前缀）

1. 注意力主项累计：
$$
\sum_{t=1}^{T} O(t^2 D) = O(T^3 D)
$$

2. 投影主项累计：
$$
\sum_{t=1}^{T} O(t D^2) = O(T^2 D^2)
$$

### 3.2 use_cache（逐步增量）

1. 注意力主项累计（常见每步 1 token）：
$$
\sum_{t=1}^{T} O(t D) = O(T^2 D)
$$

2. 投影主项累计：
$$
O(T D^2)
$$

结论：use_cache 显著降低重复计算，长序列收益明显；代价是缓存带来的线性显存增长。

---

## 4. prefill / decode 对应关系

1. `prefill`：第一次输入（上下文批量输入），建立初始 cache。
2. `decode`：后续 next 输入（通常是下一个 token），复用并扩展 cache。

当前示例中：
- `x_prefill`: `[B, 5, D]`
- `x_decode`: `[B, 1, D]`

---

## 5. 一句话总结

`use_cache` 本质是“用更多持久内存（存 K/V）换更少重复计算（增量解码加速）”。

## 参考

https://zhuanlan.zhihu.com/p/2022596782975193682

