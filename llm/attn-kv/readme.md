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

