# SIMD Branchless 专题

这个目录用于整理更偏 SIMD 的 branchless 技巧，重点不是普通标量位运算，而是把条件选择、过滤、压缩写回放进向量执行模型里讨论。

计划覆盖的主题：

- compare + blend
- movemask
- compress-store

建议后续按独立实验拆分，例如：

- `compare_blend/`: 向量比较与按 lane 选择
- `movemask/`: 从向量比较结果提取 bitmask，再配合 `ctz` / `popcnt`
- `compress_store/`: 依据 mask 做压缩写回

当前已落地的第一个示例：

- `compare_blend/`: 用 AVX2 按 lane 比较并选择 `max(a, b)`
- `movemask/`: 用 AVX2 比较生成 bitmask，再配合 `ctz` 找到首个命中位置
- `compress_store/`: 用 AVX2 `movemask` 批量筛选并压缩写回连续输出

## 现有示例横向对比

- `compare_blend/`: 解决的是“每个 lane 各自做二选一”，核心是 `compare -> mask -> select`
- `movemask/`: 解决的是“找第一个命中位置”，核心是把 lane 结果收缩成 bitmask 再消费
- `compress_store/`: 解决的是“把命中元素压紧写回”，核心是 `compare -> mask -> shuffle -> contiguous store`

如果把这三个例子按数据流复杂度排序，大致是：

- `compare_blend/`: 只做 lane 内选择
- `movemask/`: 从 lane 结果过渡到标量 bitmask
- `compress_store/`: 不仅判断，还要重排并压缩输出布局

这个专题和 `c++/branchless/` 的区别是：

- `c++/branchless/` 主要讨论标量分支改写
- 这里主要讨论 SIMD lane 级别的数据选择与压缩

后续实现时建议同时关注：

- 指令集前提，例如 AVX2/AVX-512
- alignment 与 load/store 方式
- tail handling
- mask 生成与消费成本
- 标量回退路径