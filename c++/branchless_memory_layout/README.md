# Memory-Layout Branchless 专题

这个目录用于整理更依赖数据布局和访问模式的 branchless 技巧。这里的优化重点通常不是单条 `if` 改写，而是通过改变内存组织方式，把热路径里的边界判断、状态分叉或条件写回移出主循环。

计划覆盖的主题：

- halo padding
- 双缓冲
- 两阶段 compact
- parser / table-driven

建议后续按独立实验拆分，例如：

- `halo_padding/`: 为 stencil、卷积、网格更新去掉边界判断
- `double_buffer/`: 用双缓冲隔离读写依赖与条件提交
- `two_phase_compact/`: 先生成 flags，再统一压缩或 scatter
- `table_driven_parser/`: 用状态表替代数据相关控制流

当前已落地的第一个示例：

- `halo_padding/`: 用补边界数组去掉 1D stencil 内层边界判断
- `double_buffer/`: 用 next-buffer + swap 代替条件提交式原地更新
- `two_phase_compact/`: 先生成 flags 和位置，再统一写回压缩结果
- `table_driven_parser/`: 用字符分类表和状态转移表替代分支型词法扫描

## 现有示例横向对比

- `halo_padding/`: 重点是把边界判断从热路径里移走
- `double_buffer/`: 重点是分离读路径和写路径，避免边读边条件提交
- `two_phase_compact/`: 重点是把判断和写回组织拆成两个阶段
- `table_driven_parser/`: 重点是把状态切换逻辑编码进表结构

如果按“重排数据流的力度”看，这几个示例大致是：

- `halo_padding/`: 调整输入布局
- `double_buffer/`: 调整读写时序和目标缓冲区
- `two_phase_compact/`: 调整处理阶段划分
- `table_driven_parser/`: 调整控制逻辑的表示方式

这个专题和 `c++/branchless/` 的区别是：

- `c++/branchless/` 更强调控制流改写成本
- 这里更强调数据布局、访问局部性和热路径结构重组

后续实现时建议同时关注：

- 额外内存占用
- cache locality
- 写放大与带宽压力
- 数据预处理成本
- correctness 校验是否覆盖布局变换后的语义等价性