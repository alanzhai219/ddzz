# Two-Phase Compact

这个示例演示 memory-layout 风格的两阶段 compact：

- 第一阶段只生成 `flags` 和每个元素的目标位置
- 第二阶段再统一把元素写到输出区

这种方式的重点不是单条分支技巧，而是把“判断”和“写回组织”拆开，让后续压缩路径更适合批处理和向量化。

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/c++/branchless_memory_layout/two_phase_compact
g++ -O2 -std=c++17 main.cpp -o two_phase_compact
./two_phase_compact
```