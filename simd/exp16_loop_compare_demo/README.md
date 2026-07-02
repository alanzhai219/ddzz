# Key Loop Comparison Demo

这个小目录专门演示你提到的 4 个问题，比较 `baseline_auto` 和 `manual_simd` 的关键循环：

- 指令条数
- load/store 形式
- 是否有多余 shuffle/blend
- 是否有 gather/scatter 或分支残留

## 目录内容

- `main.cpp`
  - `baseline_auto_max`: 标量写法，依赖编译器 O3 自动向量化
  - `manual_simd_max`: 手写 AVX2 `_mm256_max_epi32`
  - `manual_simd_compare_mask`: 手写 AVX2 compare+mask 版本（通常指令更多）

## 一键查看汇编

```bash
cd /home/xiuchuan/workspace/ddzz/simd/exp16_loop_compare_demo
g++ -O3 -mavx2 -std=c++17 -S -masm=intel -fverbose-asm main.cpp -o demo.s
```

打开 `demo.s`，重点看 3 个函数的主循环。

## 如何理解 4 个判断点

1. 指令条数
- 看每个函数主循环体里有多少条关键指令（不算标签和注释）。
- 如果 `manual_simd` 比 `baseline_auto` 多很多 uops，未必更快。

2. load/store 形式
- 连续内存通常是 `vmovdqu` / `vmovdqa`（或对应 load/store 指令）。
- 如果出现 `vgather*` / `vscatter*`，一般表示访问模式稀疏，代价更高。

3. 是否有多余 shuffle/blend
- `manual_simd_max` 常见核心是 load/load/max/store。
- `compare+mask` 通常是 cmp/and/andn/or/store，步骤更多。
- 如果 baseline_auto 已生成 `vpmaxsd` 这类更直接指令，你的 compare+mask 就可能是“多余步骤”。

4. gather/scatter 或分支残留
- 主循环里若有 `jg/jl/je/jne` 之类分支，可能有分支残留。
- 若循环主路径只有向量算子 + 顺序 load/store，通常是更理想的 SIMD 形态。

## 验证命令

```bash
g++ -O3 -mavx2 -std=c++17 main.cpp -o demo
./demo
```

3 个 checksum 一样说明语义一致，可以放心比较汇编和性能。

## 实战建议

在真实项目里，先做这三组对比再决策是否保留手写 SIMD：

- `baseline_auto`（真实可维护基线）
- `manual_simd_max`（你认为最优的手写版本）
- `manual_simd_compare_mask`（更“显式”的手写版本）

如果 `baseline_auto` 已接近 `manual_simd_max`，通常优先保留 baseline_auto，除非手写版有稳定且显著的收益。
