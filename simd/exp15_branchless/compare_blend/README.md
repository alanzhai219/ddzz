# Compare + Blend

这个示例演示 SIMD 里的 branchless 选择：

- 标量版本用 `if (a[i] > b[i])`
- SIMD 版本先做向量比较，再按 lane 选择结果

这里实现的是 `max(a, b)`，因为它很适合展示 `compare -> mask -> blend` 这条基本路径。

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/simd/exp15_branchless/compare_blend
g++ -O2 -mavx2 -std=c++17 main.cpp -o compare_blend_o2
./compare_blend_o2

g++ -O3 -mavx2 -std=c++17 main.cpp -o compare_blend_o3
./compare_blend_o3
```

这个版本加入了更公平的对比：

- `branchy(auto)`: 正常标量写法，允许编译器自动向量化
- `branchy(no-vec)`: 同样逻辑，但对该函数禁用自动向量化
- `avx2(compare+mask)`: 手写 compare + mask + select 路径
- `avx2(max_epi32)`: 手写 AVX2 `max_epi32` 路径

并且每轮都会消费输出结果（`guard`），避免在高优化级别下被过度消除。

如果你想确认 O3 是否把 `branchy(auto)` 自动向量化，可加报告参数：

```bash
g++ -O3 -mavx2 -std=c++17 -fopt-info-vec-optimized main.cpp -o compare_blend_o3
./compare_blend_o3
```