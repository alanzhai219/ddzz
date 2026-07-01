# Compare + Blend

这个示例演示 SIMD 里的 branchless 选择：

- 标量版本用 `if (a[i] > b[i])`
- SIMD 版本先做向量比较，再按 lane 选择结果

这里实现的是 `max(a, b)`，因为它很适合展示 `compare -> mask -> blend` 这条基本路径。

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/simd/exp15_branchless/compare_blend
g++ -O2 -msse2 -std=c++17 main.cpp -o compare_blend
./compare_blend
```