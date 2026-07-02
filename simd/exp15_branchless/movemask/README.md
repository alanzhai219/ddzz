# Movemask + ctz

这个示例演示 SIMD 里另一种很常见的 branchless 模式：

- 先对一批字节做并行比较
- 再用 `movemask` 把比较结果压成一个 bitmask
- 最后用 `ctz` 找到第一个命中的 lane

它很适合处理“找第一个满足条件的位置”这类问题。

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/simd/exp15_branchless/movemask
g++ -O2 -mavx2 -std=c++17 main.cpp -o movemask_o2
taskset -c 1 ./movemask_o2

g++ -O3 -mavx2 -std=c++17 main.cpp -o movemask_o3
taskset -c 1 ./movemask_o3
```

这个版本采用和 `compare_blend` 一致的公平基准结构：

- `branchy(auto)`: 正常标量写法，允许编译器自动向量化
- `branchy(no-vec)`: 同样逻辑，但对该函数禁用自动向量化
- `avx2(movemask)`: 手写 AVX2 `cmpeq + movemask + ctz`

并且每轮都会消费结果（`guard`），减少高优化级别下的过度消除干扰。