# Compress Store

这个示例演示 SIMD 里更完整的一条 branchless 数据流：

- compare 产生 lane mask
- movemask 提取 bitmask
- 按 bitmask 压缩命中的字节
- 连续写回输出缓冲区

这里处理的是“保留大于阈值的字节”，用来展示 `compress-store` 的基本模式。

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/simd/exp15_branchless/compress_store
g++ -O2 -mavx2 -std=c++17 main.cpp -o compress_store_o2
taskset -c 1 ./compress_store_o2

g++ -O3 -mavx2 -std=c++17 main.cpp -o compress_store_o3
taskset -c 1 ./compress_store_o3
```

这个版本采用和 `compare_blend` 一致的公平基准结构：

- `branchy(auto)`: 正常标量写法，允许编译器自动向量化
- `branchy(no-vec)`: 同样逻辑，但对该函数禁用自动向量化
- `avx2(compress)`: 手写 AVX2 `cmpgt + movemask + bit-scan`

并且每轮都会消费输出结果（`guard`），减少高优化级别下的过度消除干扰。