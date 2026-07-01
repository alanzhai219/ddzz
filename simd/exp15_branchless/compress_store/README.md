# Compress Store

这个示例演示 SIMD 里更完整的一条 branchless 数据流：

- compare 产生 lane mask
- movemask 提取 bitmask
- shuffle 根据 bitmask 压缩命中的字节
- 连续写回输出缓冲区

这里处理的是“保留大于阈值的字节”，用来展示 `compress-store` 的基本模式。

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/simd/exp15_branchless/compress_store
g++ -O2 -mssse3 -std=c++17 main.cpp -o compress_store
./compress_store
```