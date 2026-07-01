# Movemask + ctz

这个示例演示 SIMD 里另一种很常见的 branchless 模式：

- 先对一批字节做并行比较
- 再用 `movemask` 把比较结果压成一个 bitmask
- 最后用 `ctz` 找到第一个命中的 lane

它很适合处理“找第一个满足条件的位置”这类问题。

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/simd/exp15_branchless/movemask
g++ -O2 -msse2 -std=c++17 main.cpp -o movemask
./movemask
```