# Halo Padding

这个示例演示 memory-layout 风格的去分支技巧：

- 朴素版本在每个元素上检查左右边界
- halo padding 版本先在两端补一圈边界值，再让热路径只做统一 stencil 计算

这里用的是一个 1D 三点 stencil：

```text
out[i] = left + center + right
```

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/c++/branchless_memory_layout/halo_padding
g++ -O2 -std=c++17 main.cpp -o halo_padding
./halo_padding
```