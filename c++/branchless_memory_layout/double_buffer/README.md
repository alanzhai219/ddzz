# Double Buffer

这个示例演示 memory-layout 风格的双缓冲：

- baseline 版本每轮先拷贝快照，再只在状态变化时原地提交
- double-buffer 版本直接把下一轮状态写到 `next` 缓冲区，最后交换 `current/next`

它的重点不是“少写一次 if”，而是把“读旧状态”和“写新状态”彻底拆开，让热路径变成统一写回。

这里用的是一个一维多数投票规则：

```text
next[i] = (left + center + right >= 2) ? 1 : 0
```

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/c++/branchless_memory_layout/double_buffer
g++ -O2 -std=c++17 main.cpp -o double_buffer
./double_buffer
```