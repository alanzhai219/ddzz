# Table-Driven Parser

这个示例用一个很小的 toy lexer 展示 table-driven parser：

- branchy 版本用 if/else 判断字符类型和状态切换
- table-driven 版本先查字符分类表，再查状态转移表

这里统计两类 token：

- 标识符
- 整数

编译方式：

```bash
cd /home/xiuchuan/workspace/ddzz/c++/branchless_memory_layout/table_driven_parser
g++ -O2 -std=c++17 main.cpp -o table_driven_parser
./table_driven_parser
```