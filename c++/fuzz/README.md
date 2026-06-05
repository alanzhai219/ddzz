# What is Fuzz

It is a tool for catching the memory issue.

## how to write with Fuzz

```c
extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
```

1. extern "C"：防止 C++ 的“名称修饰” (Name Mangling)
C++ 支持函数重载（比如允许有两个同名的函数，但参数不同）。为了实现这一点，C++ 编译器在编译时，会把函数的参数类型信息“加密”到函数名中。
如果不加 extern "C"，编译器可能会把函数名变成类似 _Z22LLVMFuzzerTestOneInputPKhm 这样的乱码。
但是，libFuzzer 引擎（底层是 C/C++ 混合编写）在运行时，只会死板地去寻找名为 LLVMFuzzerTestOneInput 的纯文本符号。
作用：extern "C" 告诉 C++ 编译器：“请像 C 语言一样编译这个函数，不要修改它的名字”，确保 libFuzzer 引擎能准确找到并调用它。

2. int：返回值类型
规则：这个函数必须返回 0。
原因：libFuzzer 约定，返回 0 表示“本次测试正常结束”。如果你返回了非 0 的值，libFuzzer 会认为你的代码主动报告了一个错误，它会立即停止 Fuzzing 并报错。

3. LLVMFuzzerTestOneInput：固定的入口名称
这是 libFuzzer 引擎硬性规定的函数名。你不能把它改成 MyFuzzFunction，否则引擎找不到入口。

4. const uint8_t *data：输入的随机字节流
uint8_t：表示无符号 8 位整数，也就是原始的字节 (Byte)。libFuzzer 生成的输入不是字符串，而是纯粹的二进制数据（可能包含 0x00 等不可见字符）。
const：表示只读。你的测试代码绝对不能去修改 data 指向的内存。因为 libFuzzer 可能会复用这块内存，或者修改它会导致引擎内部状态混乱。
*data：指向这块内存的指针。

5. size_t size：字节流的长度
极其重要：因为 data 不是以 \0 结尾的 C 字符串，你绝对不能使用 strlen(data)。
你必须严格依赖 size 变量来判断数据的边界。任何对 data 的读取或拷贝操作，都不能超出 size 的范围，否则就会触发越界读取（Out-of-bounds Read）漏洞。

## how to build with Fuzz

```bash
clang++ -fsanitize=fuzzer,address -g simple.cpp -o simple_fuzzer
```

## run and get output

执行可执行程序，会在本地生成`crash-...`的文件

## run with `crash` log

```bash
./a.out crash-xxx
```
