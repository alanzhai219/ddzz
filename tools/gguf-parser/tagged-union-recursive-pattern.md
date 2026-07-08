# Tagged Union + Recursive Container 设计模式总结

## 1. 这是什么模式

这是 C 语言里常见的一个组合范式：

1. `tagged union`（可辨识联合体）
2. `recursive container`（递归容器）

在你的代码里：

- `gguf_metadata_value_type` 是 `tag`（类型标签）
- `union gguf_metadata_value_t` 是 `union payload`（值载荷）
- `array` 分支里再次包含 `gguf_metadata_value_t *array`，形成递归

它等价于高层语言中的“动态 Value 类型”（类似 JSON value: number/string/array/...）。

---

## 2. 结构映射（对应当前代码）

### 2.1 类型标签

```c
typedef enum {
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    ...
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    ...
} gguf_metadata_value_type;
```

作用：告诉读取方“当前 `union` 应该按哪一支解释”。

### 2.2 联合体载荷

```c
union gguf_metadata_value_t {
    uint8_t uint8;
    int32_t int32;
    gguf_string_t string;
    struct {
        gguf_metadata_value_type type;
        uint64_t len;
        gguf_metadata_value_t *array;
    } array;
};
```

作用：同一段内存可表示多种值，但任一时刻只应使用一个分支。

### 2.3 递归数组

- `array.type`: 元素类型
- `array.len`: 元素个数（不是字节数）
- `array.array`: 元素序列

因为元素类型也可以是 `ARRAY`，所以能形成嵌套（如 `array[array[int32]]`）。

---

## 3. 为什么这种模式常用于二进制格式

1. 灵活：可以在一个统一容器里承载多种元数据类型
2. 可扩展：新增类型通常只需扩展 `enum + union branch`
3. 易序列化：天然符合 `Type + Length + Value` 的解析流程
4. 跨语言：C/C++/Rust/Go 都容易映射这种模型

---

## 4. 使用规则（避免踩坑）

1. 必须先看 `tag` 再读 `union` 字段
2. 对 `string/array` 必须管理内存生命周期
3. 对 `array` 必须信任并校验 `len`（防越界）
4. 嵌套释放必须递归（你代码的 `free_value` 就是这个思想）

一句话：`tag` 是真相，`union` 只是载体。

---

## 5. 你当前测试体现了什么

你的测试文件已经覆盖了两个关键路径：

1. 写路径（构造）
- `make_int32`
- `make_string`
- `make_array`

2. 读路径（按 tag 读取）
- `read_int32(type, v)`
- `read_string(type, v)`
- `read_array_item(type, v, index)`

并验证了递归嵌套：`array[array[int32]]`。

---

## 6. 适用与不适用

适用：

1. 配置/元数据值类型不固定
2. 二进制协议需要紧凑且可扩展
3. 需要表达层级嵌套结构

不适用：

1. 业务类型完全固定且编译期可知
2. 希望完全静态类型安全、尽量避免手工内存管理

---

## 7. 一句话记忆

这是把“动态值系统”用 C 手工实现出来的范式：

- `enum` 负责“它是什么”
- `union` 负责“它的值”
- `array` 分支负责“它可以递归嵌套”
