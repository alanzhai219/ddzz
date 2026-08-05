`contiguous()` 的实现可以分为三个步骤：

### 步骤 1：快速返回（no-op）

```cpp
if (is_continuous()) return;
```

如果 tensor 已经是连续的，什么也不做，直接返回。因为数据已经按 row-major 排列好了，不需要任何操作。

---

### 步骤 2：按 row-major 逻辑顺序遍历，逐个收集元素

```cpp
std::vector<float> flat(data_.size());          // 新的连续 buffer
std::vector<size_t> idx(shape_.size(), 0);       // 多维索引，初始全 0

for (size_t pos = 0; pos < total; ++pos) {
    // 用当前多维索引 × strides 算出在旧 data 中的位置
    size_t offset = 0;
    for (size_t d = 0; d < shape_.size(); ++d) {
        offset += idx[d] * strides_[d];
    }
    flat[pos] = data_[offset];   // 拷到新 buffer 的 pos 位置

    // 多维索引递增（进位逻辑）
    for (size_t d = shape_.size(); d > 0; --d) {
        if (++idx[d - 1] < shape_[d - 1]) break;  // 没进位，停止
        idx[d - 1] = 0;                            // 进位，归零，继续往高位进
    }
}
```

**核心思想：**

- 外层 `pos` 表示新 buffer 中的逻辑位置（row-major 顺序：0, 1, 2, ...）
- `idx` 维护当前的多维坐标，例如 shape `[2, 3]` 时依次为 `(0,0)`, `(0,1)`, `(0,2)`, `(1,0)`, `(1,1)`, `(1,2)`
- `offset = Σ idx[d] × strides_[d]` 利用旧的 strides 算出该坐标在旧 data 中的实际偏移量
- 进位逻辑模拟了一个多维计数器：最低位 `idx[last]` 每次 +1，满 `shape[last]` 就归零并向高位进 1

**具体例子：** shape `[2, 3]`，旧的 strides 是 `[1, 2]`（转置后的非连续 tensor）

| pos | idx | offset = i×1 + j×2 | flat[pos] 取值 |
|-----|-----|---------------------|----------------|
| 0 | (0,0) | 0×1 + 0×2 = 0 | data_[0] |
| 1 | (0,1) | 0×1 + 1×2 = 2 | data_[2] |
| 2 | (0,2) | 0×1 + 2×2 = 4 | data_[4] |
| 3 | (1,0) | 1×1 + 0×2 = 1 | data_[1] |
| 4 | (1,1) | 1×1 + 1×2 = 3 | data_[3] |
| 5 | (1,2) | 1×1 + 2×2 = 5 | data_[5] |

旧的 data 是 `[a, b, c, d, e, f]` 按列存储（strides `[1,2]`），经过遍历后 flat 变成 `[a, c, e, b, d, f]`，这就是标准的行优先连续排列。

---

### 步骤 3：替换内部存储 + 重建 strides

```cpp
data_ = std::move(flat);  // 用新的连续 buffer 替换旧的
compute_strides();         // 重新计算为标准的连续 strides
```

`compute_strides()` 会根据当前 shape 生成标准 row-major strides。此时 strides 变为 `[3, 1]`（shape `[2,3]` 的连续 strides）。

---

### 一句话总结

**`contiguous()` 就是用一个多维计数器按 row-major 顺序走一遍，利用旧的 strides 做"翻译"，把数据按正确顺序抄到一个新 buffer 里，然后替换掉旧的存储。**
