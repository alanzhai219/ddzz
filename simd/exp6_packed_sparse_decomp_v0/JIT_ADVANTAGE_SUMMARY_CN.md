# JIT 何时能超越 AVX-512 Intrinsics

---

## 三个 JIT 解压缩内核对比与优化策略分析

本节系统对比项目中的三个 JIT 内核——**基础版**、**前缀和优化版**、**掩码特化版**——并逐一分析每一步优化的**手法**（Trick）与**原因**（Why）。

### 内核总览

| 维度 | V1：基础 JIT | V2：前缀和 JIT | V3：掩码特化 JIT |
|------|-------------|---------------|-----------------|
| 源文件 | `decomp_kernel_jit.hpp` | `decomp_kernel_jit_opt.hpp` | `decomp_kernel_jit_opt_specialized.hpp` |
| 掩码何时已知 | 运行时 | 运行时 | JIT 编译时 |
| 循环结构 | 双层 `for` 全展开 | 外层运行时循环 + 内层展开 | 双层 C++ `for` 全展开 |
| 每 chunk 核心指令数 | 6 | 4–5 | 2–4（依数据而定） |
| 关键瓶颈 | 串行指针依赖链 | 前缀和仍有短依赖 | 无运行时依赖 |

---

### V1 → V2：打破串行指针依赖链

#### V1 的瓶颈

V1 按顺序处理 4 个 chunk，每一个 chunk 的压缩数据加载地址依赖于前一个 chunk 的 `popcnt + add` 结果：

```
chunk0: vmovdqu8 [src] → popcnt → add src, cnt
chunk1: vmovdqu8 [src] → popcnt → add src, cnt   ← 等 chunk0 完成
chunk2: vmovdqu8 [src] → popcnt → add src, cnt   ← 等 chunk1 完成
chunk3: vmovdqu8 [src] → popcnt → add src, cnt   ← 等 chunk2 完成
```

关键路径延迟 = 4 × (3 周期 `popcnt` + 1 周期 `add`) = **16 周期**。

#### Trick 1：前缀和地址预计算

V2 将 4 个 `popcnt` 计算提到前面，利用 OoO（乱序执行）CPU 让它们并行执行，然后用 3 次 `add` 做前缀和：

```asm
; Phase 2: 4 个 popcnt 独立，可在连续周期发射（throughput = 1/cycle）
popcnt reg_m1, reg_m1    ; p1
popcnt reg_m2, reg_m2    ; p2
popcnt reg_m3, reg_m3    ; p3
popcnt reg_m4, reg_m4    ; p4

; Phase 3: 前缀和计算偏移量
mov  reg_off1, reg_m1           ; off1 = p1
add  reg_m2,   reg_off1         ; off2 = p1 + p2
add  reg_m3,   reg_m2           ; off3 = p1 + p2 + p3
add  reg_m4,   reg_m3           ; total = p1 + p2 + p3 + p4
```

**原因**：
- `popcnt` 延迟 3 周期，吞吐 1/周期。4 个独立 `popcnt` 可在 3+3=6 周期内全部完成（最后一个在第 6 周期就绪），而非串行的 4×3=12 周期。
- 前缀和只需 3 次 `add`（各 1 周期），总共约 **6 周期**，相比 V1 的 16 周期改善了 **2.67×**。

> Note:
> - 延迟 (Latency = 3 周期)：
指的是单个指令从“开始执行”到“结果可用”所需要的时间。
如果你只执行 1 条 popcnt，它需要 3 个周期才能拿到结果。
> - 吞吐 (Throughput = 1/周期)：
指的是执行单元每隔多久可以接收一条新指令。
吞吐为 1/周期，意味着 CPU 不需要等上一条 popcnt 做完，每个周期都可以发射（Issue）一条新的 popcnt 指令进入流水线。
> - `popcnt` 指令延迟和吞吐查询：https://zhuanlan.zhihu.com/p/681293647

**指令级并行ILP**：
- 公式：
  $T_{total} = \text{Latency} + (\text{Count} - 1) \times \text{ThroughputInterval}$

#### Trick 2：Fused Masked Expand-Load

V1 先 `vmovdqu8` 加载到 ZMM，再 `vpexpandb` 就地扩展（2 条指令）。V2 使用 `vpexpandb zmm{k}{z}, [mem]` 一条指令直接从内存做掩码扩展加载：

```asm
; V1（2 步）：
vmovdqu8 zmm0, [src]                        ; 步骤 1：盲加载 64B
vpexpandb zmm0 | k1 | T_z, zmm0            ; 步骤 2：就地扩展

; V2（1 步）：
vpexpandb zmm0 | k1 | T_z, [src]           ; 直接从内存扩展加载
```

**原因**：`vpexpandb` 本身支持内存源操作数。合并后少一次 ZMM 写和读，减少了 µop 数量和端口压力。

#### Trick 3：Store 延迟（Deferred Stores）

V1 在每个 chunk 的 expand 之后立即 store。V2 把 4 个 expand 全做完再集中 store：

```asm
; 先做 4 个 expand（读密集）
vpexpandb zmm0 | k1 | T_z, [src]
vpexpandb zmm1 | k2 | T_z, [src + off1]
vpexpandb zmm2 | k3 | T_z, [src + off2]
vpexpandb zmm3 | k4 | T_z, [src + off3]

; 再做 4 个 store（写密集）
vmovdqu8 [dst + 0],   zmm0
vmovdqu8 [dst + 64],  zmm1
vmovdqu8 [dst + 128], zmm2
vmovdqu8 [dst + 192], zmm3
```

**原因**：将读端口和写端口的使用分离到不同时间窗口，减少 load/store 单元争用。同时 CPU 写缓冲区（Store Buffer）可以在后续指令执行期间异步写回，不阻塞 expand 的执行。

#### Trick 4：运行时循环 + `align(64)`

V1 用 C++ 编译期 `for` 循环展开所有 block（`blocks_` 个），生成 `blocks_ × N` 条指令。V2 改为运行时循环（`dec reg; jnz`），并将循环入口对齐到 64 字节：

```asm
align(64);           ; 循环入口对齐到缓存行
L(loop_blocks);
  ... 16 组展开 ...
  dec reg_block_cnt;
  jnz loop_blocks;
```

**原因**：
- 完全展开在 `blocks_` 很大（如 1000）时生成 MB 级机器码，严重冲刷 I-cache。
- 运行时循环重用同一块代码。`align(64)` 避免循环入口跨越缓存行，消除 I-cache fetch penalty。
- 每次迭代内仍有 16 组 ×4 chunk 展开，在 I-cache 压力与 ILP 之间取得平衡。

#### Trick 5：减少 callee-saved 寄存器保存

V1 保存 r12-r15 + k1-k4（8 次 push/pop + 4 次 kmovq）。V2 只保存 r12-r15（4 次 push/pop），k 寄存器使用 caller-saved 策略不保存：

**原因**：k 寄存器在 System V ABI 下是 caller-saved（调用者负责保存），被调用函数无需保存。V1 的保存是冗余的。

#### Trick 6：指针推进替代全展开偏移

V1 用编译期常量 `wei_offset + cl*64` 做 destination 偏移。V2 在循环末尾 `add reg_decomp_dst, 4096` 推进目标指针，内层用局部变量 `reg_dst_local` 配合小偏移 `dst_off`：

**原因**：避免超大立即数偏移（当 block 很多时偏移超过 32 位限制），用一次 `add` 替代为每个 block 计算独立偏移。

---

### V2 → V3：将数据"烧进"机器码

V3 的核心思想是：在 LLM 推理中，稀疏掩码在模型加载后不再改变。JIT 编译器在代码生成时读取实际掩码值，消除一切运行时不确定性。

#### Trick 7：源偏移量编译时预计算

V2 在运行时用 `popcnt + prefix-sum` 计算源偏移量。V3 在 C++ 代码生成时用 `__builtin_popcountll()` 累加一个 C++ 变量 `src_off`，该变量最终变为指令中的立即数：

```cpp
int src_off = 0;   // C++ 变量，非寄存器
for (chunk ...) {
    int popcnt = __builtin_popcountll(mask);
    // 发射：vpexpandb zmm, [reg_src + src_off]
    //                                 ^^^^^^^^ 立即数！
    src_off += popcnt;   // C++ 加法，运行时不存在
}
```

**原因**：运行时 0 条 `popcnt` 指令、0 条 `add` 指令、0 条指针推进。所有地址计算在 JIT 编译时完成并硬编码为 `[base + imm]`，消除了全部地址计算的延迟和端口占用。

#### Trick 8：掩码值嵌入为立即数

V2 在运行时 `mov reg, [bitmask + off]` 从内存加载掩码。V3 直接 `mov reg, 0x0101...`（64 位立即数）：

```asm
; V2: 运行时内存加载
mov  rax, [reg_bitmask + bm_off]    ; L1 命中 ~4 周期

; V3: 掩码嵌入指令流
mov  rax, 0x0101010101010101         ; 无内存访问，指令解码即可用
```

**原因**：消除掩码内存加载，减少 load 端口压力。掩码值从 I-cache（指令缓存）获得，不占用 D-cache（数据缓存）带宽。

#### Trick 9：数据感知的"死代码消除"

V3 在 JIT 编译时检查掩码值，为三种情况生成不同指令序列：

| mask 值 | 生成的指令 | 指令数 | 消除了什么 |
|---------|-----------|--------|-----------|
| `0x0`（全零） | `vpxord + vmovdqu8` | 2 | 压缩数据加载、k-mask 设置、vpexpandb |
| `0xFFFF...`（全满） | `vmovdqu8 加载 + vmovdqu8 存储` | 2 | k-mask 设置、vpexpandb（改为直接拷贝） |
| 其它（部分） | `mov + kmovq + vpexpandb + vmovdqu8` | 4 | popcnt、add（偏移量已为立即数） |

```cpp
if (mask == 0) {
    vpxord(zr, zr, zr);                     // 2 条指令
} else if (mask == 0xFFFFFFFFFFFFFFFFULL) {
    vmovdqu8(zr, ptr[reg_src + src_off]);    // 2 条指令
} else {
    mov(reg_tmp, mask);                      // 4 条指令
    kmovq(kr, reg_tmp);
    vpexpandb(zr | kr | T_z, ptr[reg_src + src_off]);
}
vmovdqu8(ptr[reg_dst + dst_off], zr);
```

**原因**：通用内核对所有 chunk 执行相同的 6 条指令序列。当 30%+ chunk 为全零/全满时（结构化稀疏场景），V3 将这些 chunk 的指令数从 6 条降至 2 条，减少 66% 指令。这是 AOT 编译器无法做到的——它不知道哪些 chunk 是全零。

#### Trick 10：ZMM/K 寄存器轮转提高 ILP

V3 使用 `chunk & 3` 在 zmm0–zmm3 和 k1–k4 之间轮转：

```cpp
Xbyak::Zmm zr = Xbyak::Zmm(chunk & 3);
Xbyak::Opmask kr = Xbyak::Opmask(1 + (chunk & 3));
```

**原因**：连续 chunk 使用不同物理寄存器，消除了 WAW（写后写）和 WAR（写后读）假依赖。CPU 可以在前一个 chunk 的 store 未完成时就开始下一个 chunk 的 expand。

#### Trick 11：对齐量编译时预计算

V2 在运行时用 `neg + and_ + add` 计算 inter-block 对齐。V3 在 C++ 中 `int misalign = src_off & 0x3F; if (misalign) src_off += (64 - misalign);` 完成：

**原因**：消除了 3 条对齐计算指令（每 block 一次）。在 1000 block 场景下节省 3000 条指令。

#### Trick 12：无 callee-saved 保存

V3 只使用 `rax`（caller-saved）、`r8`、`r9` 作为指针，不使用 r12-r15：

**原因**：V3 运行时不需要循环计数器（全部展开）、不需要前缀和临时变量。指针从不被修改（所有偏移是立即数），故只需 2 个指针寄存器 + 1 个临时寄存器，无需 push/pop 开销。

---

### 优化策略演进总结

```
V1 基础 JIT                V2 前缀和 JIT               V3 掩码特化 JIT
──────────                 ──────────                  ──────────
串行 popcnt→add            并行 popcnt + 前缀和          编译时预计算（0 条指令）
分离 load + expand         融合 expand-from-memory       融合 + 按数据选择指令
逐 chunk store             延迟 store，批量写回           逐 chunk store（已无竞争）
全展开（I-cache 爆炸）       运行时循环 + 内展开            全展开（掩码固定，I-cache 可接受）
冗余 K-reg 保存            省略 K-reg 保存               无 callee-saved 保存
                           循环入口 align(64)            N/A（无循环）
                                                        死代码消除（零块/稠密块）
                                                        掩码/偏移作为立即数
                                                        ZMM/K 寄存器轮转
```

### Trick 汇总表

| # | 优化手法 | 引入版本 | 消除的开销 | 原理 |
|---|---------|---------|-----------|------|
| 1 | 前缀和地址预计算 | V2 | 串行 popcnt→add 依赖链 | 独立 popcnt 并行 + 3 次 add 前缀和，关键路径 16→6 周期 |
| 2 | Fused masked expand-load | V2 | 冗余 vmovdqu8 加载 | vpexpandb 支持内存源操作数，省 1 条 µop |
| 3 | Store 延迟批量写回 | V2 | load/store 端口争用 | 读写分离到不同时间窗口，Store Buffer 异步写回 |
| 4 | 运行时循环 + align(64) | V2 | I-cache 冲刷 | 复用代码 + 消除跨缓存行 fetch penalty |
| 5 | 省略冗余 K-reg 保存 | V2 | push/pop + kmovq 开销 | K 寄存器在 SysV ABI 下是 caller-saved |
| 6 | 指针推进替代大偏移 | V2 | 超大立即数编码 | 用 add 推进指针，内层用小偏移 |
| 7 | 源偏移量编译时预计算 | V3 | 全部 popcnt 和 add | C++ 变量 src_off 在 JIT 编译时累加，运行时为立即数 |
| 8 | 掩码嵌入为立即数 | V3 | 掩码内存加载 | 消除 load 端口压力，从 I-cache 获取掩码 |
| 9 | 数据感知死代码消除 | V3 | 全零/全满 chunk 的冗余指令 | 零块 2 条指令，全满 2 条指令 vs 通用 6 条 |
| 10 | ZMM/K 寄存器轮转 | V3 | WAW/WAR 假依赖 | chunk & 3 循环分配物理寄存器 |
| 11 | 对齐量编译时预计算 | V3 | 运行时对齐指令 (neg+and+add) | 在 C++ 侧计算，0 运行时开销 |
| 12 | 无 callee-saved 保存 | V3 | push/pop 序言尾声 | 用极少寄存器 + 立即数寻址，无需保存 |

### 适用场景

| 场景 | 推荐内核 | 原因 |
|------|---------|------|
| 掩码动态变化 | V2（前缀和 JIT） | 无法在编译时特化掩码，V2 的前缀和已足够优化 |
| 掩码固定 + 均匀随机稀疏 | V3 ≈ V2 | 全零/全满 chunk 很少，V3 优势不明显 |
| 掩码固定 + 结构化稀疏 | **V3（特化 JIT）** | 大量全零/全满 chunk 触发死代码消除，性能提升 1.36–1.62× |
| block 数极多（>1000） | V2 | V3 全展开导致 I-cache 爆炸；V2 运行时循环更友好 |

---

## 背景

在通用解压缩场景（动态位掩码）下，AVX-512 intrinsics 内核比最优的通用 JIT 内核（前缀和寻址）快约 1.3%。本文档描述了 JIT **反超**的场景——在特定条件下，JIT 性能超越 intrinsics **1.36–1.62 倍**。

---

## 核心思想：运行时特化

在 LLM 推理中，权重稀疏掩码在**模型加载时就已确定**，之后在每次推理调用中被重复使用。JIT 编译器可以在代码生成阶段检查实际的掩码值，从而生成静态（AOT）编译器无法产生的代码：

| 运行时被消除的开销 | 实现方式 |
|-------------------|---------|
| 所有 `popcnt` 指令 | 源偏移量作为立即数常量预计算 |
| 所有指针推进 `add` | 地址变为 `[base + 立即数]` — 无串行依赖 |
| 所有位掩码内存加载 | 掩码值嵌入为 64 位立即数 |
| 所有对齐运算 | 块间填充在 JIT 编译时预计算 |
| 零块的冗余 expand | 替换为 `vpxord + store`（2 条指令） |
| 稠密块的冗余 expand | 替换为 `vmovdqu8` 拷贝（2 条指令） |

---

## 生成代码对比

### 通用内核（intrinsics 或通用 JIT）— 每个 chunk：6 条指令

```asm
mov  rax, [bitmask + runtime_off]  ; 内存加载（位掩码）
kmovq k1, rax                      ; k-mask 设置
vpexpandb zmm{k1}{z}, [src]        ; 扩展加载（数据依赖的地址）
popcnt rax, rax                    ; 3 周期延迟
add  src, rax                      ; 依赖 popcnt → 串行链
vmovdqu8 [dst + off], zmm          ; 存储
```

### JIT 特化版 — 部分填充 chunk：4 条指令，0 地址计算周期

```asm
mov  rax, 0x0101...                ; 立即数掩码（无内存加载）
kmovq k1, rax                      ; k-mask 设置
vpexpandb zmm{k1}{z}, [base + IMM] ; 静态地址（无依赖）
vmovdqu8 [dst + IMM], zmm          ; 存储
```

### JIT 特化版 — 全零 chunk：2 条指令

```asm
vpxord zmm, zmm, zmm               ; 寄存器清零
vmovdqu8 [dst + IMM], zmm          ; 存储（无需读取压缩数据）
```

### JIT 特化版 — 全稠密 chunk：2 条指令

```asm
vmovdqu8 zmm, [base + IMM]         ; 直接 64 字节拷贝（无需 k-mask）
vmovdqu8 [dst + IMM], zmm          ; 存储
```

---

## 基准测试配置

```bash
g++ decomp_bench_jit_advantage.cpp -I../../3rdparty/xbyak -march=native -O2 -o decomp_bench_jit_advantage
./decomp_bench_jit_advantage
```

- **数据规模**：4 个块 × 4096 字节，500K 次迭代，2000 次预热
- **对比内核**：AVX-512 Intrinsics（通用） / JIT 通用版（前缀和） / JIT 特化版（掩码感知）

---

## 测试结果

### 场景 A — 随机 70% 稀疏度（非结构化剪枝）

所有 256 个 chunk 均为部分填充（无全零/全稠密捷径可用）。

| 内核 | 微秒/调用 | 对比 Intrinsics |
|------|----------|----------------|
| AVX-512 Intrinsics | ~0.204 | 基准线 |
| JIT 通用版 | ~0.202 | ~1.0× |
| JIT 特化版 | ~0.202 | ~1.0×（持平） |

**分析**：在均匀随机稀疏度下，每个 chunk 都是部分填充。JIT 的唯一优势在于静态寻址（消除了 `popcnt`/`add`），但在小数据量下，乱序执行引擎可以隐藏该延迟。结果：持平。

### 场景 B — 结构化稀疏度（30% 全零块 + 10% 全稠密块）

这是通道剪枝或块稀疏 LLM 权重的典型模式。

| 内核 | 微秒/调用 | 对比 Intrinsics |
|------|----------|----------------|
| AVX-512 Intrinsics | ~0.202 | 基准线 |
| JIT 通用版 | ~0.220 | 0.92×（更慢！） |
| **JIT 特化版** | **~0.149** | **1.36–1.62× 更快 ★** |

生成代码中的 chunk 分布：

| chunk 类型 | 数量 | 占比 | 每 chunk 指令数 |
|-----------|------|------|----------------|
| 全零 (mask=0) | 69/256 | 27% | 2（vpxord + store） |
| 全稠密 (mask=~0) | 29/256 | 11% | 2（vmovdqu8 拷贝） |
| 部分填充 | 158/256 | 62% | 4（mov+kmovq+vpexpandb+store） |

**JIT 获胜原因**：Intrinsics 对所有 chunk 都执行 `vpexpandb + popcnt + add`（6 条指令）。而特化 JIT 对 38% 的 chunk（全零 + 全稠密）仅使用 2 条指令，完全消除了压缩数据读取和 k-mask 逻辑。

---

## 对 LLM 推理的意义

```
模型加载                              推理（数百万次调用）
   │                                        │
   ├─ 解析权重                               ├─ 调用 jit_func(&args)
   ├─ 提取稀疏掩码                            │   └─ 每次调用 0.149 微秒
   ├─ JIT 编译特化内核                         │       （无 popcnt，无位掩码加载，
   │   └─ 编译开销：~微秒级                    │        无指针运算）
   └─ 就绪                                   └─ ...
```

JIT 编译开销（~微秒）在**数百万次推理调用中被摊销**。每次调用节省约 33% 的指令。

---

## 何时使用哪种内核

| 场景 | 最优内核 | 原因 |
|------|---------|------|
| 固定掩码，大量重复使用 | **JIT 特化版** | 编译开销可摊销，静态地址，死代码消除 |
| 动态掩码，大块数据 | AVX-512 Intrinsics | 无 JIT 编译开销，编译器调度优良 |
| 动态掩码，通用场景 | JIT 通用版（前缀和） | 与 intrinsics 差距仅 1.3% |

---

## JIT 编译开销分析

### 问题：掩码未知时，每次都要重新编译

前面的 benchmark 将 JIT 编译时间排除在测量之外——内核在 benchmark 循环前就编译好了。但在真实场景中，如果 mask **事先未知**，每次都需要重新实例化 `jit_decompress_specialized_t`，编译开销就不可忽略。

### 编译开销的构成

| 操作 | 来源 | 典型耗时 |
|------|------|---------|
| `mmap` 分配可执行内存 | Xbyak 构造函数 `CodeGenerator(4096*256)` = 1MB | 2–10 μs |
| 代码生成循环 | 遍历 `blocks × 64` 个 chunk，每个 chunk 发射 2–4 条 x86 指令 | 与规模成正比 |
| `mprotect` (W→X) | `ready()` 调用 | 1–5 μs |
| 析构时 `munmap` | 释放内存 | 2–10 μs |

每条 Xbyak 指令发射的本质是：编码 x86 指令（查表 + 位运算）→ 写入几个字节到缓冲区，每条指令大约 **20–100 ns**（纯用户态，无系统调用）。

### 估算公式

$$T_{compile} = T_{mmap} + T_{mprotect} + N_{chunks} \times C_{avg\_insns} \times T_{per\_insn}$$

代入典型值：

$$T_{compile} \approx 5\mu s + 3\mu s + (blocks \times 64) \times 3 \times 50ns$$

| blocks | chunks | 代码生成 | 系统调用 | **总编译时间** |
|--------|--------|----------|----------|---------------|
| 1 | 64 | ~10 μs | ~8 μs | **~18 μs** |
| 4 | 256 | ~38 μs | ~8 μs | **~46 μs** |
| 16 | 1024 | ~154 μs | ~8 μs | **~162 μs** |
| 64 | 4096 | ~614 μs | ~8 μs | **~622 μs** |

### 盈亏平衡点

假设 specialized JIT 每次调用比 generic JIT 快 $\Delta t$ 微秒，需要调用 $N$ 次才能回本：

$$N_{break\_even} = \frac{T_{compile}}{\Delta t}$$

benchmark 中 4 blocks 的场景下，假设 specialized 比 generic 快 ~0.2 μs/call：

$$N_{break\_even} = \frac{46\mu s}{0.2\mu s} \approx 230 \text{ 次}$$

> **同一个 mask 至少要调用 ~230 次，specialized JIT 才能盈利。**

注：benchmark 代码已添加编译时间计时和盈亏平衡点计算（`run_scenario` 中的 "JIT Compile Cost Amortization" 输出段），运行时可直接观察实际数据。

### LLM 推理场景下的摊销

LLM 自回归生成时，每生成一个 token 就要对同一个权重执行一次解压：

```
模型加载                              推理（每层权重 × 每 token）
   │                                        │
   ├─ 解析权重                               ├─ 解压权重 → 矩阵乘法
   ├─ 提取稀疏掩码                            │   同一掩码重复使用
   ├─ JIT 编译特化内核 (~46 μs/layer)         │
   │                                        │
   └─ 就绪                                   └─ 生成 1000 tokens × 32 layers
                                                = 32,000 次调用同一 mask
                                                编译开销被 32,000 次摊销
```

### 不适用的场景

| 场景 | 原因 | 推荐方案 |
|------|------|---------|
| 动态稀疏（activation sparsity） | 每次 forward 的 mask 不同，无法复用 | JIT 通用版 |
| 一次性解压 | 解压完直接用 dense 数据，没有复用机会 | AVX-512 Intrinsics |
| 大量不同 weight shape | 每种 mask 编译一个内核，I-cache 压力大 | JIT 通用版 + 内核缓存 |

### 实际工程缓解策略

| 策略 | 做法 |
|------|------|
| **内核缓存** | 对 mask 数组求 hash，缓存已编译内核。相同 mask → 直接复用函数指针 |
| **异步编译** | 首次调用用 generic JIT；后台线程编译 specialized 内核；编译完原子替换函数指针 |
| **分层策略（HotSpot 模式）** | 统计调用次数，超过阈值才触发特化编译 |
| **降低编译成本** | 减小 Xbyak 代码缓冲区大小；池化复用 `CodeGenerator` 对象，避免反复 `mmap`/`munmap` |

---

## 核心结论

> **JIT 的真正力量不在于匹配编译器的输出——而在于做编译器根本做不到的事**：基于运行时数据特化代码。

当位掩码值从运行时变量变为编译时常量，JIT 能够消除整类指令（`popcnt`、指针运算、位掩码加载），并为全零/全稠密 chunk 生成经过死代码消除的特化路径。这是运行时代码生成的典型应用场景。

---

## 深入理解：AOT vs JIT 的能力边界

### AOT 与 JIT 看到的世界

AOT（静态/提前编译器，如 GCC、Clang）在编译时只能看到**代码**，看不到**数据**。JIT 编译器在运行时同时看到**代码和数据**，可以把数据"烧进"生成的机器码中。

```
AOT 编译器看到的：                    JIT 编译器看到的：
┌──────────────┐                    ┌──────────────┐
│   源代码      │                    │   源代码      │
│  (变量是符号)  │                    │  (变量是符号)  │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │ 编译                               │ + 运行时数据
       ▼                                   ▼
┌──────────────┐                    ┌──────────────────────┐
│  通用机器码    │                    │  mask = 0xFF00FF00... │
│  (必须处理所有  │                    │  popcnt = 32          │
│   可能的输入)  │                    │  offset = 已知常量     │
└──────────────┘                    └──────────┬───────────┘
                                              │ 生成
                                              ▼
                                   ┌──────────────────────┐
                                   │  专用机器码             │
                                   │  (只处理这一种输入)      │
                                   └──────────────────────┘
```

---

### 示例 1：本项目中的 vpexpandb 解压缩

**AOT 编译器生成的代码**（必须处理任意掩码）：

```cpp
// 编译器不知道 mask 的值，必须在运行时：
// 1. 从内存加载 mask
// 2. 计算 popcnt（确定多少非零字节）
// 3. 用 popcnt 推进源指针
// 4. 执行 vpexpandb
for (int cl = 0; cl < 64; cl++) {
    uint64_t mask = bitmask[cl];           // 编译器不知道这是 0、~0、还是其他
    zmm = _mm512_maskz_expandloadu_epi8(mask, src);
    _mm512_storeu_si512(dst + cl*64, zmm);
    src += _mm_popcnt_u64(mask);           // 必须运行时计算
}
```

编译器**必须**为每个 chunk 生成完整的 6 条指令序列，因为它不知道 mask 的值：

```asm
mov   rax, [bitmask + cl*8]    ; ← 不能省：mask 未知
kmovq k1, rax                  ; ← 不能省：需要设置 k-mask
vpexpandb zmm, [src]{k1}{z}    ; ← 不能省：必须执行 expand
popcnt rax, rax                ; ← 不能省：偏移量未知
add   src, rax                 ; ← 不能省：指针依赖 popcnt
vmovdqu8 [dst], zmm            ; ← 不能省：必须存储
```

**JIT 编译器看到实际数据后**：

```cpp
// JIT 在代码生成时知道：bitmask[17] = 0x0000000000000000（全零）
// 所以这个 chunk：
//   - 不需要从压缩流读取任何数据
//   - 不需要设置 k-mask
//   - 不需要 vpexpandb
//   - 不需要 popcnt
//   - 源指针偏移量 = 0（编译时已知）
```

生成的代码只有 **2 条指令**：

```asm
vpxord zmm0, zmm0, zmm0       ; 清零
vmovdqu8 [dst + 1088], zmm0   ; 存储（偏移量是立即数）
```

**对比**：同一个 chunk，AOT = 6 条指令 + 串行依赖链，JIT = 2 条指令 + 零依赖。

---

### 示例 2：常量传播 — if 分支消除

**AOT 代码**：

```cpp
void process(int mode, float* data, int n) {
    for (int i = 0; i < n; i++) {
        if (mode == 1)      data[i] *= 2.0f;
        else if (mode == 2) data[i] += 1.0f;
        else                data[i] = 0.0f;
    }
}
```

AOT 编译器**不知道 mode 的值**，必须在循环内保留所有 3 个分支：

```asm
.loop:
    cmp  edi, 1          ; mode == 1?
    je   .mul
    cmp  edi, 2          ; mode == 2?
    je   .add
    ; else: zero
    ...
    jmp  .next
.mul:
    ...
    jmp  .next
.add:
    ...
.next:
    dec  ecx
    jnz  .loop
```

每次迭代：2 次比较 + 1 次跳转 = **分支预测压力 + 指令膨胀**。

**JIT 知道 mode=2**，直接生成：

```asm
.loop:
    vaddps zmm0, zmm0, zmm_one   ; 只有 add 路径，无分支
    dec  ecx
    jnz  .loop
```

消除了：所有比较指令、所有分支跳转、不可达代码（mode 1 和 else 路径）。

---

### 示例 3：循环次数已知 — 完全展开

**AOT 代码**：

```cpp
void memcpy_blocks(void* dst, void* src, int n_blocks) {
    for (int i = 0; i < n_blocks; i++)
        memcpy(dst + i*64, src + i*64, 64);
}
```

编译器不知道 `n_blocks`，必须生成通用循环（计数器、比较、跳转）。

**JIT 知道 n_blocks = 3**，直接展开：

```asm
vmovdqu64 zmm0, [src]
vmovdqu64 [dst], zmm0
vmovdqu64 zmm1, [src + 64]
vmovdqu64 [dst + 64], zmm1
vmovdqu64 zmm2, [src + 128]
vmovdqu64 [dst + 128], zmm2
ret
```

消除了：循环计数器、比较、跳转、循环归纳变量。

---

### 示例 4：稀疏矩阵乘法 — 结构已知

**AOT 代码**（通用 CSR SpMV）：

```cpp
// 编译器不知道每行有多少非零元素
for (int row = 0; row < N; row++) {
    float sum = 0;
    for (int j = row_ptr[row]; j < row_ptr[row+1]; j++) {  // 内循环次数未知
        sum += values[j] * x[col_idx[j]];                   // 间接寻址
    }
    y[row] = sum;
}
```

每次迭代需要：加载 `row_ptr`、比较、加载 `col_idx`（间接寻址）、加载 `values`。

**JIT 看到实际矩阵后**（例如某行只有 3 个非零元素在列 5、12、37）：

```asm
; row 42: 3 non-zeros at columns 5, 12, 37
vmulss xmm0, [values + 408], [x + 20]       ; val[i] * x[5]  — 直接地址
vfmadd231ss xmm0, [values + 412], [x + 48]  ; += val[i+1] * x[12]
vfmadd231ss xmm0, [values + 416], [x + 148] ; += val[i+2] * x[37]
vmovss [y + 168], xmm0
```

消除了：`row_ptr` 加载、循环比较跳转、`col_idx` 间接寻址（列索引直接编码为偏移量）。

---

### AOT vs JIT 能力总结

| 优化 | AOT 编译器 | JIT 编译器 |
|------|-----------|-----------|
| 循环展开（次数未知） | ❌ 只能猜测或部分展开 | ✅ 知道确切次数，完全展开 |
| 分支消除（条件未知） | ❌ 必须保留所有分支 | ✅ 只保留实际执行的路径 |
| 常量传播（值未知） | ❌ 变量必须在寄存器中 | ✅ 值嵌入为立即数 |
| 死代码消除（依赖数据） | ❌ 不知道哪些代码不可达 | ✅ 直接跳过不可达路径 |
| 地址计算消除 | ❌ 偏移量运行时计算 | ✅ 偏移量编码为立即数 |
| 数据感知指令选择 | ❌ 同一指令处理所有输入 | ✅ 全零用 vpxord，全满用 vmovdqu8 |

**根本原因**：AOT 编译器的输入是**程序**（有限的静态信息），JIT 编译器的输入是**程序 + 数据**（完整的运行时信息）。数据越"有结构"（有规律、有特殊值），JIT 的优势越大。在本项目中，结构化稀疏度下 JIT 特化版比 intrinsics 快 **1.36–1.62×**，正是因为 38% 的 chunk 是全零或全稠密，JIT 为它们生成了完全不同的指令序列。

---

## JIT 如何"知道"常量值？— 机制详解

一个常见的疑问是：JIT 怎么"知道"数据的值？答案很简单——**你在 C++ 构造函数中把数据传进去，代码生成器像普通 C++ 代码一样读取这些值，然后把它们作为立即数嵌入到机器码中**。

### 两个"时间维度"

JIT 的核心在于理解有两个时间维度：

```
时间 1：JIT 编译时（C++ 代码运行）          时间 2：运行时（机器码执行）
─────────────────────────────────          ───────────────────────────────

bitmask_ = [0x00, 0xFF.., 0x0101..]       （bitmask 数组不被访问）

for chunk in 0..63:                       （无循环 — 已完全展开）
  mask = bitmask_[chunk]   // C++ 读取
  popcnt = popcount(mask)  // C++ 计算    （无 popcnt 指令）
  src_off += popcnt        // C++ 加法    （无 add 指令）

  if mask == 0:            // C++ 分支
    发射: vpxord + store                   vpxord zmm0, zmm0, zmm0
                                           vmovdqu8 [dst+1088], zmm0

  elif mask == ~0:         // C++ 分支
    发射: vmovdqu8 加载 + store             vmovdqu8 zmm0, [src+320]
                                           vmovdqu8 [dst+1152], zmm0

  else:
    发射: mov 立即数 + kmovq + vpexpandb    mov rax, 0x0101010101010101
                                           kmovq k1, rax
                                           vpexpandb zmm0{k1}{z}, [src+352]
                                           vmovdqu8 [dst+1216], zmm0
```

关键：**C++ 的 `for` 循环、`if` 分支、`src_off += popcnt` 全部在构造时执行完毕**。它们的输出是机器码字节。当 `jit_func(&args)` 被调用时，这些循环和分支已经不存在——只剩下发射出的指令。

### 本项目中的实际代码

```cpp
class jit_decompress_specialized_t : public Xbyak::CodeGenerator {
    int blocks_;
    const uint64_t* bitmask_;   // ← 实际掩码数据作为参数传入

    // 构造函数接收运行时数据
    jit_decompress_specialized_t(int blocks, const uint64_t* bitmask)
        : blocks_(blocks), bitmask_(bitmask) {  // ← 存为成员变量
        generate();   // ← 代码生成在构造函数中发生
    }

    void generate() {
        // 这是一个普通的 C++ 函数。它只执行一次（构造时）。
        // 它读取实际的掩码值并发射机器码。

        int src_off = 0;   // ← C++ 变量，不是寄存器！

        for (int block = 0; block < blocks_; ++block) {
            for (int chunk = 0; chunk < 64; ++chunk) {

                // *** 关键代码 ***
                // bitmask_ 是真实数据 → mask 是已知的 C++ 值
                uint64_t mask = bitmask_[block * 64 + chunk];
                int popcnt = __builtin_popcountll(mask);

                if (mask == 0) {
                    // C++ 看到 mask == 0 → 发射清零存储代码
                    vpxord(zmm0, zmm0, zmm0);
                    // src_off += 0; （不推进，在 JIT 编译时已计算）

                } else if (mask == 0xFFFFFFFFFFFFFFFFULL) {
                    // C++ 看到 mask == 全 1 → 发射直接拷贝代码
                    vmovdqu8(zmm0, ptr[reg_src + src_off]);
                    //                          ^^^^^^^
                    //                          src_off 是 C++ 的 int
                    //                          变成指令中的立即数位移

                } else {
                    // C++ 看到部分掩码 → 将掩码嵌入为立即数
                    mov(rax, mask);
                    //       ^^^^
                    //       发射: 48 B8 01 01 01 01 01 01 01 01
                    //       掩码的值直接编码为机器码的字节！
                    kmovq(k1, rax);
                    vpexpandb(zmm0 | k1 | T_z, ptr[reg_src + src_off]);
                }

                vmovdqu8(ptr[reg_dst + dst_off], zmm0);

                src_off += popcnt;  // ← C++ 加法，不是 asm 指令！
                //                     这个值在运行时已经不存在。
                //                     它在代码生成过程中被消耗掉了。
            }
        }
        ret();
    }
};
```

### 简易示例：常量加法

```cpp
#include "xbyak/xbyak.h"

// 一个 JIT 函数：将一个常量（JIT 编译时已知）加到参数上
class AddConst : public Xbyak::CodeGenerator {
public:
    AddConst(int value) {   // ← 'value' 是运行时数据，但在 JIT 编译时已知
        // Linux ABI：第一个参数在 rdi，返回值在 rax
        lea(rax, ptr[rdi + value]);   // value 变成立即数位移
        //                   ^^^^^
        //                   发射: lea rax, [rdi + 42]
        //                   数字 42 被烧进了机器码的字节中
        ret();
    }
};

int main() {
    int secret = 42;  // 假设这个值来自文件、用户输入等

    AddConst jit(secret);              // JIT 编译：secret → 立即数
    auto f = jit.getCode<int(*)(int)>();
    jit.ready();

    printf("%d\n", f(10));   // 输出 52 — 没有内存加载，没有变量
    printf("%d\n", f(100));  // 输出 142
}
```

生成的机器码：

```asm
lea rax, [rdi + 0x2A]   ; 0x2A = 42，硬编码在指令字节中
ret
```

**没有变量、没有内存加载、没有寄存器保存着 42**。值 42 是指令编码的一部分——就像 `mov rax, 5` 把数字 5 嵌入指令一样。JIT "知道"常量是因为**生成机器码的 C++ 代码可以像普通变量一样读取它，并将它用作立即数操作数**。

### 本质总结

```
普通程序：                            JIT 程序：
  读取数据 → 计算 → 输出结果            读取数据 → 生成代码 → 执行代码 → 输出结果
                                              ↑
                                        数据在这里被"消耗"，
                                        变成了指令的一部分
```

JIT 就是一个**以数据为输入、以机器码为输出的程序**。C++ 的 `if`/`for`/变量赋值在代码生成阶段全部执行完毕，运行时执行的只是生成出来的精简指令序列。这就是为什么 JIT 能做到 AOT 编译器做不到的事——它在生成代码的那一刻，同时拥有程序逻辑和数据的完整信息。

---

## 第四部分：JIT 的分支消除——只生成需要的代码

### 核心理解

JIT 内核**只为特定条件生成对应的那一条分支代码路径，其它分支根本不会被生成**。

这是 JIT 相对于 AOT（提前编译）最本质的优势之一。

### AOT 编译器 vs JIT 编译器的分支处理

#### AOT（传统静态编译）

AOT 编译器在编译时**不知道运行时的数据值**，因此它必须为所有可能的分支都生成代码，并在运行时通过条件判断（`cmp` + `je`/`jne`/`jmp`）来选择执行哪一条路径：

```
; AOT 生成的代码——所有分支都存在
    cmp  [chunk_type], ZERO
    je   .handle_zero          ; 跳转到零块处理
    cmp  [chunk_type], DENSE
    je   .handle_dense         ; 跳转到稠密块处理
    jmp  .handle_partial       ; 否则处理部分块

.handle_zero:
    vpxord zmm0, zmm0, zmm0   ; 零块逻辑
    vmovdqu8 [dst], zmm0
    jmp  .next_chunk

.handle_dense:
    vmovdqu8 zmm0, [src]      ; 稠密块逻辑
    vmovdqu8 [dst], zmm0
    jmp  .next_chunk

.handle_partial:
    kmovq  k1, [mask]         ; 部分块逻辑
    vpexpandb zmm0{k1}{z}, [src]
    vmovdqu8 [dst], zmm0
    jmp  .next_chunk
```

**问题**：即使某个块永远是零块，跳转到 `.handle_dense` 和 `.handle_partial` 的代码仍然存在于二进制文件中，CPU 仍需读取和解码这些无用指令。更重要的是，每个块都需要执行 `cmp`/`je`/`jmp` 分支判断，浪费了时钟周期。

#### JIT（运行时编译）

JIT 在**代码生成阶段**就已经知道每个块的类型（因为位掩码是已知数据）。所以它**只为实际需要的类型生成指令**，不需要的分支代码**根本不存在于生成的机器码中**：

```c++
// C++ 代码生成阶段（JIT 编译时执行）
for (int i = 0; i < num_chunks; i++) {
    if (masks[i] == 0x0) {
        // 该块是零块 → 只生成零块指令
        vpxord(zmm0, zmm0, zmm0);
        vmovdqu8(ptr[dst + i*64], zmm0);
    } else if (masks[i] == 0xFFFFFFFFFFFFFFFF) {
        // 该块是稠密块 → 只生成拷贝指令
        vmovdqu8(zmm0, ptr[src + offset]);
        vmovdqu8(ptr[dst + i*64], zmm0);
    } else {
        // 部分块 → 只生成 expand 指令
        mov(rax, masks[i]);   // 掩码作为立即数
        kmovq(k1, rax);
        vpexpandb(zmm0 | k1 | T_z, ptr[src + offset]);
        vmovdqu8(ptr[dst + i*64], zmm0);
    }
}
```

假设有 4 个块，掩码分别为 `[0x0, 0xFF...FF, 0x0, 0x00FF00FF]`，JIT 生成的机器码**仅包含**：

```
; JIT 生成的代码——扁平化指令流，无分支
; 块 0：零块
    vpxord zmm0, zmm0, zmm0
    vmovdqu8 [dst + 0], zmm0

; 块 1：稠密块
    vmovdqu8 zmm0, [src + 0]
    vmovdqu8 [dst + 64], zmm0

; 块 2：零块
    vpxord zmm0, zmm0, zmm0
    vmovdqu8 [dst + 128], zmm0

; 块 3：部分块
    mov rax, 0x00FF00FF
    kmovq k1, rax
    vpexpandb zmm0{k1}{z}, [src + 64]
    vmovdqu8 [dst + 192], zmm0
```

**关键区别**：
- **没有 `cmp`** — 不需要比较块类型
- **没有 `je`/`jne`/`jmp`** — 不需要条件跳转
- **没有未使用的代码** — 零块不会生成 expand 指令，稠密块不会生成 kmov 指令
- **所有偏移量都是立即数** — `[dst + 0]`、`[src + 64]` 等在生成时就已确定

### 对性能的影响

| 方面 | AOT | JIT |
|------|-----|-----|
| 分支指令 | 每个块 2-3 条 `cmp`/`jmp` | **零条** |
| 分支预测失败惩罚 | 可能发生（~15 周期/次） | **不可能发生** |
| 指令缓存利用 | 所有分支代码占用 I-cache | **只有实际执行的代码占用** |
| 代码膨胀 | 全部分支代码都在二进制中 | **只有需要的指令** |

### 类比

可以这样理解 AOT vs JIT 的分支处理：

- **AOT** 像是印刷一本菜谱书：中餐、西餐、日料的做法全部印在里面，每次做饭时翻到对应的页面（分支跳转）
- **JIT** 像是知道今晚要做什么菜之后，只打印那一道菜的做法——其它菜的做法根本不会出现在纸上

---

## 第五部分：三个时间阶段——AOT 编译、JIT 编译、JIT 运行时

### 核心问题

JIT 程序中有一个容易混淆但极其重要的概念：**代码中的每一行，到底是在哪个时刻执行的？** 共有三个时间阶段：

```
阶段 1：AOT 编译期               阶段 2：JIT 编译期               阶段 3：JIT 运行时
(g++ 编译 .cpp)                 (构造函数执行)                  (生成的机器码被调用)
────────────────                ────────────────                ────────────────
输入：C++ 源码                   输入：运行时数据（掩码）          输入：压缩数据指针
输出：宿主程序二进制               输出：机器码字节流               输出：解压后的数据
执行者：g++ 编译器                执行者：CPU 运行 C++ 代码        执行者：CPU 运行生成的机器码
```

### 以 `jit_decompress_specialized_t` 为例的逐行分析

下面用颜色标注每一行代码属于哪个阶段：

```cpp
// ═══════════════════════════════════════════════════════
// 【阶段 1：AOT 编译期 — g++ 执行】
// g++ 把这整个 .hpp 文件编译成 x86-64 宿主程序。
// 类定义、成员变量布局、函数签名全部在此阶段确定。
// ═══════════════════════════════════════════════════════

class jit_decompress_specialized_t : public Xbyak::CodeGenerator {
    // ↓ 这些是 AOT 编译期确定的类型和内存布局
    Xbyak::Reg64 param1 = rdi;        // AOT: 编译器知道 rdi 的编码
    Xbyak::Reg64 reg_src = r8;        // AOT: r8 寄存器编码为常量
    Xbyak::Reg64 reg_dst = r9;        // AOT: r9 寄存器编码为常量
    Xbyak::Reg64 reg_tmp = rax;       // AOT: rax 寄存器编码为常量

    int blocks_;                       // AOT: 布局确定，值在阶段 2 赋值
    const uint64_t* bitmask_;          // AOT: 指针类型确定，值在阶段 2 赋值

    // ↓ 构造函数的"壳"在 AOT 期编译，但"执行"在阶段 2
    jit_decompress_specialized_t(int blocks, const uint64_t* bitmask)
        : Xbyak::CodeGenerator(4096 * 256),
          blocks_(blocks), bitmask_(bitmask)
    {
        generate();    // AOT 编译此调用指令；实际执行在阶段 2
    }

    void generate() {
        // ═══════════════════════════════════════════════════════
        // 【阶段 2：JIT 编译期 — 构造函数被调用时，CPU 运行这段 C++】
        // 此函数内的所有 C++ 语句（for、if、变量赋值）在此阶段执行。
        // Xbyak 的 mov()、vpexpandb() 等调用实际上是在往内存缓冲区
        // 写入机器码字节——它们是"代码生成器"，不是"指令执行"。
        // ═══════════════════════════════════════════════════════

        // ────── 阶段 2 执行：调用 Xbyak 发射 2 条 mov 指令的机器码 ──────
        // 这两条 mov 生成的机器码属于【阶段 3】，但生成动作属于【阶段 2】
        mov(reg_src, ptr[param1 + offsetof(jit_specialized_params_t, compressed_buf)]);
        mov(reg_dst, ptr[param1 + offsetof(jit_specialized_params_t, decomp_buf)]);

        // ────── 阶段 2 执行：C++ 变量初始化 ──────
        // src_off 是一个纯 C++ 变量，只存在于阶段 2。
        // 它永远不会变成寄存器或内存位置——它会变成立即数。
        int src_off = 0;    // 【阶段 2 变量】

        // ────── 阶段 2 执行：C++ for 循环 ──────
        // 这个 for 循环在阶段 2 执行完毕。阶段 3 没有循环！
        for (int block = 0; block < blocks_; ++block) {        // 【阶段 2 循环】
            int bm_base = block * 64;       // 【阶段 2 计算】
            int dst_base = block * 4096;    // 【阶段 2 计算】

            for (int chunk = 0; chunk < 64; ++chunk) {          // 【阶段 2 循环】

                // ────── 阶段 2 执行：读取实际掩码数据 ──────
                // bitmask_ 指针在阶段 2 可用（构造函数传入）
                // mask 是一个 C++ 变量，在阶段 2 已知
                uint64_t mask = bitmask_[bm_base + chunk];      // 【阶段 2 数据读取】
                int popcnt = __builtin_popcountll(mask);        // 【阶段 2 计算】
                int dst_off = dst_base + chunk * 64;            // 【阶段 2 计算】

                Xbyak::Zmm zr = Xbyak::Zmm(chunk & 3);         // 【阶段 2 寄存器选择】
                Xbyak::Opmask kr = Xbyak::Opmask(1 + (chunk & 3)); // 【阶段 2】

                // ────── 阶段 2 执行：C++ if 分支 ──────
                // 这个 if/else 在阶段 2 执行！只有一条分支的代码会被发射。
                // 其它分支的机器码根本不会被生成。
                if (mask == 0) {                                // 【阶段 2 分支判断】
                    // 阶段 2 调用 Xbyak → 发射阶段 3 机器码
                    vpxord(zr, zr, zr);                         // →【阶段 3 指令】
                    zero_chunks_++;                             // 【阶段 2 统计】

                } else if (mask == 0xFFFFFFFFFFFFFFFFULL) {     // 【阶段 2 分支判断】
                    // src_off 是阶段 2 的 C++ int
                    // 但 ptr[reg_src + src_off] 中，src_off 被编码为立即数
                    vmovdqu8(zr, ptr[reg_src + src_off]);       // →【阶段 3 指令】
                    full_chunks_++;                             // 【阶段 2 统计】

                } else {                                        // 【阶段 2 分支】
                    mov(reg_tmp, mask);                         // →【阶段 3 指令】
                    //          ^^^^
                    //   mask 是阶段 2 的 uint64_t 变量
                    //   被编码为 "mov rax, 0x..." 中的 64 位立即数
                    //   ——阶段 2 的变量在这里"穿越"到了阶段 3

                    kmovq(kr, reg_tmp);                         // →【阶段 3 指令】
                    vpexpandb(zr | kr | T_z,
                              ptr[reg_src + src_off]);          // →【阶段 3 指令】
                    //                       ^^^^^^^
                    //   src_off (阶段 2 的 int) → 编码为指令中的位移立即数
                    partial_chunks_++;                           // 【阶段 2 统计】
                }

                vmovdqu8(ptr[reg_dst + dst_off], zr);           // →【阶段 3 指令】
                //                      ^^^^^^^
                //   dst_off (阶段 2 的 int) → 编码为指令中的位移立即数

                src_off += popcnt;  // 【阶段 2 C++ 加法】—— 运行时此变量不存在！
            }

            // ────── 阶段 2 执行：C++ 对齐计算 ──────
            int misalign = src_off & 0x3F;                      // 【阶段 2 计算】
            if (misalign) src_off += (64 - misalign);           // 【阶段 2 计算】
            // 阶段 3 没有任何对齐指令！
        }

        ret();  // → 【阶段 3 指令】：函数返回
        code_size_ = getSize();  // 【阶段 2】：记录生成的字节数
    }
};
```

### 三阶段职责划分表

| 代码元素 | 阶段 1 (AOT) | 阶段 2 (JIT 编译) | 阶段 3 (JIT 运行) |
|---------|-------------|------------------|------------------|
| `class` 定义、成员布局 | **确定** | — | — |
| `Xbyak::Reg64 reg_src = r8` | **编译为常量** | 用于选择寄存器编码 | 生成的指令使用 r8 |
| `blocks_`, `bitmask_` 成员变量 | 类型确定 | **赋值并读取** | 不存在 |
| `generate()` 函数体 | **编译为机器码** | **执行** | 不存在 |
| `int src_off = 0` | 编译为栈变量 | **创建、累加、销毁** | 变为立即数常量 |
| `for (block...)` / `for (chunk...)` | 编译为循环指令 | **执行所有迭代** | 不存在（全部展开） |
| `if (mask == 0)` | 编译为比较跳转 | **执行分支判断** | 不存在（只留一条路径） |
| `bitmask_[chunk]` | 编译为内存加载 | **读取实际掩码值** | 不存在（值嵌入指令） |
| `__builtin_popcountll(mask)` | 编译为 popcnt | **执行并得到结果** | 不存在（结果为立即数） |
| `vpxord(zr, zr, zr)` | 编译为 Xbyak 调用 | **执行 → 发射字节** | CPU 执行 `vpxord` |
| `mov(reg_tmp, mask)` | 编译为 Xbyak 调用 | **执行 → 发射 `mov rax, IMM`** | CPU 执行 `mov rax, 0x...` |
| `vpexpandb(... ptr[reg_src + src_off])` | 编译为 Xbyak 调用 | **执行 → 发射 vpexpandb** | CPU 执行 `vpexpandb zmm, [r8+IMM]` |
| `src_off += popcnt` | 编译为 add 指令 | **执行 C++ 加法** | 不存在 |
| `ret()` | 编译为 Xbyak 调用 | **执行 → 发射 `ret`** | CPU 执行 `ret` |

### 数据如何"穿越"阶段

JIT 的精髓在于阶段 2 的数据（C++ 变量）如何变成阶段 3 的常量（机器码字节）：

```
阶段 2 (C++ 世界)                    →    阶段 3 (机器码世界)
─────────────────                         ─────────────────

uint64_t mask = 0x00FF00FF00FF00FF   →    mov rax, 0x00FF00FF00FF00FF
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          mask 的值被编码为 10 字节机器码：
                                          48 B8 FF 00 FF 00 FF 00 FF 00

int src_off = 320                    →    vpexpandb zmm0{k1}{z}, [r8 + 0x140]
                                                                      ^^^^^
                                          320 (0x140) 被编码为 disp32 位移

int dst_off = 4160                   →    vmovdqu8 [r9 + 0x1040], zmm0
                                                        ^^^^^^
                                          4160 (0x1040) 被编码为 disp32 位移

mask == 0                            →    （vpxord 被发射，vpexpandb 不被发射）
                                          C++ 的 if 判断结果决定了哪些
                                          机器码字节存在于阶段 3
```

### 时间线可视化

```
时间 ──────────────────────────────────────────────────────────────────────►

      ┌───────────┐         ┌─────────────────┐     ┌──────────────────┐
      │ 阶段 1     │         │ 阶段 2           │     │ 阶段 3            │
      │ g++ 编译   │         │ JIT 编译         │     │ JIT 运行          │
      │           │         │                 │     │                  │
      │ 输入:      │         │ 输入:            │     │ 输入:             │
      │  .hpp源码  │         │  blocks=4       │     │  &params         │
      │           │         │  bitmask=[...]   │     │  (src,dst指针)    │
      │           │         │                 │     │                  │
      │ 做了什么:   │         │ 做了什么:         │     │ 做了什么:          │
      │ ·类型检查   │         │ ·读 bitmask 值   │     │ ·从 params 加载   │
      │ ·模板实例化  │         │ ·C++ for 循环    │     │   src/dst 指针    │
      │ ·编译generate│        │ ·C++ if 分支     │     │ ·执行 vpxord      │
      │  函数为x86  │         │ ·计算 popcnt     │     │ ·执行 vmovdqu8    │
      │ ·编译Xbyak  │         │ ·计算 src_off    │     │ ·执行 vpexpandb   │
      │  库函数    │         │ ·调用Xbyak发射    │     │ ·执行 vmovdqu8    │
      │           │         │  机器码字节       │     │ ·ret 返回         │
      │           │         │                 │     │                  │
      │ 输出:      │         │ 输出:            │     │ 输出:             │
      │ a.out 二进制│         │ 可执行的机器码     │     │ 解压后的数据       │
      │ (含generate │         │ 字节流（在堆上）   │     │                  │
      │  的机器码)  │         │                 │     │                  │
      └─────┬─────┘         └────────┬────────┘     └──────────────────┘
            │                        │
            │  运行 a.out             │  调用 jit_func(&params)
            │  构造 jit 对象          │  （数百万次）
            ▼                        ▼
       只发生一次                只发生一次              发生 N 次
       (开发/构建时)            (模型加载时)           (每次推理调用)
```

### 关键洞察

1. **`generate()` 函数有双重身份**：
   - 阶段 1 视角：它是一段被 g++ 编译的 C++ 函数，被编译成普通的 x86 指令（遍历数组、做 if 判断、调用 Xbyak API）
   - 阶段 2 视角：它是一个**代码生成器**——它执行时不产生"结果"，而是产生"代码"

2. **Xbyak 调用 ≠ 指令执行**：
   - `vpexpandb(zmm0 | k1 | T_z, ptr[r8 + 320])` 在阶段 2 执行时，只是往内存缓冲区写了约 10 个字节（EVEX 前缀 + opcode + ModRM + SIB + disp32）
   - 这些字节在阶段 3 被 CPU 解码并执行为真正的 AVX-512 指令

3. **C++ 变量的"消亡"**：
   - `src_off`、`mask`、`popcnt`、`dst_off` 在阶段 2 结束后全部销毁
   - 但它们的**值**以立即数的形式"永生"在了生成的机器码中
   - 阶段 3 中没有任何对应的寄存器或内存位置——值已融入指令编码

4. **C++ 控制流的"消亡"**：
   - `for` 循环在阶段 2 执行了 `blocks_ × 64` 次迭代，每次发射几条指令
   - 阶段 3 看到的是一串**没有循环的平铺指令**
   - `if (mask == 0)` 在阶段 2 做完判断，阶段 3 中**没有比较和跳转指令**

5. **三阶段的不对称性**：

   | | 阶段 1 | 阶段 2 | 阶段 3 |
   |--|--------|--------|--------|
   | 执行次数 | 1 次（构建时） | 1 次（初始化时） | **百万次**（推理时） |
   | 代码量 | ~100 行 C++ | ~100 行 C++ 被执行 | ~数百条机器指令 |
   | 有循环？ | 有（被编译） | 有（被执行） | **无**（已展开） |
   | 有分支？ | 有（被编译） | 有（被执行） | **无**（已消除） |
   | 有popcnt？| 有（builtin） | 有（被执行） | **无**（已变为立即数） |

   这就是 JIT 的核心价值：**阶段 2 的开销只发生一次，但阶段 3 的每一次调用都从中受益**。

---

## 第六部分：同一逻辑放在阶段 2 还是阶段 3？—— JIT 的核心设计决策

### 问题引出

在 JIT 编程中，有一个反复出现的设计问题：**同一段逻辑，既可以在阶段 2（JIT 编译期）用 C++ 执行，结果"烧"进机器码；也可以在阶段 3（JIT 运行时）用发射的指令执行。那该放在哪里？**

以 inter-block 对齐计算为例，V2 和 V3 做了截然不同的选择：

```
同一个目标：将 compressed_src 指针对齐到 64 字节边界

V2 选择 → 放在阶段 3（运行时）：
    mov(reg_align_tmp, reg_src);
    sub(reg_align_tmp, compressed_ptr);
    neg(reg_align_tmp);
    and_(reg_align_tmp, 0x3f);
    add(reg_src, reg_align_tmp);
    // → 5 条指令，每次调用都要执行

V3 选择 → 放在阶段 2（JIT 编译时）：
    int misalign = src_off & 0x3F;
    if (misalign) src_off += (64 - misalign);
    // → 0 条运行时指令，结果已编码为后续指令的立即数
```

**为什么 V3 能放在阶段 2 而 V2 不能？** 答案就是决策的关键所在。

### 决策原则：所有输入在阶段 2 是否已知？

这是唯一的根本性判断：

```
                  ┌──────────────────────────┐
                  │  这段逻辑的所有输入        │
                  │  在 JIT 编译期是否已知？    │
                  └─────────┬────────────────┘
                            │
                   ┌────────┴────────┐
                   │                 │
                  是                 否
                   │                 │
                   ▼                 ▼
           ┌─────────────┐   ┌──────────────┐
           │ 可以放在阶段 2 │   │ 必须放在阶段 3 │
           │（C++ 计算，   │   │（发射指令，    │
           │ 结果为立即数） │   │ 运行时执行）   │
           └─────────────┘   └──────────────┘
```

### 案例逐一分析

#### 案例 1：对齐计算

**V2 的情况**：`reg_src` 是一个运行时寄存器，它的值随每次调用传入的 `compressed_buf` 而不同。JIT 编译期不知道 `compressed_buf` 的具体地址值 → **输入未知 → 必须放阶段 3**。

**V3 的情况**：`src_off` 是一个 C++ 的 `int` 变量，在 `generate()` 执行过程中通过 `__builtin_popcountll` 累加而来。所有 `popcnt` 的输入 `mask` 来自 `bitmask_[]`，而 `bitmask_` 在构造时传入 → **输入全知 → 可以放阶段 2**。

```cpp
// V3 中 src_off 的"一生"（全在阶段 2）：
int src_off = 0;                               // 阶段 2：初始化
for (chunk...) {
    uint64_t mask = bitmask_[chunk];            // 阶段 2：读已知数据
    int popcnt = __builtin_popcountll(mask);    // 阶段 2：计算已知结果
    // ... 发射指令，用 src_off 作为立即数 ...
    src_off += popcnt;                          // 阶段 2：累加已知值
}
int misalign = src_off & 0x3F;                  // 阶段 2：已知值的位运算
if (misalign) src_off += (64 - misalign);       // 阶段 2：已知值的加法
// src_off 从未成为寄存器，始终是 C++ 的 int
```

#### 案例 2：popcnt 计算

| | 输入 | 阶段 2 已知？ | 放在哪里？ |
|--|------|-------------|----------|
| V1/V2 | `reg_comp_mask_tmp`（运行时寄存器） | ❌ 掩码从内存动态加载 | **阶段 3**：`popcnt(reg, reg)` |
| V3 | `bitmask_[chunk]`（C++ 数组元素） | ✅ 构造时传入 | **阶段 2**：`__builtin_popcountll(mask)` |

#### 案例 3：掩码加载

| | 输入 | 阶段 2 已知？ | 放在哪里？ |
|--|------|-------------|----------|
| V1/V2 | 掩码值在运行时从 `bitmask_ptr` 加载 | ❌ | **阶段 3**：`mov reg, [mem]` + `kmovq k, reg` |
| V3 | 掩码值在构造时已知 | ✅ | **阶段 2**：C++ 读掩码 → `mov(reg, mask)` 将值嵌入为 64 位立即数 |

#### 案例 4：循环控制

| | 输入 | 阶段 2 已知？ | 放在哪里？ |
|--|------|-------------|----------|
| V2 | `blocks_`（构造时传入） | ✅ | 可以放阶段 2（全展开），但 V2 **选择放阶段 3**（见下文） |
| V3 | `blocks_`（构造时传入） | ✅ | **阶段 2**：C++ `for` 循环全展开 |

**等一下——V2 的 `blocks_` 明明在阶段 2 已知，为什么还放在阶段 3？** 这引出了第二条原则。

### 第二条原则："可以"与"应该"的区别

即使输入在阶段 2 全部已知，放在阶段 2 也不一定更好。需要权衡 **代价**：

```
阶段 2 计算的代价：
  ✅ 运行时 0 条指令（免费）
  ❌ 可能导致代码膨胀（每次迭代都展开为独立指令 → 更多机器码字节）

阶段 3 执行的代价：
  ✅ 代码紧凑（循环复用同一块指令）
  ❌ 每次调用都要执行 N 条指令
```

**权衡公式**（非严格，但提供直觉）：

```
净收益 = (阶段 3 每次调用省的周期 × 调用次数) − (阶段 2 代码膨胀的 I-cache 代价)
```

#### V2 为什么对 block 循环选择阶段 3？

```
                          V2 的 block 循环：放阶段 2 vs 阶段 3

  放阶段 2（全展开）：                    放阶段 3（运行时循环）：
  ┌──────────────────────┐              ┌──────────────────────┐
  │ blocks=1000 时：       │              │ blocks=1000 时：       │
  │  1000 × 16组 × ~40条   │              │ 16组 × ~40条 ≈ 640条   │
  │  ≈ 640,000 条指令       │              │ + 循环控制 ~3条         │
  │  ≈ ~4MB 机器码          │              │ ≈ ~4KB 机器码           │
  │                        │              │                        │
  │  I-cache ≈ 32KB        │              │  I-cache ≈ 32KB        │
  │  → 代码远超 I-cache     │              │  → 代码在 I-cache 中    │
  │  → 大量 I-cache miss    │              │  → 第 2 次迭起就命中    │
  │  → 反而更慢！           │              │  → 每次迭代 ~3 条额外    │
  └──────────────────────┘              └──────────────────────┘
                                         ↑
                                         V2 选择这个 ✅
```

3 条循环控制指令（`dec + jnz + add`）的开销远小于 I-cache miss 的惩罚（~100 周期/次）。

#### V3 为什么又选择全展开？

因为 V3 有 **V2 没有的理由**——每个 chunk 的指令序列**不同**（零块 2 条、稠密块 2 条、部分块 4 条）。如果用运行时循环，就必须在循环体内加分支判断 chunk 类型，这正是 V3 要消除的东西。

```
V3 的全展开是必要的：每个 chunk 生成不同的指令

chunk 0:  vpxord + store              ← 零块
chunk 1:  mov + kmovq + vpexpandb + store  ← 部分块
chunk 2:  vmovdqu8 + store            ← 稠密块
chunk 3:  vpxord + store              ← 零块
...

这些不同的指令序列无法放入统一的循环体！
（除非加 if/else 分支 → 那就退化回了 AOT 逻辑）
```

但 V3 也有 I-cache 的代价。当 blocks 很大（>几十个）时，代码膨胀会吃掉特化带来的收益。这就是为什么在适用场景中建议 "掩码固定 + block 数极多 → 用 V2"。

### 决策框架总结

```
┌───────────────────────────────────────────────────────┐
│            JIT 设计决策：放阶段 2 还是阶段 3？          │
├───────────────────────────────────────────────────────┤
│                                                       │
│  步骤 1：输入是否在 JIT 编译时全部已知？                 │
│     │                                                 │
│     ├─ 否 → 【必须放阶段 3】（别无选择）                │
│     │       例：compressed_buf 地址、运行时传入的指针     │
│     │                                                 │
│     └─ 是 → 进入步骤 2                                 │
│                                                       │
│  步骤 2：放阶段 2 会导致多少代码膨胀？                   │
│     │                                                 │
│     ├─ 膨胀可控（<= I-cache 容量）                     │
│     │   → 【推荐放阶段 2】                             │
│     │     运行时每省 1 条指令 × 百万次调用 = 巨大收益     │
│     │     例：src_off 累加、popcnt、对齐计算             │
│     │                                                 │
│     └─ 膨胀严重（全展开远超 I-cache）                   │
│         → 进入步骤 3                                   │
│                                                       │
│  步骤 3：展开后每次迭代的指令序列是否相同？               │
│     │                                                 │
│     ├─ 相同 → 【放阶段 3，用运行时循环】                │
│     │         例：V2 的 block 循环                     │
│     │         （循环体一样，循环开销仅 2-3 条指令）       │
│     │                                                 │
│     └─ 不同 → 【放阶段 2，接受膨胀，但限制规模】         │
│               例：V3 的 chunk 展开（每个 chunk 指令不同） │
│               注意：blocks 过多时应退回 V2 方案          │
│                                                       │
└───────────────────────────────────────────────────────┘
```

### 本项目中每个逻辑的阶段选择一览

| 逻辑 | V1 | V2 | V3 | 为什么 V3 能/该 移到阶段 2 |
|------|----|----|-----|--------------------------|
| 源偏移累加 (`src += popcnt`) | 阶段 3 | 阶段 3 | **阶段 2** | `popcnt` 的输入 `mask` 在阶段 2 已知 |
| popcnt 计算 | 阶段 3 | 阶段 3 | **阶段 2** | `mask` 值来自 `bitmask_[]`，阶段 2 可读 |
| 掩码加载 | 阶段 3 | 阶段 3 | **阶段 2** | 掩码值直接嵌入为立即数 |
| 对齐计算 | 阶段 3 | 阶段 3 | **阶段 2** | `src_off` 是阶段 2 的 C++ int |
| 零块/稠密块判断 | N/A | N/A | **阶段 2** | `mask` 值已知，C++ `if` 判断 |
| block 循环 | 阶段 2¹ | **阶段 3** | 阶段 2¹ | V2 为 I-cache 选阶段 3；V3 因指令不同必须展开 |
| chunk 内展开 (×4) | 阶段 2¹ | 阶段 2¹ | 阶段 2¹ | 三版均在 JIT 编译期展开（规模可控） |
| 从 params 加载 src/dst 指针 | **阶段 3** | **阶段 3** | **阶段 3** | 指针值每次调用不同 → 必须阶段 3 |
| vpexpandb 执行 | **阶段 3** | **阶段 3** | **阶段 3** | 压缩数据每次调用不同 → 必须阶段 3 |
| vmovdqu8 store | **阶段 3** | **阶段 3** | **阶段 3** | 必须在运行时写入目标缓冲区 |

¹ "阶段 2" 指 C++ `for` 循环在 JIT 编译期执行，结果是展开后的指令。

### 类比：菜谱 vs 购物清单

```
阶段 2 = 你在家写购物清单           阶段 3 = 你去超市买东西
──────────────────────           ──────────────────────

已知信息：                         已知信息：
  · 冰箱里有什么（掩码数据）          · 超市的货架地址（compressed_buf）
  · 家里几口人（blocks 数）          · 购物车容量（decomp_buf）
  · 每人吃多少（popcnt）

能在家做的事：                      必须去超市做的事：
  · 计算总量（src_off 累加）          · 拿起货物放入购物车
  · 划掉不需要的（死代码消除）         · 走到收银台结账
  · 写好"第 3 排左数第 5 个"          · 把东西搬回家
    （地址编码为立即数）

     ↓ 写好的清单就是机器码            ↓ 照着清单执行即可

去超市时不需要：                    
  · 重新数家里几口人（no popcnt）
  · 重新检查冰箱（no mask load）
  · 计算需要买多少（no add）
  · 决定去哪个货架（地址已写死）
```

### 核心结论

> **JIT 编程的设计本质是在两个时间维度间划分计算**。每一段逻辑都有一个最优的"安放时刻"——取决于输入的可知性和展开的代价。将尽可能多的计算"上移"到阶段 2，是 JIT 获得性能优势的根本手段；但必须警惕代码膨胀超过 I-cache 容量的拐点。

V2 和 V3 的区别不在于算法不同，而在于**同一个算法被放在了不同的时间维度执行**：

```
V2:  mask 未知 → popcnt 在阶段 3 → 前缀和在阶段 3 → 对齐在阶段 3
V3:  mask 已知 → popcnt 在阶段 2 → 累加在阶段 2   → 对齐在阶段 2
     ^^^^^^^^
     唯一的区别：掩码何时可知
     这一个条件的改变，把一整条计算链从阶段 3 提升到了阶段 2
```
