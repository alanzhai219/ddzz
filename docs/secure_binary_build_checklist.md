# Secure Binary Build Checklist

从 clang-tidy、gcc/clang/msvc、BinSkim 三个层次构建安全的 binary。

## 1. Clang-Tidy — 源码静态分析

Clang-Tidy 在编译前对源码做语义级检查，捕获编译器警告无法覆盖的深层缺陷。

### 安全相关 Checker 分类

| 类别 | 典型 Checker | 检查内容 |
|------|-------------|---------|
| `bugprone-*` | `bugprone-use-after-move`, `bugprone-integer-division`, `bugprone-dangling-handle` | 常见编程错误 |
| `clang-analyzer-security.*` | `security.insecureAPI.gets`, `security.FloatLoopCounter` | 不安全 API、安全反模式 |
| `clang-analyzer-core.*` | `core.NullDereference`, `core.DivideZero`, `core.UndefinedBinaryOperatorResult` | 空指针、除零、未定义行为 |
| `cert-*` | `cert-err33-c`, `cert-msc30-c`, `cert-str34-c` | CERT 安全编码规范 |
| `cppcoreguidelines-*` | `cppcoreguidelines-pro-bounds-*`, `cppcoreguidelines-pro-type-*`, `cppcoreguidelines-no-malloc` | C++ Core Guidelines 的类型/边界安全 |
| `misc-*` | `misc-redundant-expression`, `misc-unused-parameters` | 冗余/死代码 |

### 推荐配置（`.clang-tidy`）

```yaml
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-security.*,
  clang-analyzer-core.*,
  cert-*,
  cppcoreguidelines-no-malloc,
  cppcoreguidelines-pro-bounds-*,
  cppcoreguidelines-pro-type-*,
WarningsAsErrors: '*'
```

### 完整配置文件（`.clang-tidy`）

将以下内容保存为项目根目录下的 `.clang-tidy` 文件：

```yaml
---
Checks: >
  -*,
  bugprone-*,
  clang-analyzer-security.*,
  clang-analyzer-core.*,
  clang-analyzer-cplusplus.*,
  clang-analyzer-unix.*,
  cert-*,
  cppcoreguidelines-no-malloc,
  cppcoreguidelines-pro-bounds-array-to-pointer-decay,
  cppcoreguidelines-pro-bounds-constant-array-index,
  cppcoreguidelines-pro-bounds-pointer-arithmetic,
  cppcoreguidelines-pro-type-const-cast,
  cppcoreguidelines-pro-type-cstyle-cast,
  cppcoreguidelines-pro-type-reinterpret-cast,
  cppcoreguidelines-pro-type-union-access,
  misc-redundant-expression,
  misc-unused-parameters,
  misc-misplaced-const,
  modernize-use-nullptr,
  readability-implicit-bool-conversion,

WarningsAsErrors: >
  bugprone-*,
  clang-analyzer-security.*,
  clang-analyzer-core.*,
  cert-*,

HeaderFilterRegex: '.*'

CheckOptions:
  - key: bugprone-argument-comment.StrictMode
    value: true
  - key: bugprone-assert-side-effect.AssertMacros
    value: 'assert,ASSERT,Q_ASSERT'
  - key: cppcoreguidelines-pro-bounds-constant-array-index.GslHeader
    value: '<gsl/gsl>'
...
```

### 使用命令

```bash
# -------- 前置：生成 compile_commands.json --------
# CMake 项目
cmake -B build -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# 或在 CMakeLists.txt 中永久启用
#   set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Bear 包装 Make 项目（适用于非 CMake 项目）
bear -- make -j$(nproc)

# -------- 单文件扫描 --------
clang-tidy -p build src/main.cpp

# -------- 全项目批量扫描 --------
run-clang-tidy -p build -j$(nproc)

# -------- 自动修复 --------
run-clang-tidy -p build -j$(nproc) -fix

# -------- 指定额外检查（覆盖 .clang-tidy） --------
clang-tidy -p build \
  -checks='-*,bugprone-*,cert-*,clang-analyzer-security.*' \
  src/main.cpp

# -------- CI 集成示例（失败即退出） --------
run-clang-tidy -p build -j$(nproc) 2>&1 | tee clang-tidy.log
if grep -q "warning:" clang-tidy.log; then
  echo "clang-tidy found issues, failing build"
  exit 1
fi
```

### 集成要点

- 依赖 `compile_commands.json`（CMake: `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`）
- 非 CMake 项目可用 `bear` 工具包装 make 来生成 `compile_commands.json`
- CI 中用 `run-clang-tidy` 批量扫描
- MSVC 项目可通过 `clang-cl` 模式或导出 compile_commands 来使用 clang-tidy

---

## 2. GCC / Clang / MSVC — 编译与链接加固

### 2.1 编译期警告

| 检查能力 | GCC / Clang | MSVC |
|---------|------------|------|
| 基础警告集 | `-Wall -Wextra` | `/W4` |
| 警告即错误 | `-Werror` | `/WX` |
| 格式化字符串 | `-Wformat=2 -Wformat-security` | 默认在 `/W4` 中覆盖 |
| 隐式类型转换 | `-Wconversion -Wsign-conversion` | `/W4` 部分覆盖 |
| 变量遮蔽 | `-Wshadow` | `/W4`（C4456-4459） |
| switch 贯穿 | `-Wimplicit-fallthrough` | `/W4`（C26819，需启用 Core Check） |
| 未初始化变量 | `-Wuninitialized` | `/W4`（C4700） + `/sdl`（C4703） |
| 安全弃用函数 | N/A | `/sdl`（强制报 `scanf` → `scanf_s` 等） |

### 2.2 运行时防护机制

| 防护机制 | GCC / Clang | MSVC | 对抗攻击 |
|---------|------------|------|---------|
| **Stack Canary** | `-fstack-protector-strong` | `/GS`（默认启用） | 栈缓冲区溢出 |
| **FORTIFY_SOURCE** | `-D_FORTIFY_SOURCE=2`（需 `-O1`+） | N/A（由 `/sdl` + Secure CRT 替代） | 堆/栈溢出 |
| **ASLR (PIE/DYNAMICBASE)** | `-fPIE -pie` | `/DYNAMICBASE`（默认启用） | 地址预测 |
| **High-Entropy ASLR** | 内核 + PIE 自动支持 | `/HIGHENTROPYVA`（64-bit 默认启用） | ASLR 暴力破解 |
| **RELRO** | `-Wl,-z,relro,-z,now` | N/A（PE 无 GOT 概念） | GOT overwrite |
| **NX / DEP** | `-Wl,-z,noexecstack` | `/NXCOMPAT`（默认启用） | Shellcode 注入 |
| **CFI / CFG** | `-fsanitize=cfi`（Clang）/ `-fcf-protection=full`（GCC, Intel CET） | `/guard:cf`（Control Flow Guard） | ROP/JOP、虚表劫持 |
| **Stack Clash Protection** | `-fstack-clash-protection` | 默认有 stack probe（`/Gs`） | Stack clash |
| **自动零初始化** | `-ftrivial-auto-var-init=zero` | `/sdl` 部分覆盖（指针成员初始化为 NULL） | 未初始化内存泄漏 |
| **Safe SEH** | N/A | `/SAFESEH`（32-bit） | SEH 覆盖攻击 |
| **SDL 检查** | N/A | `/sdl`（安全开发生命周期增强） | 综合安全加固 |

### 2.3 CMake 安全加固配置

**CMakeLists.txt 配置**

```cmake
# ============================================================
# 安全编译标志 — 在项目顶层 CMakeLists.txt 中添加
# ============================================================

# ---------- 警告即错误（CMake 3.24+，跨编译器统一） ----------
# 等价于 GCC/Clang 的 -Werror 和 MSVC 的 /WX，
# 但由 CMake 统一管理，无需在 if(MSVC)/else() 中分别设置
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# ---------- 通用警告 ----------
if(MSVC)
    add_compile_options(/W4 /sdl)
    # /WX 已由 CMAKE_COMPILE_WARNING_AS_ERROR 处理，无需重复添加
else()
    add_compile_options(
        -Wall -Wextra
        # -Werror 已由 CMAKE_COMPILE_WARNING_AS_ERROR 处理，无需重复添加
        -Wformat=2 -Wformat-security
        -Wconversion -Wsign-conversion
        -Wshadow
        -Wimplicit-fallthrough
    )
endif()

# ---------- 运行时防护 ----------
if(MSVC)
    add_compile_options(/GS /guard:cf /DYNAMICBASE)
    add_link_options(/GUARD:CF /DYNAMICBASE /HIGHENTROPYVA /NXCOMPAT /CETCOMPAT)
else()
    add_compile_options(
        -fstack-protector-strong
        -fstack-clash-protection
        -D_FORTIFY_SOURCE=2
        -ftrivial-auto-var-init=zero
        -fPIE
    )

    # Intel CET（GCC 8+ / Clang 14+，仅 x86/x86_64）
    include(CheckCCompilerFlag)
    check_c_compiler_flag(-fcf-protection=full HAS_CF_PROTECTION)
    if(HAS_CF_PROTECTION)
        add_compile_options(-fcf-protection=full)
    endif()

    add_link_options(
        -pie
        -Wl,-z,relro,-z,now
        -Wl,-z,noexecstack
    )
endif()
```

### 2.4 直接编译命令

**GCC / Clang**

```bash
# -------- 编译 --------
CFLAGS="-O2 \
  -Wall -Wextra -Werror \
  -Wformat=2 -Wformat-security \
  -Wconversion -Wsign-conversion -Wshadow -Wimplicit-fallthrough \
  -fstack-protector-strong \
  -fstack-clash-protection \
  -fcf-protection=full \
  -D_FORTIFY_SOURCE=2 \
  -ftrivial-auto-var-init=zero \
  -fPIE"

LDFLAGS="-pie \
  -Wl,-z,relro,-z,now \
  -Wl,-z,noexecstack"

# 示例：编译单个文件
gcc $CFLAGS -c -o main.o main.c
gcc $LDFLAGS -o my_app main.o

# 示例：一步编译链接
gcc $CFLAGS $LDFLAGS -o my_app main.c utils.c

# -------- 验证 binary 加固状态（需安装 checksec） --------
checksec --file=my_app
# 期望输出：
#   RELRO:    Full RELRO
#   Stack:    Canary found
#   NX:       NX enabled
#   PIE:      PIE enabled
#   FORTIFY:  Enabled

# -------- 查看编译器默认启用了哪些安全标志 --------
gcc -Q --help=optimizers | grep -i stack
gcc -Q --help=target | grep -i cf-protection
```

**MSVC（Developer Command Prompt / PowerShell）**

```powershell
# -------- 编译 --------
cl /O2 /W4 /WX /GS /sdl /guard:cf ^
   /D_CRT_SECURE_NO_WARNINGS=0 ^
   /c main.c utils.c

# -------- 链接 --------
link /OUT:my_app.exe main.obj utils.obj ^
     /GUARD:CF /DYNAMICBASE /HIGHENTROPYVA ^
     /NXCOMPAT /CETCOMPAT /SAFESEH

# -------- 一步编译链接 --------
cl /O2 /W4 /WX /GS /sdl /guard:cf ^
   main.c utils.c ^
   /Fe:my_app.exe ^
   /link /GUARD:CF /DYNAMICBASE /HIGHENTROPYVA /NXCOMPAT /CETCOMPAT

# -------- 验证 PE 安全标志 --------
# 使用 dumpbin 检查
dumpbin /headers my_app.exe | findstr /i "DLL characteristics"
# 期望看到: Dynamic base, NX compatible, Guard CF, High Entropy VA

# 使用 PowerShell 检查 ASLR/DEP
Get-ProcessMitigation -Name my_app.exe
```

> `/CETCOMPAT` 启用 Intel CET（Shadow Stack + IBT），等同于 GCC 的 `-fcf-protection=full`，需要 Windows 10 21H1+ 和支持 CET 的 CPU。
> `/SAFESEH` 仅对 32-bit 目标有效，64-bit 下自动忽略。

### 2.5 GCC/Clang vs MSVC 特性差异总结

| 特性 | GCC/Clang | MSVC | 说明 |
|------|-----------|------|------|
| RELRO / GOT 保护 | 有 | 无 | PE 格式无 GOT，MSVC 用 CFG 替代间接调用保护 |
| FORTIFY_SOURCE | 有 | 无 | MSVC 用 Secure CRT（`_s` 函数族）替代 |
| `/sdl` | 无 | 有 | MSVC 独有的综合安全加固开关 |
| SAFESEH | 无 | 有 | 仅 32-bit PE 相关 |
| CFI 实现 | Clang LTO-based CFI / GCC CET | CFG（Control Flow Guard） | 机制不同，目标一致 |

---

## 3. BinSkim — 二进制产物审计

BinSkim 对最终 binary（ELF / PE）做安全属性验证，确保编译层的加固**真正落地**。

### 关键规则

| Rule ID | 检查项 | ELF 对应标志 | PE 对应标志 |
|---------|--------|-------------|------------|
| `BA2001` | PIE / ASLR | `-fPIE -pie` | `/DYNAMICBASE` |
| `BA2002` | NX / DEP | `-Wl,-z,noexecstack` | `/NXCOMPAT` |
| `BA2005` | RELRO | `-Wl,-z,relro,-z,now` | N/A (PE 无此概念) |
| `BA2006` | Stack Canary | `-fstack-protector-strong` | `/GS` |
| `BA2007` | Safe SEH | N/A | `/SAFESEH` (32-bit) |
| `BA2008` | 无 RWX 段 | 链接器配置 | 链接器配置 |
| `BA2010` | High-Entropy ASLR | PIE + 内核支持 | `/HIGHENTROPYVA` |
| `BA2011` | CFG 启用 | N/A | `/guard:cf` |
| `BA2024` | FORTIFY_SOURCE | `-D_FORTIFY_SOURCE=2` | N/A |
| `BA2025` | CET / Shadow Stack | `-fcf-protection=full` | `/CETCOMPAT` |

### 安装

```bash
# -------- 方式 1：.NET 全局工具（跨平台，推荐） --------
dotnet tool install --global Microsoft.CodeAnalysis.BinSkim

# -------- 方式 2：从 GitHub Releases 下载预编译二进制 --------
# https://github.com/microsoft/binskim/releases
# 下载对应平台的 zip 包解压即可

# -------- 方式 3：NuGet 包（CI 集成） --------
nuget install Microsoft.CodeAnalysis.BinSkim
```

### 使用命令

```bash
# -------- 基本扫描 --------
# 扫描单个 binary
BinSkim analyze my_app --output results.sarif --verbose

# 扫描整个构建产出目录（递归）
BinSkim analyze "build/bin/*" --recurse --output results.sarif

# -------- 指定扫描平台 --------
# Linux ELF
BinSkim analyze "build/**/*.so" "build/**/my_app" --recurse --output results.sarif

# Windows PE
BinSkim analyze "build\Release\**\*.exe" "build\Release\**\*.dll" --recurse --output results.sarif

# -------- 指定配置文件（自定义启用/禁用规则） --------
BinSkim analyze my_app --config policy.xml --output results.sarif

# -------- 仅报告错误（忽略 warning/note） --------
BinSkim analyze my_app --level Error --output results.sarif

# -------- CI 集成：非零退出码表示失败 --------
BinSkim analyze "build/bin/*" --recurse --output results.sarif
if [ $? -ne 0 ]; then
  echo "BinSkim found security issues, blocking release"
  exit 1
fi
```

### 自定义策略配置（`policy.xml`）

```xml
<?xml version="1.0" encoding="utf-8"?>
<properties>
  <!-- 启用所有 BA 规则，设置为 Error 级别 -->
  <property name="BA2001.Enabled" value="true" />  <!-- PIE / ASLR -->
  <property name="BA2002.Enabled" value="true" />  <!-- NX / DEP -->
  <property name="BA2005.Enabled" value="true" />  <!-- RELRO (ELF) -->
  <property name="BA2006.Enabled" value="true" />  <!-- Stack Canary -->
  <property name="BA2007.Enabled" value="true" />  <!-- Safe SEH (PE 32-bit) -->
  <property name="BA2008.Enabled" value="true" />  <!-- 无 RWX 段 -->
  <property name="BA2010.Enabled" value="true" />  <!-- High-Entropy ASLR -->
  <property name="BA2011.Enabled" value="true" />  <!-- CFG (PE) -->
  <property name="BA2024.Enabled" value="true" />  <!-- FORTIFY_SOURCE (ELF) -->
  <property name="BA2025.Enabled" value="true" />  <!-- CET / Shadow Stack -->

  <!-- 按需禁用（例如：32-bit 项目不需要 HIGHENTROPYVA） -->
  <!-- <property name="BA2010.Enabled" value="false" /> -->
</properties>
```

### 输出结果解读

```bash
# SARIF 是 JSON 格式，可用 jq 快速提取失败项
cat results.sarif | jq '.runs[].results[] | select(.level == "error") | {ruleId, message: .message.text}'

# 示例输出：
# {
#   "ruleId": "BA2006",
#   "message": "my_app does not contain stack protector canaries."
# }
```

输出 SARIF 格式，可直接导入 GitHub Advanced Security、Azure DevOps、Jenkins 等 CI 系统。

---

## 4. Utils — 实用工具与技巧

### 4.1 `CMAKE_COMPILE_WARNING_AS_ERROR`（CMake 3.24+）

CMake 3.24 引入的变量，提供**跨编译器**的统一 warning-as-error 控制，无需在 `if(MSVC)/else()` 分支中分别写 `/WX` 和 `-Werror`。

**在 CMakeLists.txt 中启用**

```cmake
# 全局启用：所有 target 生效
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

# 或针对单个 target 启用/关闭
set_target_properties(my_lib PROPERTIES COMPILE_WARNING_AS_ERROR ON)
set_target_properties(third_party_lib PROPERTIES COMPILE_WARNING_AS_ERROR OFF)
```

**通过命令行控制（不修改 CMakeLists.txt）**

```bash
# 启用
cmake -B build -DCMAKE_COMPILE_WARNING_AS_ERROR=ON

# 禁用（本地开发时临时关闭，不影响 CI）
cmake -B build -DCMAKE_COMPILE_WARNING_AS_ERROR=OFF
```

**CMake 自动映射到各编译器的等价标志：**

| 编译器 | CMake 自动添加的标志 |
|--------|-------------------|
| GCC    | `-Werror` |
| Clang  | `-Werror` |
| MSVC   | `/WX` |
| Intel  | `-Werror` (Linux) / `/WX` (Windows) |

**与手动设置 `-Werror`/`/WX` 的区别：**

- `CMAKE_COMPILE_WARNING_AS_ERROR` 可以在命令行覆盖，开发者本地构建时可通过 `-DCMAKE_COMPILE_WARNING_AS_ERROR=OFF` 临时关闭，而硬编码的 `-Werror` 无法被外部覆盖
- 可以按 target 粒度控制（`COMPILE_WARNING_AS_ERROR` 属性），对第三方库关闭，对自己的代码开启
- 使用 `--compile-no-warning-as-error`（CMake 3.24+ 命令行选项）可全局覆盖

```bash
# 开发者本地构建：即使 CMakeLists.txt 中设置了 ON，也可以强制关闭
cmake --build build --compile-no-warning-as-error
```

### 4.2 `make -k`（Keep Going）

`make -k` 在遇到错误时**不立即停止**，而是继续编译其他不依赖于失败目标的文件。适用于安全扫描场景，可以一次性收集所有编译错误和警告，而非逐个修复。

**基本用法**

```bash
# 默认行为：遇到第一个错误即停止
make -j$(nproc)
# 输出：
#   src/foo.c:42: error: ...
#   make: *** [foo.o] Error 1
#   （停止，后续文件未编译）

# 使用 -k：尽可能多地编译，收集所有错误
make -k -j$(nproc)
# 输出：
#   src/foo.c:42: error: ...
#   src/bar.c:87: error: ...
#   src/baz.c:15: warning: ...
#   make: *** [foo.o] Error 1
#   make: *** [bar.o] Error 1
#   （所有可编译的文件都尝试了）
```

**安全加固场景下的典型用法**

```bash
# 场景 1：首次启用 -Werror 后，一次性收集所有警告
CFLAGS="-Wall -Wextra -Werror" make -k -j$(nproc) 2>&1 | tee build_errors.log

# 统计错误和警告数量
echo "=== 错误统计 ==="
grep -c "error:" build_errors.log
echo "=== 警告统计 ==="
grep -c "warning:" build_errors.log

# 按文件分组统计
grep "error:" build_errors.log | cut -d: -f1 | sort | uniq -c | sort -rn

# 场景 2：配合 clang-tidy 批量扫描
# run-clang-tidy 内部也支持类似的 keep-going 行为
run-clang-tidy -p build -j$(nproc) 2>&1 | tee clang-tidy.log

# 场景 3：CMake 构建使用 keep-going
cmake --build build -j$(nproc) -- -k         # Unix Makefiles
cmake --build build -j$(nproc) -- /maxcpucount /k  # MSBuild (MSVC)
```

**`make -k` vs `make` 的行为对比：**

| 行为 | `make` | `make -k` |
|------|--------|-----------|
| 遇到错误 | 立即停止 | 跳过依赖失败目标，继续编译其他 |
| 最终退出码 | 非零（首个错误） | 非零（有任何错误） |
| 适用场景 | 日常开发（快速反馈） | 大规模修复（一次性收集所有问题） |
| CI 推荐 | 是（快速失败，节省资源） | 仅在需要完整错误报告时使用 |

> **注意**：`make -k` 的最终退出码仍然是非零（如果有任何错误），不会影响 CI 的失败判定。它只是改变了"何时停止"，不改变"是否报错"。

---

## 三层闭环

```
源码 ──clang-tidy──→ 语义级缺陷（空指针、不安全API、CERT违规）
        │
        ▼
编译 ──gcc/clang/msvc flags──→ 注入运行时防护（Canary、ASLR、CFI/CFG、DEP）
        │
        ▼
产物 ──BinSkim──→ 验证所有防护在最终binary中生效
```

| 层次 | 工具 | 检查时机 | 失败策略 |
|------|------|---------|---------|
| 源码分析 | clang-tidy | PR gate / pre-commit | 阻断合入 |
| 编译加固 | gcc/clang/msvc 安全标志 | 构建系统 (CMake/MSBuild) | 构建失败 |
| 产物审计 | BinSkim | CI post-build | 阻断发布 |

**核心价值**：clang-tidy 和编译标志是"**加固**"——让 binary 安全；BinSkim 是"**审计**"——证明 binary 安全。尤其在大型项目中，子模块、第三方库、不同平台的构建配置可能覆盖安全标志，BinSkim 作为最后一道关卡能兜底发现遗漏。
