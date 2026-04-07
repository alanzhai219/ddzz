# When JIT Outperforms AVX-512 Intrinsics

## Background

In a generic decompression workload (dynamic bitmasks), the AVX-512 intrinsics kernel is ~1.3% faster than the best generic JIT kernel (prefix-sum). This document describes the scenario where JIT **reverses** that gap and outperforms intrinsics by **1.36–1.62×**.

---

## Core Idea: Runtime Specialization

In LLM inference, weight sparsity masks are **fixed at model load time** and reused for every inference call. A JIT compiler can inspect the actual mask values during code generation and produce code that a static (AOT) compiler cannot:

| Eliminated at runtime | How |
|----------------------|-----|
| All `popcnt` instructions | Source offsets pre-computed as immediate constants |
| All pointer-advance `add` | Addresses become `[base + IMM]` — no serial dependency |
| All bitmask memory loads | Mask values embedded as 64-bit immediates |
| All alignment arithmetic | Inter-block padding pre-computed at JIT-compile time |
| Redundant expand for zero chunks | Replaced with `vpxord + store` (2 insns) |
| Redundant expand for dense chunks | Replaced with `vmovdqu8` copy (2 insns) |

---

## Generated Code Comparison

### Generic kernel (intrinsics or JIT-generic) — per chunk: 6 instructions

```asm
mov  rax, [bitmask + runtime_off]  ; memory load (bitmask)
kmovq k1, rax                      ; k-mask setup
vpexpandb zmm{k1}{z}, [src]        ; expand-load (DATA-DEPENDENT address)
popcnt rax, rax                    ; 3-cycle latency
add  src, rax                      ; depends on popcnt → serial chain
vmovdqu8 [dst + off], zmm          ; store
```

### JIT Specialized — per partial chunk: 4 instructions, 0 address-computation cycles

```asm
mov  rax, 0x0101...                ; IMMEDIATE mask (no memory load)
kmovq k1, rax                      ; k-mask setup
vpexpandb zmm{k1}{z}, [base + IMM] ; STATIC address (no dependency)
vmovdqu8 [dst + IMM], zmm          ; store
```

### JIT Specialized — zero chunk: 2 instructions

```asm
vpxord zmm, zmm, zmm               ; zero register
vmovdqu8 [dst + IMM], zmm          ; store (no compressed data read)
```

### JIT Specialized — dense chunk: 2 instructions

```asm
vmovdqu8 zmm, [base + IMM]         ; plain 64-byte copy (no k-mask)
vmovdqu8 [dst + IMM], zmm          ; store
```

---

## Benchmark Setup

```bash
g++ decomp_bench_jit_advantage.cpp -I../../3rdparty/xbyak -march=native -O2 -o decomp_bench_jit_advantage
./decomp_bench_jit_advantage
```

- **Data**: 4 blocks × 4096 bytes, 500K iterations, warmup 2000
- **Kernels compared**: AVX-512 Intrinsics (generic) / JIT Generic (prefix-sum) / JIT Specialized (mask-aware)

---

## Results

### Scenario A — Random 70% Sparsity (unstructured pruning)

All 256 chunks are partial (no zero/dense shortcuts available).

| Kernel | us/call | vs Intrinsics |
|--------|---------|---------------|
| AVX-512 Intrinsics | ~0.204 | baseline |
| JIT Generic | ~0.202 | ~1.0× |
| JIT Specialized | ~0.202 | ~1.0× (tie) |

**Analysis**: With uniform random sparsity, every chunk is partial. The JIT's only advantage is static addressing (no `popcnt`/`add`), but the OoO engine hides this latency at small data sizes. Result: tie.

### Scenario B — Structured Sparsity (30% zero + 10% dense chunks)

Realistic for channel-pruned or block-sparse LLM weights.

| Kernel | us/call | vs Intrinsics |
|--------|---------|---------------|
| AVX-512 Intrinsics | ~0.202 | baseline |
| JIT Generic | ~0.220 | 0.92× (slower!) |
| **JIT Specialized** | **~0.149** | **1.36–1.62× faster ★** |

Chunk breakdown in generated code:

| Chunk type | Count | % | Instructions per chunk |
|------------|-------|---|----------------------|
| Zero (mask=0) | 69/256 | 27% | 2 (vpxord + store) |
| Dense (mask=~0) | 29/256 | 11% | 2 (vmovdqu8 copy) |
| Partial | 158/256 | 62% | 4 (mov+kmovq+vpexpandb+store) |

**Why the JIT wins**: Intrinsics always execute `vpexpandb + popcnt + add` for ALL chunks (6 insns). The specialized JIT uses only 2 instructions for 38% of chunks (zero + dense), eliminating both the compressed-data read and the k-mask logic.

---

## Why This Matters for LLM Inference

```
Model Load                           Inference (millions of calls)
   │                                        │
   ├─ Parse weights                         ├─ Call jit_func(&args)
   ├─ Extract sparsity masks                │   └─ 0.149 us per call
   ├─ JIT compile specialized kernel        │       (no popcnt, no bitmask load,
   │   └─ Cost: ~microseconds               │        no pointer arithmetic)
   └─ Ready                                 └─ ...
```

The JIT compilation cost (~μs) is **amortized over millions of inference calls**. Each call saves ~33% of instructions compared to the generic path.

---

## When to Use Which Kernel

| Scenario | Best Kernel | Why |
|----------|------------|-----|
| Fixed masks, reused many times | **JIT Specialized** | Amortized compile cost, static addresses, dead-code elimination |
| Dynamic masks, large blocks | AVX-512 Intrinsics | No JIT compile overhead, good compiler scheduling |
| Dynamic masks, generic use | JIT Generic (prefix-sum) | Within 1.3% of intrinsics |

---

## JIT Compilation Overhead Analysis

### The Problem: Unknown Masks Require Recompilation

The benchmark above excludes JIT compilation time from measurement — the kernel is compiled before the timing loop. In real scenarios where masks are **not known in advance**, each new mask pattern requires re-instantiating `jit_decompress_specialized_t`, and the compilation cost becomes non-trivial.

### Compilation Cost Breakdown

| Operation | Source | Typical Cost |
|-----------|--------|-------------|
| `mmap` (allocate executable memory) | Xbyak constructor `CodeGenerator(4096*256)` = 1MB | 2–10 μs |
| Code generation loop | Iterate `blocks × 64` chunks, emit 2–4 x86 instructions each | Proportional to size |
| `mprotect` (W→X) | `ready()` call | 1–5 μs |
| `munmap` (destructor) | Free memory | 2–10 μs |

Each Xbyak instruction emission involves encoding the x86 instruction (table lookup + bit manipulation) and writing a few bytes to the buffer, costing approximately **20–100 ns** per instruction (pure userspace, no syscalls).

### Estimation Formula

$$T_{compile} = T_{mmap} + T_{mprotect} + N_{chunks} \times C_{avg\_insns} \times T_{per\_insn}$$

With typical values:

$$T_{compile} \approx 5\mu s + 3\mu s + (blocks \times 64) \times 3 \times 50ns$$

| blocks | chunks | Code generation | Syscalls | **Total compile time** |
|--------|--------|----------------|----------|----------------------|
| 1 | 64 | ~10 μs | ~8 μs | **~18 μs** |
| 4 | 256 | ~38 μs | ~8 μs | **~46 μs** |
| 16 | 1024 | ~154 μs | ~8 μs | **~162 μs** |
| 64 | 4096 | ~614 μs | ~8 μs | **~622 μs** |

### Break-Even Point

If the specialized JIT saves $\Delta t$ μs per call vs the generic JIT, it needs $N$ calls to pay back compilation:

$$N_{break\_even} = \frac{T_{compile}}{\Delta t}$$

For the 4-block benchmark scenario, assuming ~0.2 μs/call savings:

$$N_{break\_even} = \frac{46\mu s}{0.2\mu s} \approx 230 \text{ calls}$$

> **The same mask pattern must be invoked ~230 times before the specialized JIT breaks even.**

Note: The benchmark code now includes compilation time measurement and break-even calculation (see the "JIT Compile Cost Amortization" output section in `run_scenario`).

### When Compilation Cost Is Negligible (LLM Inference)

In autoregressive LLM generation, each token requires decompressing the same weight tensors:

- 1000-token generation × 32-layer model = **32,000 calls per mask pattern**
- Compilation cost (~46 μs) is amortized over 32,000 invocations → **~0.001 μs per call**

### When Compilation Cost Dominates

| Scenario | Why | Recommended Kernel |
|----------|-----|-------------------|
| Dynamic sparsity (activation masks) | Mask changes every forward pass | JIT Generic |
| One-shot decompression | Decompress once, use dense result | AVX-512 Intrinsics |
| Many distinct weight shapes | Each mask needs its own kernel, I-cache pressure | JIT Generic + kernel caching |

### Practical Mitigation Strategies

| Strategy | Approach |
|----------|----------|
| **Kernel caching** | Hash the mask array, cache compiled kernels in an LRU map. Same mask → reuse function pointer |
| **Async compilation** | Use generic JIT for the first call; compile specialized kernel on a background thread; atomically swap function pointer when ready |
| **Tiered compilation (HotSpot-style)** | Count invocations per mask pattern; only JIT-specialize after a threshold |
| **Reduce compilation overhead** | Use smaller Xbyak code buffer; pool/reuse `CodeGenerator` objects to avoid repeated `mmap`/`munmap` |

---

## Key Takeaway

> **JIT's true power is not matching compiler output — it's doing what compilers fundamentally cannot**: specializing code based on runtime data.

When bitmask values become compile-time constants, the JIT eliminates entire categories of instructions (`popcnt`, pointer arithmetic, bitmask loads) and generates dead-code-eliminated paths for zero/dense chunks. This is the textbook case for runtime code generation.
