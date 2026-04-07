# Standalone Paged Attention Teaching Project

## Goals

This directory contains a compact teaching-oriented implementation of a paged attention runtime that mirrors the high-level design discussed from the OpenVINO CPU plugin:

1. Scheduler / block manager layer
2. Common address translation layer
3. Execution layer

The project intentionally focuses on learning and explanation rather than peak performance.

## What Is Included

- `python/standalone_pa.py`
  - A PyTorch-based Python teaching implementation
  - Supports one prefill followed by multiple decode steps in the same runtime
  - Uses multiple layers to simulate multiple attention op nodes in an LLM
  - Supports simplified int8 KV cache compression
  - Supports block reclaim, sequence finish, beam fork, and beam merge

- `cpp/standalone_pa.cpp`
  - A standalone C++ version with the same structure as the Python version
  - Uses only the standard library
  - Also supports one prefill followed by multiple decode steps
  - Uses multiple layers and simplified int8 KV cache compression
  - Also supports block reclaim, sequence finish, beam fork, and beam merge

- `docs/manual_walkthrough.md`
  - A hand-worked example showing how blocks change during one prefill and two decode steps

- `docs/design_zh_cn.md`
  - A detailed Chinese explanation of the runtime flow, data structures, and design rationale

## Directory Layout

```text
pa/
├── README.md
├── cpp/
│   └── standalone_pa.cpp
├── docs/
│   ├── design_zh_cn.md
│   └── manual_walkthrough.md
└── python/
    └── standalone_pa.py
```

## Design Mapping

### 1. Scheduler / GenAI-like layer

Class:

- `KVBlockManager`

Responsibilities:

- Maintain free physical blocks
- Maintain physical block reference counts
- Maintain `sequence -> physical blocks` mapping
- Reserve blocks for prefill and decode
- Release blocks when a sequence finishes
- Rebind one beam slot to another beam during merge
- Trigger copy-on-write when multiple beams share a partially filled tail block
- Build runtime metadata:
  - `past_lens`
  - `subsequence_begins`
  - `block_indices`
  - `block_indices_begins`

### 2. Common address-translation layer

Class:

- `ExecutorPACommon`

Responsibilities:

- Translate logical token positions to physical KV cache slots
- Build slot mapping for KV writes
- Collect physical block positions for attention reads

### 3. Execution layer

Class:

- `PagedAttentionExecutor`

Responsibilities:

- Write K/V into paged KV cache
- Read historical K/V using block tables
- Execute prefill attention
- Execute decode attention
- Apply simplified int8 compression and dequantization
- Copy a physical block when the scheduler requests copy-on-write

### 4. Multi-layer runtime

Classes:

- `ToyLayer`
- `ToyLLMRuntime`

Responsibilities:

- Simulate multiple attention op nodes in an LLM
- Keep one independent KV cache per layer
- Run one prefill step followed by multiple decode steps in the same runtime instance
- Apply scheduler-driven block copy plans across all layers
- Support beam fork, beam merge, and sequence finish APIs

## Simplifications Compared to a Production Runtime

This project intentionally omits several production concerns:

- No threading
- No by-channel quantization
- No reorder scratch buffer optimization
- No fused kernels or hardware-specific acceleration
- No rope rotation / xattention / adaptive R-KV

These were omitted to keep the code small and readable while preserving the core mental model.

## Runtime Flow

### Prefill

1. Reserve enough blocks for the prompt tokens
2. Build metadata from the scheduler
3. For each layer:
   - project `Q/K/V`
   - write `K/V` into that layer's KV cache
   - read historical and current KV via block table
   - compute causal attention for the prompt tokens
4. Commit prompt length into sequence state

### Decode

1. Reserve one additional token slot per active sequence
2. Build metadata again
3. For each layer:
   - project `Q/K/V` for the single new token
   - append `K/V` into that layer's KV cache
   - read full context via block table
   - compute attention for the new token
4. Commit one new token into sequence state

### Beam Fork, Merge, And Finish

The extended teaching runtime now models common beam-search lifecycle operations:

1. `fork_sequence(parent, child)`
  - The child shares the parent's logical block list
  - Physical block ref-counts are incremented

2. Copy-on-write on decode
  - If two beams share a partially filled tail block and one beam appends a token,
    that beam first gets a private replacement block
  - The old tail block is copied into the new block across every layer
  - The append then writes into the private block, so the sibling beam stays unchanged

3. `beam_merge(dst, src)`
  - The destination beam releases its current blocks
  - It then points to the source beam's block list and past length
  - This mimics reassigning a beam slot to a winning hypothesis

4. `finish_sequence(seq_id)`
  - Releases the sequence's blocks
  - Any block whose ref-count drops to zero returns to the free-block pool

## Compression Model

The teaching implementation uses simplified per-token symmetric int8 compression:

For each token vector `x`:

```text
scale = max(abs(x)) / 127
q = round(x / scale)
dequant(x) = q * scale
```

This is much simpler than production implementations, but it captures the key idea that KV cache can be stored in compressed form and dequantized on read.

## How To Run

### Python

Requirements:

- Python 3
- PyTorch

Run:

```bash
python3 python/standalone_pa.py
```

Expected behavior:

- One prefill pass
- Two decode passes
- Beam fork and branch decode with tail-block copy-on-write
- Beam merge and sequence finish with block reclaim
- Printed output shapes, block allocation state, and block ref-count state

### C++

Compile:

```bash
g++ -std=c++17 -O2 cpp/standalone_pa.cpp -o standalone_pa
```

Run:

```bash
./standalone_pa
```

Expected behavior:

- One prefill pass
- Two decode passes
- Beam fork and branch decode with tail-block copy-on-write
- Beam merge and sequence finish with block reclaim
- Printed output shapes, block allocation state, and block ref-count state

## Suggested Reading Order

1. Read this README first
2. Read `docs/design_zh_cn.md` if you prefer a Chinese explanation
3. Read `docs/manual_walkthrough.md`
4. Read `python/standalone_pa.py`
5. Read `cpp/standalone_pa.cpp`

The Python version is easier to understand first. The C++ version is included to mirror the same architecture in a systems language.