# NCHW pooling: reference vs optimized JIT-style implementation

This directory contains standalone educational implementations of forward 2D pooling for `float` NCHW tensors:

- [ref_pooling.cpp](ref_pooling.cpp): scalar reference implementation.
- [jit_pooling.cpp](jit_pooling.cpp): AVX2-specialized, JIT-style kernel pipeline with a scalar fallback.
- [pooling_test.cpp](pooling_test.cpp): compares reference and optimized results for max and average pooling.

`jit_pooling.cpp` is **not linked to oneDNN/Xbyak**. It expresses the same execution strategy in portable C++ plus AVX2 intrinsics. In OpenVINO, the analogous machine instructions are emitted by Xbyak in `jit_uni_pool_kernel` at primitive initialization time.

## Optimizations used by the optimized implementation

1. **Channel SIMD** — converts one NCHW channel block into `[H][W][8]` scratchpad storage, then applies one AVX2 vector instruction to 8 channels.
2. **ISA dispatch** — checks AVX2 at runtime; otherwise calls the scalar reference kernel.
3. **Output-width unrolling** — computes up to four `ow` positions in one kernel call, with independent vector accumulators.
4. **Padding specialization** — dispatches each output-width tile to either a branch-free interior kernel or a boundary kernel.
5. **NCHW-to-blocked conversion** — converts only one `(n, channel-block)` slice rather than materializing a complete blocked tensor.
6. **Thread-private scratchpads** — every worker owns a separate blocked source/destination slice.
7. **Cache-friendly work partitioning** — atomic work-stealing distributes `(n, channel-block)` tasks, so each worker reuses the converted source slice for all output rows.
8. **Average-pooling reciprocal reuse** — the interior region broadcasts `1 / (KH * KW)` once per tile and uses vector multiplication instead of per-element division.
9. **Masked channel tail** — the final incomplete channel block is padded on input and written back using only the valid lanes.
10. **Post-op fusion** — `apply_post_ops()` executes `ReLU`, clamp, or scale/bias while the result is held in vector registers, immediately before storing the blocked output.

The illustrative optimized path supports `max`, `avg_include_padding`, and `avg_exclude_padding`, plus fused `ReLU`, clamp, and scale/bias post-ops. Max-pooling workspace indices and binary post-ops remain extension points.

## Compile example

```sh
g++ -std=c++17 -O3 -Wall -Wextra -pthread -c ref_pooling.cpp jit_pooling.cpp
g++ -std=c++17 -O3 -Wall -Wextra -pthread ref_pooling.cpp jit_pooling.cpp pooling_test.cpp -o pooling_test
./pooling_test
```

No global `-mavx2` is required on GCC/Clang: the AVX2 functions are marked with a per-function target attribute. Calling the optimized function on a CPU without AVX2 safely falls back to the reference path.
