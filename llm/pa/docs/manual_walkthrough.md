# Manual Walkthrough: One Prefill And Two Decode Steps

This walkthrough explains how block allocation and KV cache addressing change over time for a tiny example.

## Setup

Assume:

- `block_size = 4`
- physical blocks available: `0, 1, 2, 3, 4, 5, ...`
- two active sequences:
  - `seq A = 100`
  - `seq B = 200`

Each layer owns its own KV cache tensor, but all layers share the same scheduler metadata:

- `past_lens`
- `subsequence_begins`
- `block_indices`
- `block_indices_begins`

The block table logic is the same for every layer.

## Step 0: Initial State

Sequence state:

```text
seq A: past_len = 0, logical_blocks = []
seq B: past_len = 0, logical_blocks = []
free blocks: [0, 1, 2, 3, 4, 5, ...]
```

No KV has been written yet.

---

## Step 1: Prefill

Input prompt lengths:

- `seq A`: 3 tokens
- `seq B`: 2 tokens

### 1.1 Reserve blocks

Because `block_size = 4`:

- `seq A` needs `ceil(3 / 4) = 1` block
- `seq B` needs `ceil(2 / 4) = 1` block

Allocate:

```text
seq A -> block 0
seq B -> block 1
free blocks: [2, 3, 4, 5, ...]
```

State becomes:

```text
seq A: past_len = 0, logical_blocks = [0]
seq B: past_len = 0, logical_blocks = [1]
```

### 1.2 Build metadata

Batch order is `[A, B]`, prompt lengths are `[3, 2]`.

So:

```text
past_lens = [0, 0]
subsequence_begins = [0, 3, 5]
```

Now collect logical blocks needed for each sequence:

- `seq A`: total tokens after prefill = `0 + 3 = 3`, needs 1 block -> `[0]`
- `seq B`: total tokens after prefill = `0 + 2 = 2`, needs 1 block -> `[1]`

So:

```text
block_indices = [0, 1]
block_indices_begins = [0, 1, 2]
```

### 1.3 Slot mapping for KV writes

For `seq A`:

- token 0 -> block 0, offset 0 -> slot `0 * 4 + 0 = 0`
- token 1 -> block 0, offset 1 -> slot `1`
- token 2 -> block 0, offset 2 -> slot `2`

For `seq B`:

- token 0 -> block 1, offset 0 -> slot `1 * 4 + 0 = 4`
- token 1 -> block 1, offset 1 -> slot `5`

Combined slot mapping:

```text
[0, 1, 2, 4, 5]
```

### 1.4 KV cache layout after prefill

Per layer, the cache looks conceptually like this:

```text
block 0: [A0, A1, A2, _]
block 1: [B0, B1, _, _]
```

Underscore means unused slot.

### 1.5 Commit sequence lengths

After prefill:

```text
seq A: past_len = 3, logical_blocks = [0]
seq B: past_len = 2, logical_blocks = [1]
```

---

## Step 2: Decode #1

Each active sequence appends one new token.

### 2.1 Reserve capacity

New total lengths if we append one token:

- `seq A`: `3 + 1 = 4` -> still fits in block 0
- `seq B`: `2 + 1 = 3` -> still fits in block 1

No new block is needed.

State remains:

```text
seq A: past_len = 3, logical_blocks = [0]
seq B: past_len = 2, logical_blocks = [1]
```

### 2.2 Build metadata

Current decode batch is one token per sequence:

```text
q_lens = [1, 1]
past_lens = [3, 2]
subsequence_begins = [0, 1, 2]
```

Total tokens after appending current decode token:

- `seq A`: `4` -> 1 block -> `[0]`
- `seq B`: `3` -> 1 block -> `[1]`

So:

```text
block_indices = [0, 1]
block_indices_begins = [0, 1, 2]
```

### 2.3 Slot mapping for KV writes

For `seq A`:

- new token position = 3
- block 0, offset 3 -> slot `3`

For `seq B`:

- new token position = 2
- block 1, offset 2 -> slot `6`

Combined slot mapping:

```text
[3, 6]
```

### 2.4 KV cache layout after decode #1

```text
block 0: [A0, A1, A2, A3]
block 1: [B0, B1, B2, _]
```

### 2.5 Read path during attention

For the new token of `seq A`, the executor reads context positions:

```text
[(block 0, off 0), (block 0, off 1), (block 0, off 2), (block 0, off 3)]
```

For the new token of `seq B`, the executor reads:

```text
[(block 1, off 0), (block 1, off 1), (block 1, off 2)]
```

### 2.6 Commit lengths

```text
seq A: past_len = 4, logical_blocks = [0]
seq B: past_len = 3, logical_blocks = [1]
```

---

## Step 3: Decode #2

Again append one token per active sequence.

### 3.1 Reserve capacity

- `seq A`: `4 + 1 = 5` -> now needs 2 blocks
- `seq B`: `3 + 1 = 4` -> still fits in 1 block

Allocate one new block for `seq A`:

```text
seq A -> add block 2
free blocks: [3, 4, 5, ...]
```

State before commit:

```text
seq A: past_len = 4, logical_blocks = [0, 2]
seq B: past_len = 3, logical_blocks = [1]
```

### 3.2 Build metadata

```text
q_lens = [1, 1]
past_lens = [4, 3]
subsequence_begins = [0, 1, 2]
```

Total tokens after append:

- `seq A`: 5 -> needs blocks `[0, 2]`
- `seq B`: 4 -> needs blocks `[1]`

So:

```text
block_indices = [0, 2, 1]
block_indices_begins = [0, 2, 3]
```

### 3.3 Slot mapping for KV writes

For `seq A`:

- new token position = 4
- logical block = 1
- physical block = 2
- offset = 0
- slot = `2 * 4 + 0 = 8`

For `seq B`:

- new token position = 3
- block 1, offset 3 -> slot `7`

Combined slot mapping:

```text
[8, 7]
```

### 3.4 KV cache layout after decode #2

```text
block 0: [A0, A1, A2, A3]
block 1: [B0, B1, B2, B3]
block 2: [A4, _, _, _]
```

### 3.5 Read path during attention

For the new token of `seq A`, the executor reads:

```text
[(0,0), (0,1), (0,2), (0,3), (2,0)]
```

For the new token of `seq B`, the executor reads:

```text
[(1,0), (1,1), (1,2), (1,3)]
```

### 3.6 Commit lengths

```text
seq A: past_len = 5, logical_blocks = [0, 2]
seq B: past_len = 4, logical_blocks = [1]
```

---

## Why This Example Matters

This small example already shows the key properties of paged attention:

1. Logical sequence positions are translated to physical blocks through a block table
2. New blocks are added only when a sequence crosses a page boundary
3. Each layer can share the same scheduler metadata while keeping its own KV cache

---

## Step 4: Beam Fork

Now fork a third sequence from `seq A`:

```text
fork_sequence(100, 300)
```

State becomes:

```text
seq A: past_len = 5, logical_blocks = [0, 2]
seq B: past_len = 4, logical_blocks = [1]
seq C: past_len = 5, logical_blocks = [0, 2]
```

Physical block ref-counts:

```text
block 0 -> 2 users
block 1 -> 1 user
block 2 -> 2 users
```

At this point `seq A` and `seq C` share exactly the same KV history.

---

## Step 5: Branch Decode With Copy-On-Write

Decode one new token for `seq A` and `seq C`.

Their shared tail block is block 2, and it is only partially filled:

```text
seq A / seq C past_len = 5
block_size = 4
tail position = 5 mod 4 = 1
```

That means appending the next token would write into offset 1 of block 2. Because block 2 is shared, the runtime must do copy-on-write first.

### 5.1 Scheduler copy-on-write

Allocate a fresh block 3 for `seq A` and copy block 2 into it:

```text
seq A: [0, 3]
seq C: [0, 2]
```

Ref-counts now:

```text
block 0 -> 2 users
block 1 -> 1 user
block 2 -> 1 user
block 3 -> 1 user
```

### 5.2 Append decode token

The new token for `seq A` goes to block 3, offset 1.
The new token for `seq C` goes to block 2, offset 1.

After commit:

```text
seq A: past_len = 6, logical_blocks = [0, 3]
seq B: past_len = 4, logical_blocks = [1]
seq C: past_len = 6, logical_blocks = [0, 2]
```

Now the two beams have diverged safely.

---

## Step 6: Beam Merge

Suppose beam slot `seq B` should now reuse the hypothesis from `seq C`.

Call:

```text
beam_merge(200, 300)
```

Meaning:

- release the old blocks owned by `seq B`
- make `seq B` point at the same block list and past length as `seq C`

Before merge, `seq B` used block 1 alone, so block 1 becomes free.

After merge:

```text
seq A: past_len = 6, logical_blocks = [0, 3]
seq B: past_len = 6, logical_blocks = [0, 2]
seq C: past_len = 6, logical_blocks = [0, 2]
```

Ref-counts:

```text
block 0 -> 3 users
block 2 -> 2 users
block 3 -> 1 user
block 1 -> free
```

This is the teaching-model version of reassigning a beam slot to a winning beam.

---

## Step 7: Sequence Finish And Block Reclaim

If `seq C` finishes first:

```text
finish_sequence(300)
```

Then:

```text
seq A: past_len = 6, logical_blocks = [0, 3]
seq B: past_len = 6, logical_blocks = [0, 2]
```

Ref-counts:

```text
block 0 -> 2 users
block 2 -> 1 user
block 3 -> 1 user
```

If `seq A` finishes next, block 3 becomes free immediately because no other sequence references it.

If `seq B` finally finishes, blocks 0 and 2 also return to the free pool.

This is the core reclaim rule:

```text
ref_count(block) drops to 0 -> block returns to free list
```

---

## Final Mental Model

The extended teaching runtime now covers four scheduler behaviors:

1. allocate blocks for new tokens
2. share blocks across beams by ref-counting
3. split shared tail blocks with copy-on-write before mutation
4. reclaim blocks when beams are merged away or finished

That is the smallest useful model that explains how paged KV cache can support realistic beam-search lifecycle management without losing the original page-attention mental model.
2. Prefill and decode use the same block table abstraction but different query lengths
3. Physical blocks do not need to be globally contiguous per sequence
4. New blocks are allocated only when capacity is exceeded
5. Every layer can reuse the same block metadata while owning its own KV cache tensor

That is the core mental model behind the teaching implementation.