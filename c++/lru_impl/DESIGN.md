# Type-Erased Heterogeneous LRU Cache — Design Document

## 1. Overview

A three-layer, header-only C++11 cache library that provides:

- **Per-type-pair LRU eviction** — each distinct `<Key, Value>` combination gets its own independent LRU cache.
- **Type-erased unified API** — callers interact with a single `MultiCache` object regardless of how many type pairs exist.
- **Lazy get-or-create semantics** — on a cache miss, a user-supplied builder functor creates the value, which is automatically stored.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      MultiCache                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  unordered_map<type_id, shared_ptr<CacheEntryBase>>│  │
│  │                                                   │  │
│  │  type_id=0 ──► CacheEntry<ConvKey, Primitive>     │  │
│  │  type_id=1 ──► CacheEntry<ReorderKey, Primitive>  │  │
│  │  type_id=2 ──► CacheEntry<MvnKey, JitKernel>     │  │
│  │  ...                                              │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
         │                    │                  │
         ▼                    ▼                  ▼
   ┌───────────┐       ┌───────────┐      ┌───────────┐
   │CacheEntry │       │CacheEntry │      │CacheEntry │
   │<K1, V1>   │       │<K2, V2>   │      │<K3, V3>   │
   │           │       │           │      │           │
   │getOrCreate│       │getOrCreate│      │getOrCreate│
   └─────┬─────┘       └─────┬─────┘      └─────┬─────┘
         │                    │                  │
         ▼                    ▼                  ▼
   ┌───────────┐       ┌───────────┐      ┌───────────┐
   │ LruCache  │       │ LruCache  │      │ LruCache  │
   │ <K1, V1>  │       │ <K2, V2>  │      │ <K3, V3>  │
   │           │       │           │      │           │
    │ node+map  │       │ node+map  │      │ node+map  │
   └───────────┘       └───────────┘      └───────────┘
```

## 3. Layer Responsibilities

### Layer 1: `LruCache<Key, Value>` — Storage

| Aspect | Detail |
|--------|--------|
| Data structures | intrusive doubly-linked nodes `LruNode<Key, Value>` (order) + `std::unordered_map<Key, LruNode<Key, Value>*>` (lookup) |
| `put(key, val)` | If key exists, update value and move its node to the front; otherwise allocate a new node, attach it to the front, and evict the tail if full |
| `get(key)` | O(1) hash lookup; on hit move the node to the front by detach + attach; on miss return default `Value()` |
| `touch(node)` | Internal recency-update helper: if the node is not already at the front, detach it from its current position and reattach it at the front so it becomes the MRU entry |
| `evict(n)` | Remove `n` entries from the tail and free their nodes |
| Complexity | All operations O(1) amortized |

**Key requirement:** `Key` must provide `size_t hash() const` and `operator==`.

**Ownership model:** `LruCache` owns all `LruNode` instances, deletes them on eviction/destruction, disables copying, and supports move semantics.

**`_cacheMapper` structure:**

- Concrete type: `std::unordered_map<Key, LruNode<Key, Value>*, key_hasher>`.
- Role: hash-based index from cache key to the corresponding linked-list node.
- Why it exists: without `_cacheMapper`, `get(key)` and existing-key `put(key, val)` would need a linear scan over the linked list to find the target node.
- What it enables: average O(1) lookup of the node pointer, followed by O(1) recency maintenance via `touch(node)`.
- Division of responsibility: `_cacheMapper` answers “where is the node for this key?”, while the doubly-linked list answers “which node is MRU/LRU?”.
- Stored value choice: the map stores `LruNode*` rather than `Value` so the cache can both read the payload and move the exact node to the head without another search.
- Lifetime rule: when a node is inserted, the map gets a `key -> node*` entry; when a node is evicted, its map entry is erased before the node is deleted.

**Recency terms:**

- **MRU (Most Recently Used)** — the head node; the entry most recently accessed or updated.
- **LRU (Least Recently Used)** — the tail node; the entry that has gone the longest without being accessed or updated, and is evicted first when capacity is full.

### Layer 2: `CacheEntry<Key, Value>` — Get-or-Create Policy

Inherits from `CacheEntryBase` (virtual dtor for type erasure).

- **`getOrCreate(key, builder)`** — look up in the underlying `LruCache`; on miss, call `builder(key)` to produce the value, store it, and return `{value, Miss}`. On hit, return `{value, Hit}`.
- **Zero-capacity fast path** — when capacity is 0, always calls the builder without touching the cache.
- **Empty-value convention** — default-constructed `Value()` is treated as "not present" and is never stored.

### Layer 3: `MultiCache` — Type-Erased Dispatch

- Stores `CacheEntry` instances as `shared_ptr<CacheEntryBase>` in a flat `unordered_map<size_t, ...>`.
- **Type ID generation:** uses a `static` function-local variable initialized from an `atomic_size_t` counter, giving each `CacheEntry<K,V>` instantiation a unique runtime ID.
- **Lazy registration:** the first `getOrCreate<K,V>(...)` call creates the `CacheEntry<K,V>` with the configured capacity.
- **Single capacity** applies uniformly to every type pair.

## 4. Why Three Layers?

| Problem | Solution |
|---------|----------|
| Need O(1) LRU eviction | `LruCache` — custom doubly-linked nodes + hash map |
| Need "compute if absent" semantic | `CacheEntry::getOrCreate` wraps lookup + build + store |
| Dozens of different Key/Value types must share one cache object | `MultiCache` type-erases entries via base class pointer + runtime type ID |
| Adding new cached types should require zero changes to the cache framework | Template-based lazy registration in `MultiCache::getEntry<K,V>()` |

## 5. Key Design Decisions

1. **Header-only** — all three classes are templates, enabling zero-linkage usage.
2. **NOT thread-safe** — intentional; thread safety is the caller's responsibility (e.g., each inference thread owns its own `MultiCache`).
3. **Default-value-as-empty convention** — simplifies the API (no `std::optional`) but means `Value()` must represent "no result."
4. **Capacity is per-type-pair** — capacity N means each `<K,V>` pair caches up to N entries independently.

## 6. Complexity Summary

| Operation | Time | Space |
|-----------|------|-------|
| `LruCache::get` | O(1) amortized | — |
| `LruCache::put` | O(1) amortized | — |
| `LruCache::evict(n)` | O(n) | — |
| `CacheEntry::getOrCreate` | O(1) + builder cost on miss | — |
| `MultiCache::getOrCreate` | O(1) type dispatch + above | — |
| Total space per type pair | O(capacity) | linked nodes + map buckets |

## 7. File Layout

```
lru_impl/
├── include/
│   ├── lru_cache.h       # Layer 1: LRU data structure
│   ├── lru_node.h        # Shared doubly-linked node for LRU storage
│   ├── cache_entry.h     # Layer 2: get-or-create wrapper
│   └── multi_cache.h     # Layer 3: type-erased dispatch
├── test/
│   └── test_cache.cpp    # Unit tests
├── CMakeLists.txt
└── DESIGN.md             # This document
```

## 8. Usage Example

```cpp
#include "multi_cache.h"

// Define a key type
struct MatmulKey {
    int M, N, K;
    size_t hash() const { return std::hash<int>()(M) ^ (std::hash<int>()(N) << 16) ^ std::hash<int>()(K); }
    bool operator==(const MatmulKey& o) const { return M == o.M && N == o.N && K == o.K; }
};

// Use
lru::MultiCache cache(1000);

std::pair<std::shared_ptr<Primitive>, lru::CacheEntryBase::LookUpStatus> result =
    cache.getOrCreate(
        MatmulKey{128, 256, 512},
        [](const MatmulKey& k) { return create_expensive_primitive(k); }
    );

std::shared_ptr<Primitive> primitive = result.first;
lru::CacheEntryBase::LookUpStatus status = result.second;

if (status == lru::CacheEntryBase::LookUpStatus::Hit) {
    // reused cached primitive
}
```
