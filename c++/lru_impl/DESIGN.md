# Type-Erased Heterogeneous LRU Cache — Design Document

## 1. Overview

A three-layer, header-only C++17 cache library that provides:

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
   │ list+map  │       │ list+map  │      │ list+map  │
   └───────────┘       └───────────┘      └───────────┘
```

## 3. Layer Responsibilities

### Layer 1: `LruCache<Key, Value>` — Storage

| Aspect | Detail |
|--------|--------|
| Data structures | `std::list<pair<Key,Value>>` (order) + `std::unordered_map<Key, list::iterator>` (lookup) |
| `put(key, val)` | Insert at front; if full, evict tail (LRU); if exists, update + promote to MRU |
| `get(key)` | O(1) lookup; on hit promote to MRU via `splice`; on miss return default `Value()` |
| `evict(n)` | Remove `n` entries from tail |
| Complexity | All operations O(1) amortized |

**Key requirement:** `Key` must provide `size_t hash() const` and `operator==`.

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
| Need O(1) LRU eviction | `LruCache` — classic list + map |
| Need "compute if absent" semantic | `CacheEntry::getOrCreate` wraps lookup + build + store |
| Dozens of different Key/Value types must share one cache object | `MultiCache` type-erases entries via base class pointer + runtime type ID |
| Adding new cached types should require zero changes to the cache framework | Template-based lazy registration in `MultiCache::getEntry<K,V>()` |

## 5. Key Design Decisions

1. **Header-only** — all three classes are templates (except `_typeIdCounter` which is `static inline`), enabling zero-linkage usage.
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
| Total space per type pair | O(capacity) | list nodes + map buckets |

## 7. File Layout

```
lru_impl/
├── include/
│   ├── lru_cache.h       # Layer 1: LRU data structure
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

auto [primitive, status] = cache.getOrCreate(
    MatmulKey{128, 256, 512},
    [](const MatmulKey& k) { return create_expensive_primitive(k); }
);

if (status == lru::CacheEntryBase::LookUpStatus::Hit) {
    // reused cached primitive
}
```
