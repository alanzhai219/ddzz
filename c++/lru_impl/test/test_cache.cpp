#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>

#include "multi_cache.h"

// ============================================================================
// Test helpers
// ============================================================================

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                    \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "FAIL: " << (msg) << " (" << __FILE__ << ":" << __LINE__ \
                      << ")" << std::endl;                                        \
            ++g_tests_failed;                                                     \
        } else {                                                                  \
            ++g_tests_passed;                                                     \
        }                                                                         \
    } while (0)

// A simple key type satisfying the hash()/operator== contract.
struct IntKey {
    int value;
    size_t hash() const { return std::hash<int>()(value); }
    bool operator==(const IntKey& o) const { return value == o.value; }
};

// A second key type for heterogeneous MultiCache testing.
struct StringKey {
    std::string value;
    size_t hash() const { return std::hash<std::string>()(value); }
    bool operator==(const StringKey& o) const { return value == o.value; }
};

// ============================================================================
// LruCache tests
// ============================================================================

void test_lru_basic_put_get() {
    lru::LruCache<IntKey, int> cache(3);
    cache.put(IntKey{1}, 10);
    cache.put(IntKey{2}, 20);

    TEST_ASSERT(cache.get(IntKey{1}) == 10, "get existing key 1");
    TEST_ASSERT(cache.get(IntKey{2}) == 20, "get existing key 2");
    TEST_ASSERT(cache.get(IntKey{99}) == 0, "get non-existing key returns default");
}

void test_lru_eviction() {
    lru::LruCache<IntKey, int> cache(2);
    cache.put(IntKey{1}, 10);
    cache.put(IntKey{2}, 20);
    // Cache is full: {2(MRU), 1(LRU)}
    cache.put(IntKey{3}, 30);
    // Key 1 should be evicted
    TEST_ASSERT(cache.get(IntKey{1}) == 0, "evicted key 1 returns default");
    TEST_ASSERT(cache.get(IntKey{2}) == 20, "key 2 survives eviction");
    TEST_ASSERT(cache.get(IntKey{3}) == 30, "key 3 was inserted");
}

void test_lru_access_promotes() {
    lru::LruCache<IntKey, int> cache(2);
    cache.put(IntKey{1}, 10);
    cache.put(IntKey{2}, 20);
    // Access key 1 to promote it to MRU
    cache.get(IntKey{1});
    // Now order: {1(MRU), 2(LRU)}
    cache.put(IntKey{3}, 30);
    // Key 2 (LRU) should be evicted, not key 1
    TEST_ASSERT(cache.get(IntKey{2}) == 0, "key 2 evicted after key 1 promoted");
    TEST_ASSERT(cache.get(IntKey{1}) == 10, "key 1 survives due to promotion");
    TEST_ASSERT(cache.get(IntKey{3}) == 30, "key 3 present");
}

void test_lru_update_existing() {
    lru::LruCache<IntKey, int> cache(2);
    cache.put(IntKey{1}, 10);
    cache.put(IntKey{1}, 100);
    TEST_ASSERT(cache.get(IntKey{1}) == 100, "update overwrites value");
    TEST_ASSERT(cache.size() == 1, "update does not increase size");
}

void test_lru_zero_capacity() {
    lru::LruCache<IntKey, int> cache(0);
    cache.put(IntKey{1}, 10);
    TEST_ASSERT(cache.get(IntKey{1}) == 0, "zero capacity stores nothing");
    TEST_ASSERT(cache.size() == 0, "zero capacity size is 0");
}

void test_lru_evict_multiple() {
    lru::LruCache<IntKey, int> cache(5);
    for (int i = 0; i < 5; ++i) {
        cache.put(IntKey{i}, i * 10);
    }
    TEST_ASSERT(cache.size() == 5, "size is 5 before evict");
    cache.evict(3);
    TEST_ASSERT(cache.size() == 2, "size is 2 after evicting 3");
    // The 3 oldest (LRU) are keys 0, 1, 2
    TEST_ASSERT(cache.get(IntKey{0}) == 0, "key 0 evicted");
    TEST_ASSERT(cache.get(IntKey{1}) == 0, "key 1 evicted");
    TEST_ASSERT(cache.get(IntKey{2}) == 0, "key 2 evicted");
    TEST_ASSERT(cache.get(IntKey{3}) == 30, "key 3 survives");
    TEST_ASSERT(cache.get(IntKey{4}) == 40, "key 4 survives");
}

// ============================================================================
// CacheEntry tests
// ============================================================================

void test_cache_entry_get_or_create() {
    lru::CacheEntry<IntKey, std::shared_ptr<int>> entry(3);
    int build_count = 0;

    auto builder = [&](const IntKey& k) -> std::shared_ptr<int> {
        ++build_count;
        return std::make_shared<int>(k.value * 10);
    };

    auto [val1, s1] = entry.getOrCreate(IntKey{1}, builder);
    TEST_ASSERT(s1 == lru::CacheEntryBase::LookUpStatus::Miss, "first access is miss");
    TEST_ASSERT(*val1 == 10, "builder produced correct value");
    TEST_ASSERT(build_count == 1, "builder called once");

    auto [val2, s2] = entry.getOrCreate(IntKey{1}, builder);
    TEST_ASSERT(s2 == lru::CacheEntryBase::LookUpStatus::Hit, "second access is hit");
    TEST_ASSERT(*val2 == 10, "cached value returned");
    TEST_ASSERT(build_count == 1, "builder not called on hit");
}

void test_cache_entry_zero_capacity() {
    lru::CacheEntry<IntKey, std::shared_ptr<int>> entry(0);
    int build_count = 0;

    auto builder = [&](const IntKey& k) -> std::shared_ptr<int> {
        ++build_count;
        return std::make_shared<int>(k.value);
    };

    auto [v1, s1] = entry.getOrCreate(IntKey{1}, builder);
    auto [v2, s2] = entry.getOrCreate(IntKey{1}, builder);
    TEST_ASSERT(s1 == lru::CacheEntryBase::LookUpStatus::Miss, "zero cap: always miss 1");
    TEST_ASSERT(s2 == lru::CacheEntryBase::LookUpStatus::Miss, "zero cap: always miss 2");
    TEST_ASSERT(build_count == 2, "zero cap: builder called every time");
}

// ============================================================================
// MultiCache tests
// ============================================================================

void test_multi_cache_single_type() {
    lru::MultiCache cache(10);
    int build_count = 0;

    auto builder = [&](const IntKey& k) -> std::shared_ptr<int> {
        ++build_count;
        return std::make_shared<int>(k.value * 100);
    };

    auto [v1, s1] = cache.getOrCreate(IntKey{5}, builder);
    TEST_ASSERT(s1 == lru::CacheEntryBase::LookUpStatus::Miss, "multi: first is miss");
    TEST_ASSERT(*v1 == 500, "multi: correct value");

    auto [v2, s2] = cache.getOrCreate(IntKey{5}, builder);
    TEST_ASSERT(s2 == lru::CacheEntryBase::LookUpStatus::Hit, "multi: second is hit");
    TEST_ASSERT(build_count == 1, "multi: builder called once");
}

void test_multi_cache_heterogeneous() {
    lru::MultiCache cache(10);

    // Type pair 1: IntKey -> shared_ptr<int>
    auto [v1, s1] = cache.getOrCreate(IntKey{1}, [](const IntKey& k) {
        return std::make_shared<int>(k.value);
    });
    TEST_ASSERT(*v1 == 1, "hetero: int value correct");

    // Type pair 2: StringKey -> shared_ptr<std::string>
    auto [v2, s2] = cache.getOrCreate(StringKey{"hello"}, [](const StringKey& k) {
        return std::make_shared<std::string>(k.value + " world");
    });
    TEST_ASSERT(*v2 == "hello world", "hetero: string value correct");

    // Verify both type pairs are independent (int cache still has its entry)
    auto [v3, s3] = cache.getOrCreate(IntKey{1}, [](const IntKey&) {
        return std::make_shared<int>(999);  // should NOT be called
    });
    TEST_ASSERT(s3 == lru::CacheEntryBase::LookUpStatus::Hit, "hetero: int pair still cached");
    TEST_ASSERT(*v3 == 1, "hetero: int pair returns original value");

    auto [v4, s4] = cache.getOrCreate(StringKey{"hello"}, [](const StringKey&) {
        return std::make_shared<std::string>("should not happen");
    });
    TEST_ASSERT(s4 == lru::CacheEntryBase::LookUpStatus::Hit, "hetero: string pair still cached");
    TEST_ASSERT(*v4 == "hello world", "hetero: string pair returns original value");
}

void test_multi_cache_eviction_per_type() {
    lru::MultiCache cache(2);  // capacity 2 per type pair

    // Fill IntKey cache to capacity
    cache.getOrCreate(IntKey{1}, [](const IntKey& k) { return std::make_shared<int>(k.value); });
    cache.getOrCreate(IntKey{2}, [](const IntKey& k) { return std::make_shared<int>(k.value); });
    // This should evict IntKey{1}
    cache.getOrCreate(IntKey{3}, [](const IntKey& k) { return std::make_shared<int>(k.value); });

    int build_count = 0;
    auto [v1, s1] = cache.getOrCreate(IntKey{1}, [&](const IntKey& k) {
        ++build_count;
        return std::make_shared<int>(k.value);
    });
    TEST_ASSERT(s1 == lru::CacheEntryBase::LookUpStatus::Miss, "eviction: key 1 was evicted");
    TEST_ASSERT(build_count == 1, "eviction: builder re-called for evicted key");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    // LruCache tests
    test_lru_basic_put_get();
    test_lru_eviction();
    test_lru_access_promotes();
    test_lru_update_existing();
    test_lru_zero_capacity();
    test_lru_evict_multiple();

    // CacheEntry tests
    test_cache_entry_get_or_create();
    test_cache_entry_zero_capacity();

    // MultiCache tests
    test_multi_cache_single_type();
    test_multi_cache_heterogeneous();
    test_multi_cache_eviction_per_type();

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Passed: " << g_tests_passed << std::endl;
    std::cout << "Failed: " << g_tests_failed << std::endl;

    return g_tests_failed == 0 ? 0 : 1;
}
