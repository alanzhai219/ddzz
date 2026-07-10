#pragma once

#include <cstddef>
#include <memory>
#include <type_traits>
#include <unordered_map>

#include "cache_entry.h"

namespace lru {

/**
 * @brief Type-erased heterogeneous cache.
 *
 * Holds one independent LRU cache (CacheEntry) per distinct <Key, Value> type pair.
 * New type pairs are registered lazily on first access via getOrCreate().
 *
 * @attention NOT THREAD SAFE.
 */
class MultiCache {
public:
    using EntryBasePtr = std::shared_ptr<CacheEntryBase>;
    template <typename K, typename V>
    using EntryPtr = std::shared_ptr<CacheEntry<K, V>>;

    /**
     * @param capacity maximum records FOR EACH <Key, Value> type pair.
     *                 Zero means caching is disabled (builders always called).
     */
    explicit MultiCache(size_t capacity) : _capacity(capacity) {}

    /**
     * @brief Look up or create a value in the cache that corresponds to the <KeyType, ValueType> pair.
     * @param key     search key.
     * @param builder callable that creates a ValueType from a const KeyType&.
     */
    template <typename KeyType,
              typename BuilderType,
              typename ValueType = typename std::result_of<BuilderType&(const KeyType&)>::type>
    typename CacheEntry<KeyType, ValueType>::ResultType getOrCreate(const KeyType& key, BuilderType builder) {
        auto entry = getEntry<KeyType, ValueType>();
        return entry->getOrCreate(key, std::move(builder));
    }

private:
    template <typename T>
    static size_t getTypeId() {
        static size_t id = nextTypeId();
        return id;
    }

    template <typename KeyType, typename ValueType>
    EntryPtr<KeyType, ValueType> getEntry() {
        using EntryType = CacheEntry<KeyType, ValueType>;
        size_t id = getTypeId<EntryType>();
        auto it = _storage.find(id);
        if (it == _storage.end()) {
            auto result = _storage.insert({id, std::make_shared<EntryType>(_capacity)});
            it = result.first;
        }
        return std::static_pointer_cast<EntryType>(it->second);
    }

    static size_t nextTypeId() {
        static size_t counter = 0;
        return counter++;
    }

    size_t _capacity;
    std::unordered_map<size_t, EntryBasePtr> _storage;
};

}  // namespace lru
