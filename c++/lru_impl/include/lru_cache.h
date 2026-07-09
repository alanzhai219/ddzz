#pragma once

#include <cstddef>
#include <list>
#include <unordered_map>
#include <utility>

namespace lru {

/**
 * @brief LRU (Least Recently Used) cache implementation.
 *
 * Uses a doubly-linked list to maintain access order and a hash map for O(1) lookup.
 * The most recently accessed item is at the front of the list; eviction removes from the back.
 *
 * @tparam Key must define `size_t hash() const` and `operator==`.
 * @tparam Value must be default-constructible and equality-comparable.
 *
 * @attention NOT THREAD SAFE.
 */
template <typename Key, typename Value>
class LruCache {
public:
    using value_type = std::pair<Key, Value>;

    explicit LruCache(size_t capacity) : _capacity(capacity) {}

    /**
     * @brief Insert or update a key-value pair.
     *
     * If the key exists, its value is updated and the entry is promoted to MRU.
     * If the key is new and the cache is full, the LRU entry is evicted first.
     * No-op when capacity is 0.
     */
    void put(const Key& key, const Value& val) {
        if (0 == _capacity) {
            return;
        }
        auto mapItr = _cacheMapper.find(key);
        if (mapItr != _cacheMapper.end()) {
            touch(mapItr->second);
            mapItr->second->second = val;
        } else {
            if (_cacheMapper.size() == _capacity) {
                evict(1);
            }
            auto itr = _lruList.insert(_lruList.begin(), {key, val});
            _cacheMapper.insert({key, itr});
        }
    }

    /**
     * @brief Look up a value by key.
     * @return The value if found (entry promoted to MRU), or a default-constructed Value on miss.
     */
    Value get(const Key& key) {
        auto itr = _cacheMapper.find(key);
        if (itr == _cacheMapper.end()) {
            return Value();
        }
        touch(itr->second);
        return _lruList.front().second;
    }

    /**
     * @brief Evict up to @p n least recently used entries.
     */
    void evict(size_t n) {
        for (size_t i = 0; i < n && !_lruList.empty(); ++i) {
            _cacheMapper.erase(_lruList.back().first);
            _lruList.pop_back();
        }
    }

    [[nodiscard]] size_t getCapacity() const noexcept { return _capacity; }
    [[nodiscard]] size_t size() const noexcept { return _cacheMapper.size(); }

private:
    struct key_hasher {
        std::size_t operator()(const Key& k) const { return k.hash(); }
    };

    using lru_list_type = std::list<value_type>;
    using cache_map_value_type = typename lru_list_type::iterator;

    void touch(typename lru_list_type::iterator itr) {
        _lruList.splice(_lruList.begin(), _lruList, itr);
    }

    lru_list_type _lruList;
    std::unordered_map<Key, cache_map_value_type, key_hasher> _cacheMapper;
    size_t _capacity;
};

}  // namespace lru
