#pragma once

#include <cstddef>
#include <new>
#include <unordered_map>
#include <utility>

#include "lru_node.h"

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
    ~LruCache() { clear(); }

    LruCache(const LruCache&) = delete;
    LruCache& operator=(const LruCache&) = delete;

    LruCache(LruCache&& other) noexcept
        : _cacheMapper(std::move(other._cacheMapper)),
          _capacity(other._capacity),
          _head(other._head),
          _tail(other._tail) {
        other._head = nullptr;
        other._tail = nullptr;
        other._capacity = 0;
    }

    LruCache& operator=(LruCache&& other) noexcept {
        if (this != &other) {
            clear();
            _cacheMapper = std::move(other._cacheMapper);
            _capacity = other._capacity;
            _head = other._head;
            _tail = other._tail;

            other._head = nullptr;
            other._tail = nullptr;
            other._capacity = 0;
        }
        return *this;
    }

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
        auto map_it = _cacheMapper.find(key);
        if (map_it != _cacheMapper.end()) {
            // hit
            touch(map_it->second);
            map_it->second->value.second = val;
        } else {
            // new
            if (_cacheMapper.size() == _capacity) {
                // evict if full
                evict(1);
            }
            node_handle new_node = createNode(key, val);
            attachFront(new_node);
            _cacheMapper.insert({key, new_node});
        }
    }

    /**
     * @brief Look up a value by key.
     * @return The value if found (entry promoted to MRU), or a default-constructed Value on miss.
     */
    Value get(const Key& key) {
        auto map_it = _cacheMapper.find(key);
        if (map_it == _cacheMapper.end()) {
            return Value();
        }
        touch(map_it->second);
        return _head->value.second;
    }

    /**
     * @brief Evict up to @p n least recently used entries.
     */
    void evict(size_t n) {
        for (size_t i = 0; i < n && _tail != nullptr; ++i) {
            node_handle victim = _tail;
            detach(victim);
            _cacheMapper.erase(victim->value.first);
            deleteNode(victim);
        }
    }

    size_t getCapacity() const noexcept { return _capacity; }
    size_t size() const noexcept { return _cacheMapper.size(); }

private:
    struct key_hasher {
        std::size_t operator()(const Key& k) const { return k.hash(); }
    };

    using node_type = LruNode<Key, Value>;
    using node_handle = node_type*;

    // `touch` is the most important method as it maintains the sequence from the most to the last.
    void touch(node_handle node) {
        if (node == nullptr || node == _head) {
            return;
        }
        detach(node);
        attachFront(node);
    }

    static node_handle createNode(const Key& key, const Value& val) {
        return new node_type(key, val);
    }

    void deleteNode(node_handle node) {
        delete node;
    }

    void attachFront(node_handle node) {
        node->prev = nullptr;
        node->next = _head;
        if (_head != nullptr) {
            _head->prev = node;
        } else {
            _tail = node;
        }
        _head = node;
    }

    void detach(node_handle node) {
        if (node->prev != nullptr) {
            node->prev->next = node->next;
        } else {
            _head = node->next;
        }

        if (node->next != nullptr) {
            node->next->prev = node->prev;
        } else {
            _tail = node->prev;
        }

        node->prev = nullptr;
        node->next = nullptr;
    }

    void clear() {
        node_handle node = _head;
        while (node != nullptr) {
            node_handle next = node->next;
            delete node;
            node = next;
        }
        _head = nullptr;
        _tail = nullptr;
        _cacheMapper.clear();
    }

    // _cacheMapper is with Key and double-linked list.
    // Key is used to address which node.
    // double-liked list is used to maintain the fastest sequence.
    // In order to speed up the addressing, the recent accessed node must be placed in the first.
    // Otherwise, accessing must be start from the 1-st node and scan to the specific node.
    std::unordered_map<Key, node_handle, key_hasher> _cacheMapper;
    size_t _capacity;
    node_handle _head = nullptr;
    node_handle _tail = nullptr;
};

}  // namespace lru
