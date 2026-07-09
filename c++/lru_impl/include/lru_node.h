#pragma once

#include <utility>

namespace lru {

template <typename Key, typename Value>
struct LruNode {
    using value_type = std::pair<Key, Value>;

    explicit LruNode(const Key& key, const Value& val)
        : value(key, val), prev(nullptr), next(nullptr) {}

    value_type value;
    LruNode* prev;
    LruNode* next;
};

}  // namespace lru
