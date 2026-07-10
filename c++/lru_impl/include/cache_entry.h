#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

#include "lru_cache.h"

namespace lru {

/**
 * @brief Base class for type-erased cache entries stored in MultiCache.
 */
class CacheEntryBase {
public:
    virtual ~CacheEntryBase() = default;

    enum class LookUpStatus : int8_t {
        Hit,
        Miss
    };
};

/**
 * @brief Typed cache entry wrapping an LruCache with get-or-create semantics.
 *
 * On a cache miss the caller-supplied builder functor creates the value, which is
 * then inserted into the cache.  Default-constructed values are treated as "empty"
 * and are never stored.
 *
 * @tparam KeyType   must satisfy LruCache Key requirements.
 * @tparam ValType   must be default-constructible and equality-comparable.
 * @tparam ImplType  storage backend (default: LruCache<KeyType, ValType>).
 */
template <typename KeyType, typename ValType, typename ImplType = LruCache<KeyType, ValType>>
class CacheEntry : public CacheEntryBase {
public:
    using ResultType = std::pair<ValType, CacheEntryBase::LookUpStatus>;

    explicit CacheEntry(size_t capacity) : _impl(capacity) {}

    /**
     * @brief Look up @p key; on miss call @p builder to create and cache the value.
     * @return {value, Hit/Miss}.
     */
    ResultType getOrCreate(const KeyType& key, std::function<ValType(const KeyType&)> builder) {
        if (0 == _impl.getCapacity()) {
            return {builder(key), CacheEntryBase::LookUpStatus::Miss};
        }
        ValType retVal = _impl.get(key);
        // If the value is default-constructed, treat it as a cache miss and call the builder.
        if (retVal == ValType()) {
            retVal = builder(key);
            if (retVal != ValType()) {
                _impl.put(key, retVal);
            }
            return {retVal, CacheEntryBase::LookUpStatus::Miss};
        }
        return {retVal, CacheEntryBase::LookUpStatus::Hit};
    }
private:
    ImplType _impl;
};

}  // namespace lru
