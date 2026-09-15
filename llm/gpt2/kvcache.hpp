#pragma once

#include <cstddef>
#include <vector>

namespace gpt2 {

struct KVCACHE {
    KVCACHE(size_t n_layer = 0, size_t n_cache_len = 0)
      : m_layer(n_layer), m_cache_len(n_cache_len) {
        reset();
    }

    void reset() {
        m_kcache.assign(m_layer, {});
        m_vcache.assign(m_layer, {});
        m_cache_len = 0;
    }

    std::vector<float>& k(size_t layer_id) {
        return m_kcache[layer_id];
    }

    std::vector<float>& v(size_t layer_id) {
        return m_vcache[layer_id];
    }

    size_t get_cache_len() const {
        return m_cache_len;
    }

    void set_cache_len(size_t cache_len) {
        m_cache_len = cache_len;
    }

    std::vector<std::vector<float>> m_kcache;
    std::vector<std::vector<float>> m_vcache;
    size_t m_layer;
    size_t m_cache_len;
};
}
