#pragma once

#include <vector>

#include "kvcache"
#include "weights.hpp"

namespace gpt2 {

class GPT2 {
public:
    GPT2(const GPT2weights& m) : w_m(std::move(m)) {
        m_hidden_dim = m.config.embd / m.config.n_head;
        m_kv_cache = KVCACHE(m.config.n_layer);
    }

    std::vector<float> forward(const std::vector<int>& tokens, size_t n_past);

private:
    GPT2Weights m_w;
    size_t m_hidden_dim;
    KVCACHE m_kv_cache;
};

}   // namespace gpt2
