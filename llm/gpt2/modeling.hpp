#pragma once

#include <vector>
#include <string>
#include <random>

#include "kvcache.hpp"
#include "tensor.hpp"
#include "weights.hpp"
#include "tokenizer.hpp"

namespace gpt2 {

class GPT2 {
public:
    GPT2(const GPT2weights& m) : w_m(std::move(m)) {
        m_hidden_dim = m.config.embd / m.config.n_head;
        m_kv_cache = KVCACHE(m.config.n_layer);
        m_scale = 1.0F / std::sqrt(static_cast<float>(m_hidden_dim));
    }

    std::vector<float> forward(const std::vector<int>& tokens, size_t n_past);
    void transfomer_layer(size_t layer_id, Tensor& x, size_t n_past);
    void attn(size_t layer_id, Tensor& x, size_t n_past);
    void mlp(size_t layer_id, Tensor& x, size_t n_past);

    int temperature_search(const std::vector<float>& logits, float temperature, int top_k);
    std::string generate(const tk::Tokenizer& token, const std::string& prompt,
                         int max_tokens, float temperature, int top_k, size_t seed);
    
private:
    GPT2Weights m_w;
    size_t m_hidden_dim;
    KVCACHE m_kv_cache;
    float m_scale = 0.0F;
    std::mt19937_t m_rnd;
};

}   // namespace gpt2
