#include "modeling.hpp"
#include "tensor.hpp"

namespace gpt2 {

std::vector<float> GPT2::forward(const std::vector<int>& tokens, size_t n_past) {
    if (n_past != m_kv_cache.m_cache_len) {
        throw std::runtime_error("forward: n_past is NOT equal to kv_cache length");
    }

    const size_t S = token.size();
    const size_t N = m_w.config.n_embd;

    // embed
    Tensor te = ops::token_embed(m_w.wte, token);
    Tensor pe = ops::position_embed(m_w.wpe, n_past, S);
    Tensor x = ops::add(te, pe);

    // transformer
    for (size_t l = 0; l < m_w.config.n_layer; ++l) {
        transformer(l, x, n_past);
    }
}

void GPT2::transfomer_layer(size_t layer_id, Tensor& x, size_t n_past) {
    attn(layer_id, x, n_past);
    ffn(layer_id, x, n_past);
}

void GPT2::attn(size_t layer_id, Tensor& x, size_t n_past) {
    // TODO
}

void GPT2::ffn(size_t layer_id, Tensor& x, size_t n_past) {
    // TODO
}

}

}   // namespace gpt2
