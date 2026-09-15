#include "modeling.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "ops.hpp"

namespace gpt2 {

std::vector<float> GPT2::forward(const std::vector<int>& tokens, size_t n_past) {
    if (tokens.empty()) throw std::invalid_argument("forward: tokens must not be empty");
    if (n_past != m_kv_cache.get_cache_len()) {
        throw std::runtime_error("forward: n_past does not match KV cache length");
    }
    const size_t seq = tokens.size();
    const size_t n_embd = m_w.config.n_embd;
    Tensor x = ops::add(ops::token_embed(m_w.wte, tokens),
                        ops::position_embed(m_w.wpe, n_past, seq));
    for (size_t layer = 0; layer < m_w.config.n_layer; ++layer) {
        transfomer_layer(layer, x, n_past);
    }
    m_kv_cache.set_cache_len(n_past + seq);

    Tensor normalized = ops::layer_norm(x, m_w.ln_f_w, m_w.ln_f_b);
    const float* last = normalized.ptr() + (seq - 1) * n_embd;
    Tensor logits = ops::gemv(m_w.wte, last);
    return logits.data();
}

void GPT2::transfomer_layer(size_t layer_id, Tensor& x, size_t n_past) {
    attn(layer_id, x, n_past);
    mlp(layer_id, x, n_past);
}

void GPT2::attn(size_t layer_id, Tensor& x, size_t n_past) {
    const LayerWeights& layer = m_w.layers.at(layer_id);
    const size_t seq = x.dim(0);
    const size_t n_embd = m_w.config.n_embd;
    const size_t total = n_past + seq;

    Tensor normalized = ops::layer_norm(x, layer.ln_1_w, layer.ln_1_b);
    Tensor qkv = ops::matmul_2d(normalized, layer.attn_c_attn_w);
    ops::add_(qkv, layer.attn_c_attn_b);

    Tensor q, k, v;
    ops::split_qkv(qkv, q, k, v);
    std::vector<float>& k_cache = m_kv_cache.k(layer_id);
    std::vector<float>& v_cache = m_kv_cache.v(layer_id);
    k_cache.reserve(total * n_embd);
    v_cache.reserve(total * n_embd);
    k_cache.insert(k_cache.end(), k.ptr(), k.ptr() + k.numel());
    v_cache.insert(v_cache.end(), v.ptr(), v.ptr() + v.numel());

    Tensor query = ops::split_head(q.ptr(), seq, m_w.config.n_head, m_hidden_dim);
    Tensor key = ops::split_head(k_cache.data(), total, m_w.config.n_head, m_hidden_dim);
    Tensor value = ops::split_head(v_cache.data(), total, m_w.config.n_head, m_hidden_dim);
    Tensor scores = ops::matmul_3d(query, ops::transpose_3d(key));
    scores = ops::scale(scores, m_scale);
    scores = ops::causal_mask(scores, n_past);
    scores = ops::softmax(scores);
    Tensor attended = ops::matmul_3d(scores, value);
    Tensor projection = ops::matmul_2d(ops::merge_head(attended), layer.attn_c_proj_w);
    ops::add_(projection, layer.attn_c_proj_b);
    ops::add_(x, projection);
}

void GPT2::mlp(size_t layer_id, Tensor& x, size_t n_past) {
    (void)n_past;
    const LayerWeights& layer = m_w.layers.at(layer_id);
    Tensor hidden = ops::matmul_2d(
        ops::layer_norm(x, layer.ln_2_w, layer.ln_2_b), layer.mlp_c_fc_w);
    ops::add_(hidden, layer.mlp_c_fc_b);
    ops::gelu_(hidden);
    Tensor output = ops::matmul_2d(hidden, layer.mlp_c_proj_w);
    ops::add_(output, layer.mlp_c_proj_b);
    ops::add_(x, output);
}

int GPT2::temperature_search(const std::vector<float>& logits, float temperature,
                             int top_k) {
    if (logits.empty()) throw std::invalid_argument("temperature_search: empty logits");
    if (temperature <= 0.0F || top_k <= 0) {
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    const size_t count = std::min(static_cast<size_t>(top_k), logits.size());
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + count, ids.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    ids.resize(count);

    const float max_logit = logits[ids.front()];
    std::vector<double> probabilities;
    probabilities.reserve(count);
    for (int id : ids) probabilities.push_back(std::exp((logits[id] - max_logit) / temperature));
    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return ids[distribution(m_rnd)];
}

std::string GPT2::generate(const tk::Tokenizer& tokenizer, const std::string& prompt,
                           int max_tokens, float temperature, int top_k, uint64_t seed) {
    constexpr int end_of_text = 50256;
    m_rnd.seed(seed);
    reset_cache();
    std::vector<int> ids = tokenizer.encode(prompt);
    if (ids.empty()) ids.push_back(end_of_text);
    if (max_tokens > 0) ids.reserve(ids.size() + static_cast<size_t>(max_tokens));

    std::vector<float> logits = forward(ids, 0);
    for (int step = 0; step < max_tokens; ++step) {
        const int next = temperature_search(logits, temperature, top_k);
        ids.push_back(next);
        if (next == end_of_text) break;
        logits = forward({next}, m_kv_cache.get_cache_len());
    }
    return tokenizer.decode(ids);
}

}  // namespace gpt2
