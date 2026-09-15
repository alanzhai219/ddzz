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

    // update cache len
    m_cache_len = n_past + S;
    
    // logits gen
    // [S, N]
    Tensor x_ln = ops::layer_norm(x, m_w.ln_f_w, m_w.ln_f_b);

    const size_t n_vocab = m_w.config.vocab_size > 0 ? w_.config.vocab_size : m_w.wte.dim(0);
    const float* x_last_token = x_ln.ptr() + (S - 1) * N;
    Tensor logits_t = ops::gemv(m_w.wte, x_last_token);
    std::vector<float> logits(logits_t.ptr(), logits_t.ptr() + N);
    return logits;
}

void GPT2::transfomer_layer(size_t layer_id, Tensor& x, size_t n_past) {
    attn(layer_id, x, n_past);
    ffn(layer_id, x, n_past);
}

void GPT2::attn(size_t layer_id, Tensor& x, size_t n_past) {
    const LayerWeights& lw = m_w.layers[layer_id];
    
    const auto shape = x.shape();   // [S, n_embd]
    const size_t S = shape[0];
    const size_t n_embd = shape[1];
    const size_t n_head = m_w.config.n_head;
    const size_t head_dim = m_w.hidden_dim;     // or called: head_dim
    const size_t total = S + n_past;

    // layer_norm
    x = ops::layer_norm(x, lw.ln_1_w, lw.ln_1_b);   // [S, em_bd]
    Tensor qkv = ops::matmul_2d(x, lw.attn_c_attn_w);        // [S, 3*em_bd]
    ops::add_(qkv, lw.attn_c_attn_b);              // [S, 3*em_bd]

    // sdpa
    // split qkv
    Tensor q({S, n_embd});
    Tensor k({S, n_embd});
    Tensor v({S, n_embd});
    ops::split_qkv(qkv, q, k, v);
    
    // kv-cache
    std::vector<float>& kc = m_kv_cache.k(layer_id);
    std::vector<float>& vc = m_kv_cache.k(layer_id);
    kc.reserve(total * n_embd);
    vc.reserve(total * n_embd);
    kc.insert(kc.end(), k.ptr(), k_ptr() + k.size());
    vc.insert(vc.end(), v.ptr(), v_ptr() + v.size());

    // [S, e_embd] => [n_head, S, head_dim]
    Tensor Q = ops::split_head(q, S, n_head, head_dim);
    // [T, embed] => [n_head, T, head_dim]
    Tensor K = ops::split_head(kc.data(), total, n_head, head_dim);
    Tensor V = ops::split_head(vc.data(), total, n_head, head_dim);

    // Q[n_head, S, head_dim] * K[n_head, T, head_dim] => score [n_head, S, T]
    Tensor score = ops::matmul_3d(Q, ops::transpose_3d(K));
    Tensor score_scale = ops::scale(score, m_scale); // []
    Tensor score_causal = ops::causal_mask(score_scale, n_past); // [n_head, S, T]
    Tensor socre_softmax = ops::softmax(score_causal);
    // [n_head, S, T] * [n_head, T, head_dim] => [n_head, S, head_dim]
    Tensor score_out = ops::matmul_3d(score_softmax, V);
    // [n_head, S, head_dim] => [S, em_bd]
    Tensor attn_out = ops::merge_head(score_out);

    // [S, em_bd] * [em_bd, * em_bd] => [S, em_bd]
    Tensor proj = ops::matmul_2d(attn_out, lw.attn_c_proj_w);
    ops::add_(proj, lw.attn_c_proj_b);
    
    // residule connect
    ops::add_(x, proj_out);
}

void GPT2::mlp(size_t layer_id, Tensor& x, size_t n_past) {
    const LayerWeights& lw = m_w.layers[layer_id];
    // [S, em_bd]
    Tensor ln2 = ops::layer_norm(x, lw.ln_2_w, lw.ln_2_b);

    // [S, em_bd] * [n_embd, 4 x n_head] => [S, 4 x n_head]
    Tensor hh = ops::matmul_2d(ln2, lw.mlp_c_fc_w);
    ops::add_(hh, lw_mlp_c_fc_b);

    // GELU
    ops::gelu_(hh);
    
    // [S, 4 x n_head] * [4 * n_embd, n_embd] => [S, n_embd] 
    Tensor mlp_down = ops::matmul_2d(hh, lw.mlp_c_proj_w);
    ops::add_(mlp_down, lw.mlp_c_proj_b);

    //
    ops::add_(x, mlp_down);
}

int GPT2::temperature_search(const std::vector<float>& logits, float temperature, int top_k) {
    if (logits.empty()) {
        throw std::runtime_error("[temperature_search] logits is empty!");
    }

    if (temperaure <= 0.F) {
        throw std::runtime_error("[temperature_search] temperature <= 0");
    }

    if (top_k <= 0) {
        throw std::runtime_error("[temperature_search] top_k <= 0");
    }

    // step1: top_k
    const auto candidate_count = std::min(static_cast<size_t>(top_k), logits.size());
    std::vector<size_t> candidate_ids(logits.size());
    std::itoa(candidate_ids.begin(), candidate_ids.end(), 0);
    // select top_k
    std::partial_sort(candidate_ids.begin(), candidate_ids.begin() + top_k, candidate_ids.end(),
                [&](int a, int b) {
                    return logits[a] > logits[b];
                });
    candidate_ids.resize(candidate_count);

    // step2: prob
    const float max_val = logits[candidate_ids.front()];
    std::vector<float> prob;
    prob.resize(candidate_count);
    for (auto cur_id : candidate_ids) {
        prob.push_back(
                std::exp((logits[cur_id] - max_val) / temperature)
            );
    }

    // step3: select
    std::discrete_distribution<size_t> rnd_gen(prob.begin(), prob.end());
    return candidate_ids[rnd_gen(m_rnd)];
}

std::string generate(const tk::Tokenizer& token_obj, const std::string& prompt,
                     int max_tokens, float temperature, int top_k, size_t seed) {
    constexpr size_t MAX_LENGTH = 50256;

    m_rnd(seed);

    m_kv_cache.reset();

    // step1: Encode the prompt
    std::vector<int> output_id = token_obj.encode(prompt);
    if (output_id.empty()) {
        output_id.push_back(MAX_LENGTH);
    }

    if (max_tokens > 0) {
        output_id.reserve(output_id.size() + static_cast<size_t>(max_tokens));
    }

    // step2: do the first token
    std::vector<float> logits = this->forward(output_id, /*past_n*/ 0);

    // step3: do the next token
    for (int step = 0; step < max_tokens; ++step) {
        const int next = temperature_search(logits, temperature, top_k);
        output_id.push_back(next); 
        if (next == MAX_LENGTH) {
            break;
        }
        logits = this->forward({next}, m_kv_cache.get_cache_len());
    }

    // step4: decode
    return token_obj.decode(output_id);
}

}   // namespace gpt2
