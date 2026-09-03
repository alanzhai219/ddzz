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

}

}   // namespace gpt2
