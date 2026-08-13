#pragma once

#include <string>
#include <unordered_map>

#include "config.hpp"
#include "tensor.hpp"

namespace gpt2 {

struct LayerWeights {
    Tensor ln_1_w;          // [n_embd]
    Tensor ln_1_b;          // [n_embd]
    Tensor attn_c_attn_w;   // [n_embd, 3 * n_embd]
    Tensor attn_c_attn_b;   // [3 * n_embd]
    Tensor attn_c_proj_w;   // [n_embd, n_embd]
    Tensor attn_c_proj_b;   // [n_embd]
    Tensor ln_2_w;          // [n_embd]
    Tensor ln_2_b;          // [n_embd]
    Tensor mlp_c_fc_w;      // [n_embd, 4 * n_embd]
    Tensor mlp_c_fc_b;      // [4 * n_embd]
    Tensor mlp_c_proj_w;    // [4 * n_embd, n_embd]
    Tensor mlp_c_proj_b;    // [n_embd]
};

struct GPT2Weights {
    GPT2Config config;
    Tensor wpe;     // [max_position_embeddings, n_embd] position embedding
    Tensor wte;     // [n_vocab, n_embd] word embedding
    std::vector<LayerWeights> layers;
    Tensor ln_f_w;  // [n_embd] final layer norm weight
    Tensor ln_f_b;  // [n_embd] final layer norm bias
};

// load safetensors
std::unordered_map<std::string, Tensor> load_safetensors(const std::string& filename);

// build tensors to GPT2Weights
GPT2Weights build_gpt2_weights(const std::unordered_map<std::string, Tensor>& tensors, const GPT2Config& config);
} // namespace gpt2
