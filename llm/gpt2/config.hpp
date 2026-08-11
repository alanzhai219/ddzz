#pragma once
#include <cstddef>

namespace gpt2 {

// gpt2config: https://huggingface.co/openai-community/gpt2/raw/main/config.json
struct GPT2Config {
    size_t vocab_size    = 50257; // 50257
    size_t n_embd        = 768;   // 768
    size_t n_layer       = 12;    // 12
    size_t n_head        = 12;    // 12
    size_t max_pos_embed = 1024;  // 1024
};

} // namespace gpt2