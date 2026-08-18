#pragma once

#include <vector>

#include "tensor.hpp"

namespace gpt2 {
namespace ops {

/*
 *  wte: [vocab_size, n_embed]
*/
Tensor token_embed(const Tensor& wte, const std::vector<int>& ids);

/*
 *  wpt: [n_positions, n_embed]
 * */
Tensor position_embed(const Tensor& wpe, size_t start, size_t S);

Tensor add(const Tensor& a, const Tensor& b);

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps = 1e-5);

Tensor matmul_2d(const Tensor& a, const Tensor& b);

Tensor split_head(const Tensor& x, size_t S, size_t n_head, size_t head_dim);

Tensor merge_head(const Tensor& x);
}

}
}
