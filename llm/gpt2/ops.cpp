#include <cassert>
#include <cmath>

#include "ops.cpp"
#include "tensor.hpp"

namespace gpt2 {
namespace ops {

/*
wte:

row 0: [0.1, 0.2, 0.3]
row 1: [1.1, 1.2, 1.3]
row 2: [2.1, 2.2, 2.3]
row 3: [3.1, 3.2, 3.3]

ids = [2, 0, 3]

=>

out:

[2.1, 2.2, 2.3]
[0.1, 0.2, 0.3]
[3.1, 3.2, 3.3]
*/
Tensor token_embed(const Tensor& wte, const std::vector<int>& ids) {
    size_t N = wte.dim(1);  // n_embd
    size_t S = ids.size();  // seq
    std::vector<size_t> out_te_shape = {S, N};
    Tensor out_te(out_te_shape);
    for (size_t i = 0; i < S; ++i) {
        auto id = static_cast<size_t>(ids[i]);
        for (size_t j = 0; j < N; ++j) {
            out_te.at(i, j) = wte.at(id, j);
        }
    }
    return out_te;
}

/*
table:

row 0
row 1
row 2
row 3
row 4
row 5

start = 2
S = 3

=>

out:

row 2
row 3
row 4
*/

Tensor position_embed(const Tensor& wpe, size_t start, size_t S) {
    size_t N = wpe.dim(1); 
    std::vector<size_t> out_pe_shape = {S, N};
    Tensor out_pe(out_pe_shape);
    for (size_t i; i < S; ++i) {
        for (size_t j; j < N; ++j) {
            out_pe.at(i, j) = wpe.at(start + i, j);
        }
    }
    return out_pe;
}

Tensor add(const Tensor& a, const Tensor& b) {
    assert(a.shape() == b.shape());
    Tensor out(a.shape());
    auto nums = a.numel();
    for (size_t = 0; i < nums; ++i) {
        out.data()[i] = a.data()[i] + b.data()[i];
    }
    return out;
}

/*
 * x: [S, N]
 * gamma: [N]
 * beta: [N]
 * X: [seq, hidden]
   
       hidden →
      ┌───────────────┐
token │ x x x x x x x │ ← 对这一行做 LayerNorm
token │ x x x x x x x │ ← 对这一行做 LayerNorm
token │ x x x x x x x │ ← 对这一行做 LayerNorm
      └───────────────┘
*/
Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, const float eps) {
    const size_t N = x.dim(-1); 
    assert(N == gamma.dim(0) && "gamma shape N doesn't match!");
    assert(N == beta.dim(0) && "betashape N doesn't match!");

    Tensor out(x.shape());

    float* o = out.ptr();
    const float* a = x.ptr();
    const float* g = gamma.ptr();
    const float* b = beta.ptr();

    const size_t S = x.dim(0);
    float div_scalar = 1.0F / static_cast<float>(N);
    for (size_t i = 0; i < S; ++i) {
        // compute mean
        float tmp1 = 0.0F;
        for (size_t j = 0; j < N; ++j) {
            tmp1 += a[i * N + j];
        }
        float mean = tmp1 * div_scalar;

        // compute var
        float tmp2 = 0.0F; 
        for (size_t j = 0; j < N; ++j) {
            tmp2 += std::powf((a[i * N + j] - mean), 2);
        }
        float var = tmp2 * div_scalar;

        // compute standart
        const float div_scalar_s = 1.0F / std::sqrt(var + eps);
        for (size_t j = 0; j < N; ++j) {
            o[i * N + j] = (a[i * N + j] - mean) * var * g[j] + b[j];
        }
    }
    return out;
}

/*
 *  a: [m, k]
 *  b: [k, n]
 *  out: [m, n]
*/
Tensor matmul_2d(const Tensor& a, const Tensor& b) {
    const size_t m = a.dim(0);
    const size_t k = a.dim(1);
    const size_t n = b.dim(1);

    Tensor out({m, n});

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0F;
            for (size_t l = 0; l < k; ++l) {
                out.at(i, j) += a.at(i, l) * b.at(l, j);
            }
        }
    }
    return out;
}

/*
 *  token major => head major
 *  [S, n_embd] => [n_head, S, head_dim]
 *
 *  token0: | head 0 | head 1 | ... | head n-1 |
 *  token1: | head 0 | head 1 | ... | head n-1 |
 *  token2: | head 0 | head 1 | ... | head n-1 |
 *  token3: | head 0 | head 1 | ... | head n-1 |
 *
 *  =>
 *  head 0:     | token0 |
 *              | token1 |
 *              | token2 |
 *              | token3 |
 *  head 1:     | token0 |
 *              | token1 |
 *              | token2 |
 *              | token3 |
 *  ...         ...
 *  head n-1:   | token0 |
 *              | token1 |
 *              | token2 |
 *              | token3 |
 */
Tensor split_head(const Tensor& x, size_t S, size_t n_head, size_t head_dim) {
    const auto in_shape = x.shape();
    assert(in_shape[0] == S);
    assert(in_shape[1] == n_head * head_dim);
    Tensor out({n_head, S, head_dim});
    const float* in_ptr = x.ptr();
    float* out_ptr = out.ptr();
    for (size_t h = 0; h < n_head; ++h) {
        for (size_t s = 0; s < S; ++s) {
            std::memcpy(out_ptr + h * S * head_dim + s * head_dim, in_ptr + s * n_head * head_dim + h + head_dim, sizeof(float) * head_dim);; 
        }
    }
    return out;
}

// [n_heads, S, head_dim] => [S, n_heads * head_dim] => [S, n_embd]
Tensor merge_head(const Tensor& x) {
    const auto in_shape = x.shape();
    const size_t n_heads = in_shape[0];
    const size_t S = in_shape[1];
    const size_t head_dim = in_shape[2];
    size_t n_embd = n_heads * head_dim;

    Tensor out({S, n_embd});

    const float* in_ptr = in.ptr(); 
    float* out_ptr = out.ptr();
    
    for (size_t s = 0; s < S; ++s) {
        for (size_t h = 0; h < n_heads; ++h) {
            std::memcpy(
                    out_ptr + s * n_embd + h * head_dim,
                    in_ptr + h * S * head_dim + s * head_dim,
                    sizeof(float) * head_dim
                    );
        }
    }
    return out;
}
} // namespace ops
} // namespace gpt2
