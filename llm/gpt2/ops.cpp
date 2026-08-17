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
} // namespace ops
} // namespace gpt2
