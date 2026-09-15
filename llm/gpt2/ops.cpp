#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace gpt2::ops {

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
    Tensor out({ids.size(), N});
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] < 0 || static_cast<size_t>(ids[i]) >= wte.dim(0)) {
            throw std::out_of_range("token_embed: token id out of range");
        }
        auto id = static_cast<size_t>(ids[i]);
        for (size_t j = 0; j < N; ++j) {
            out.at(i, j) = wte.at(id, j);
        }
    }
    return out;
}

Tensor position_embed(const Tensor& wpe, size_t start, size_t seq) {
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
    size_t N = wpe.dim(1);
    if (start + seq > wpe.dim(0)) {
        throw std::out_of_range("position_embed: context length exceeded");
    }
    std::vector<size_t> out_pe_shape = {seq, N};
    Tensor out_pe(out_pe_shape);
    for (size_t i = 0; i < seq; ++i) {
        for (size_t j = 0; j < N; ++j) {
            out_pe.at(i, j) = wpe.at(start + i, j);
        }
    }
    return out_pe;
}

Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape() != b.shape()) {
        throw std::invalid_argument("add: shape mismatch");
    }
    Tensor out(a.shape());
    auto nums = a.numel();
    for (size_t i = 0; i < nums; ++i) {
        out.data()[i] = a.data()[i] + b.data()[i];
    }
    return out;
}

void add_(Tensor& a, const Tensor& b) {
    const size_t width = b.numel();
    if (width == 0 || a.numel() % width != 0) {
        throw std::invalid_argument("add_: incompatible shapes");
    }
    float* a_ptr = a.ptr();
    const float* b_ptr = b.ptr();
    const size_t rows = a.numel() / width;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t i = 0; i < width; ++i) {
            a_ptr[row * width + i] += b_ptr[i];
        }
    }
}

Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, float eps) {
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
    const size_t n = x.dim(x.ndims() - 1);
    if (gamma.numel() != n || beta.numel() != n) {
        throw std::invalid_argument("layer_norm: parameter shape mismatch");
    }
    Tensor out(x.shape());
    const size_t rows = x.numel() / n;
    for (size_t row = 0; row < rows; ++row) {
        const float* src = x.ptr() + row * n;
        float* dst = out.ptr() + row * n;
        float mean = 0.0F;
        for (size_t i = 0; i < n; ++i) {
            mean += src[i];
        }
        mean /= static_cast<float>(n);
        float var = 0.0F;
        for (size_t i = 0; i < n; ++i) {
            const float d = src[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(n);
        const float inv_std = 1.0F / std::sqrt(var + eps);
        for (size_t i = 0; i < n; ++i) {
            dst[i] = (src[i] - mean) * inv_std * gamma.ptr()[i] + beta.ptr()[i];
        }
    }
    return out;
}

Tensor matmul_2d(const Tensor& a, const Tensor& b) {
/*
 * a: [m, k]
 * b: [k, n]
 * out: [m, n]
 */
    if (a.ndims() != 2 || b.ndims() != 2 || a.dim(1) != b.dim(0)) {
        throw std::invalid_argument("matmul_2d: incompatible shapes");
    }
    const size_t m = a.dim(0), k = a.dim(1), n = b.dim(1);
    Tensor out({m, n});
    for (size_t i = 0; i < m; ++i) {
        for (size_t p = 0; p < k; ++p) {
            const float av = a.ptr()[i * k + p];
            const float* brow = b.ptr() + p * n;
            float* orow = out.ptr() + i * n;
            for (size_t j = 0; j < n; ++j) {
                orow[j] += av * brow[j];
            }
        }
    }
    return out;
}

// [S, 3*n_embd] => [S, n_embd], [S, n_embd], [S, n_embd]
void split_qkv(const Tensor& qkv, Tensor& q, Tensor& k, Tensor& v) {
    if (qkv.ndims() != 2 || qkv.dim(1) % 3 != 0) {
        throw std::invalid_argument("split_qkv: expected [S, 3*N]");
    }
    const size_t seq = qkv.dim(0), n = qkv.dim(1) / 3;
    q = Tensor({seq, n}); k = Tensor({seq, n}); v = Tensor({seq, n});
    for (size_t i = 0; i < seq; ++i) {
        const float* row = qkv.ptr() + i * 3 * n;
        std::memcpy(q.ptr() + i * n, row, n * sizeof(float));
        std::memcpy(k.ptr() + i * n, row + n, n * sizeof(float));
        std::memcpy(v.ptr() + i * n, row + 2 * n, n * sizeof(float));
    }
}

Tensor split_head(const float* x, size_t seq, size_t n_head, size_t head_dim) {
/*
 * token major => head major
 * [S, n_embd] => [n_head, S, head_dim]
 *
 * token0: | head 0 | head 1 | ... | head n-1 |
 * token1: | head 0 | head 1 | ... | head n-1 |
 * token2: | head 0 | head 1 | ... | head n-1 |
 * token3: | head 0 | head 1 | ... | head n-1 |
 * 
 * =>
 * head 0:     | token0 |
 *             | token1 |
 *             | token2 |
 *             | token3 |
 * head 1:     | token0 |
 *             | token1 |
 *             | token2 |
 *             | token3 |
 * ...         ...
 * head n-1:   | token0 |
 *             | token1 |
 *             | token2 |
 *             | token3 |
 */
    const size_t n = n_head * head_dim;
    Tensor out({n_head, seq, head_dim});
    for (size_t h = 0; h < n_head; ++h) {
        for (size_t s = 0; s < seq; ++s) {
            std::memcpy(out.ptr() + (h * seq + s) * head_dim,
                        x + s * n + h * head_dim, head_dim * sizeof(float));
        }
    }
    return out;
}

Tensor merge_head(const Tensor& x) {
// [n_heads, S, head_dim] => [S, n_heads * head_dim] => [S, n_embd]
    if (x.ndims() != 3) {
        throw std::invalid_argument("merge_head: expected rank 3");
    }
    const size_t heads = x.dim(0), seq = x.dim(1), hd = x.dim(2), n = heads * hd;
    Tensor out({seq, n});
    for (size_t h = 0; h < heads; ++h) {
        for (size_t s = 0; s < seq; ++s) {
            std::memcpy(out.ptr() + s * n + h * hd,
                        x.ptr() + (h * seq + s) * hd, hd * sizeof(float));
        }
    }
    return out;
}

Tensor softmax(const Tensor& x) {
    Tensor out(x.shape());
    const size_t width = x.dim(x.ndims() - 1), rows = x.numel() / width;
    for (size_t r = 0; r < rows; ++r) {
        const float* src = x.ptr() + r * width;
        float* dst = out.ptr() + r * width;
        float max_val = src[0];
        for (size_t i = 1; i < width; ++i) {
            max_val = std::max(max_val, src[i]);
        }
        float sum = 0.0F;
        for (size_t i = 0; i < width; ++i) {
            dst[i] = std::exp(src[i] - max_val);
            sum += dst[i];
        }
        const float inv_sum = 1.0F / sum;
        for (size_t i = 0; i < width; ++i) {
            dst[i] *= inv_sum;
        }
    }
    return out;
}

void gelu_(Tensor& x) {
    constexpr float sqrt_2_over_pi = 0.7978845608028654F;
    constexpr float coeff = 0.044715F;
    for (float& value : x.data()) {
        const float inner = sqrt_2_over_pi * (value + coeff * value * value * value);
        value = 0.5F * value * (1.0F + std::tanh(inner));
    }
}

Tensor matmul_3d(const Tensor& a, const Tensor& b) {
/*
 * a: [head_n, S, head_dim]
 * b: [head_n, head_dim, T]
 * out: [head_n, S, T]
 */
    if (a.ndims() != 3 || b.ndims() != 3 || a.dim(0) != b.dim(0) || a.dim(2) != b.dim(1)) {
        throw std::invalid_argument("matmul_3d: incompatible shapes");
    }
    const size_t batch = a.dim(0), m = a.dim(1), k = a.dim(2), n = b.dim(2);
    Tensor out({batch, m, n});
    for (size_t h = 0; h < batch; ++h) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t p = 0; p < k; ++p) {
                const float av = a.ptr()[(h * m + i) * k + p];
                for (size_t j = 0; j < n; ++j) {
                    out.ptr()[(h * m + i) * n + j] += av * b.ptr()[(h * k + p) * n + j];
                }
            }
        }
    }
    return out;
}

Tensor causal_mask(const Tensor& a, size_t n_past) {
// [n_heads, S, T]
    if (a.ndims() != 3) {
        throw std::invalid_argument("causal_mask: expected rank 3");
    }
    Tensor out(a);
    const size_t seq = a.dim(1), total = a.dim(2);
    for (size_t h = 0; h < a.dim(0); ++h) {
        for (size_t s = 0; s < seq; ++s) {
            for (size_t t = n_past + s + 1; t < total; ++t) {
                out.at(h, s, t) = -std::numeric_limits<float>::infinity();
            }
        }
    }
    return out;
}

Tensor scale(const Tensor& x, float s) {
    Tensor out(x);
    for (float& value : out.data()) {
        value *= s;
    }
    return out;
}

Tensor transpose_2d(const Tensor& x) {
/*
 * [..., Y, Z] => [..., Z, Y]
 */
    if (x.ndims() != 2) {
        throw std::invalid_argument("transpose_2d: expected rank 2");
    }
    Tensor out({x.dim(1), x.dim(0)});
    for (size_t i = 0; i < x.dim(0); ++i) {
        for (size_t j = 0; j < x.dim(1); ++j) {
            out.at(j, i) = x.at(i, j);
        }
    }
    return out;
}

Tensor transpose_3d(const Tensor& x) {
/*
 * [D0, D1, D2] => [D0, D2, D1]
 */
    if (x.ndims() != 3) {
        throw std::invalid_argument("transpose_3d: expected rank 3");
    }
    Tensor out({x.dim(0), x.dim(2), x.dim(1)});
    for (size_t b = 0; b < x.dim(0); ++b) {
        for (size_t i = 0; i < x.dim(1); ++i) {
            for (size_t j = 0; j < x.dim(2); ++j) {
                out.at(b, j, i) = x.at(b, i, j);
            }
        }
    }
    return out;
}

Tensor gemv(const Tensor& x, const float* v) {
// [n_vocab, n_embd] * [n_embd]
    if (x.ndims() != 2) {
        throw std::invalid_argument("gemv: expected rank 2");
    }
    Tensor out({x.dim(0)});
    for (size_t i = 0; i < x.dim(0); ++i) {
        for (size_t j = 0; j < x.dim(1); ++j) {
            out.ptr()[i] += x.at(i, j) * v[j];
        }
    }
    return out;
}

}  // namespace gpt2::ops
