#include <cassert>

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
} // namespace ops
} // namespace gpt2
