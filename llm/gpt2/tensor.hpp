#pragma once

#include <vector>

namespace gpt2 {

struct Tensor {
    Tensor()  = default;
    Tensor(std::vector<size_t> shape, std::vector<float> value);
    Tensor(std::vector<size_t> shape, float fill = 0.0F);
    
    virtual ~Tensor() = default;
    
    // shape: [2,3,4] => stride [12,4,1]
    // dim:   n-1, n-2, ..., 1, 0
    // idx of vec: 0, 1, 2, ..., n-2, n-1
    void compute_strides() {
        m_stride.resize(m_shape.size()); 
        m_stride.back() = 1;
        for (size_t idx = m_shape.size() - 1; idx > 0; --idx) {
            // TODO
        }
    }



private:
  std::vector<float>  m_data;
  std::vector<size_t> m_shape;
  std::vector<size_t> m_stride;
};
};
