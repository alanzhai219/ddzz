#pragma once

#include <cstddef>
#include <cassert>
#include <vector>
#include <string>
namespace gpt2 {

struct Tensor {
    Tensor()  = default;
    Tensor(const std::vector<size_t>& shape, const std::vector<float>& value);
    Tensor(const std::vector<size_t>& shape, float fill = 0.0F);
    Tensor(const Tensor&);
    Tensor(Tensor&&);
    Tensor& operator=(const Tensor&);
    Tensor& operator=(Tensor&&);
    
    virtual ~Tensor() = default;
    
    // shape: [2,3,4] => stride [12,4,1]
    // dim:   n-1, n-2, ..., 1, 0
    // idx of vec: 0, 1, 2, ..., n-2, n-1
    void compute_strides() {
        m_stride.resize(m_shape.size()); 
        m_stride.back() = 1;
        for (size_t idx = m_shape.size() - 1; idx > 0; --idx) {
            m_stride[idx - 1] = m_stride[idx] * m_shape[idx];
        }
    }

    void reshape(const std::vector<size_t>& shape) {
        assert(numel(shape) == m_data.size());
        m_shape = shape;
        compute_strides();
    }

    // access raw data
    const std::vector<float>& data() const {
        return m_data;
    }

    float* ptr() {
        return m_data.data();
    }

    // access shape
    const std::vector<size_t>& stride() const {
        return m_stride;
    }

    const std::vector<size_t>& shape() const {
        return m_shape;
    }

    size_t ndims() const {
        return m_shape.size();
    }

    size_t dim(size_t idx) const {
        assert(idx < m_shape.size());
        return m_shape[idx];
    }

    std::string shape_str() const {
        std::string str = "[";
        for (size_t i = 0; i < m_shape.size(); ++i) {
            str += std::to_string(m_shape[i]);
            if (i != m_shape.size() - 1) {
                str += ",";
            }
        }
        str += "]";
        return str;
    }

    size_t numel() const {
        size_t num = 1;
        for (auto v : m_shape) {
            num *= v;
        }
        return num;
    }

    // access tensor data by index
    float at(size_t i, size_t j, size_t k, size_t l) {
        const size_t offset = i * m_stride[0] + j * m_stride[1] + k * m_stride[2] + l * m_stride[3];
        return m_data[offset];
    }

    float at(size_t i, size_t j, size_t k) {
        const size_t offset = i * m_stride[0] + j * m_stride[1] + k * m_stride[2];
        return m_data[offset];
    }

    float at(size_t i, size_t j) {
        const size_t offset = i * m_stride[0] + j * m_stride[1];
        return m_data[offset];
    }

    float at(size_t i, size_t j, size_t k, size_t l) const {
        return at(i, j, k, l);
    }

    float at(size_t i, size_t j, size_t k) const {
        return at(i, j, k);
    }

    float at(size_t i, size_t j) const {
        return at(i, j);
    }

private:
  std::vector<float>  m_data;
  std::vector<size_t> m_shape;
  std::vector<size_t> m_stride;
};
} // namespace gpt2