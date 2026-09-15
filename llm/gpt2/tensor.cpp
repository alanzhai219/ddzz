#include <cassert>
#include <utility>
#include "tensor.hpp"

namespace gpt2 {

Tensor::Tensor(const std::vector<size_t>& shape, float fill) {
   m_shape = shape;
   size_t count = 1;
   for (size_t dim : shape) count *= dim;
   m_data.resize(count);
   for (auto &v : m_data) {
      v = fill; 
   }
   compute_strides();
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& value) {
   m_shape = shape;
   assert(numel() == value.size());
   m_data = value;
   compute_strides();
}

Tensor::Tensor(const Tensor& other) {
   m_shape = other.m_shape;
   m_data = other.m_data;
   m_stride = other.m_stride;
}

Tensor::Tensor(Tensor&& other) {
   m_shape = std::move(other.m_shape);
   m_data = std::move(other.m_data);
   m_stride = std::move(other.m_stride);
}

Tensor& Tensor::operator=(const Tensor& other) {
   if (this != &other) {
      m_shape = other.m_shape;
      m_data = other.m_data;
      m_stride = other.m_stride;
   }
   return *this;
}

Tensor& Tensor::operator=(Tensor&& other) {
   if (this != &other) {
      m_shape = std::move(other.m_shape);
      m_data = std::move(other.m_data);
      m_stride = std::move(other.m_stride);
   }
   return *this;
}

} // namespace gpt2
