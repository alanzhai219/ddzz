#include <cassert>
#include "tensor.hpp"

namespace gpt2 {

Tensor::Tensor(const std::vector<size_t>& shape, float fill) {
   m_shape = shape;
   m_data.resize(numel(shape));
   for (auto &v : m_data) {
      v = fill; 
   }
   compute_strides();
}

Tensor::Tensor(const std::vector<size_t>& shape, const std::vector<float>& value) {
   assert(numel() == value.size());
   m_shape = shape;
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
