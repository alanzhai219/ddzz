#include "tensor.hpp"

namespace gpt2 {

static size_t numel(std::vector<size_t> shape) {
   if (shape.empty()) {
      return 0;
   } 
   size_t num = 1;
   for (auto v : shape) {
      num *= v; 
   }
   return num;
}

Tensor::Tensor(std::vector<size_t> shape, float fill) {
    m_shape = shape;
    m_data.resize(numel(shape));
    for (auto v : m_data) {
       v = fill; 
    }
}

Tensor::Tensor() {

}

}
