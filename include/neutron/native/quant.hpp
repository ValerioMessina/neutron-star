#pragma once
#include "neutron/native/gguf.hpp"
#include <cstddef>
namespace neutron::native {
float dequant(const Tensor & tensor, size_t element);
float dot_row(const Tensor & tensor, size_t row, const float * x);
}
