#ifndef MATH_SYMEIG_CUDA_H
#define MATH_SYMEIG_CUDA_H
#include <vector>

#include <torch/script.h>

namespace gpe {
std::tuple<torch::Tensor, torch::Tensor> symeig_cuda_forward(const torch::Tensor& matrices);
}

#endif // MATH_SYMEIG_CUDA_H
