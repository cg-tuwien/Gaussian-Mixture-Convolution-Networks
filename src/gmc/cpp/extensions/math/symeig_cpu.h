#ifndef MATH_SYMEIG_CPU_H
#define MATH_SYMEIG_CPU_H
#include <vector>
#include <tuple>

#include <torch/script.h>

namespace gpe {
std::tuple<torch::Tensor, torch::Tensor> symeig_cpu_forward(const torch::Tensor& matrices);
}
#endif // MATH_SYMEIG_CPU_H
