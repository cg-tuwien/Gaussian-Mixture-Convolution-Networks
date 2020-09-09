#ifndef MATH_SYMEIG_H
#define MATH_SYMEIG_H

#include <vector>
#include <tuple>

#include <torch/script.h>

namespace gpe {
std::vector<torch::Tensor> symeig_cuda_forward(const torch::Tensor& matrices);
std::vector<torch::Tensor> symeig_cpu_forward(const torch::Tensor& matrices);

inline std::tuple<torch::Tensor, torch::Tensor> symeig(const torch::Tensor& matrices) {
    std::vector<torch::Tensor> v;
    if (matrices.is_cuda())
        v = symeig_cuda_forward(matrices);
    else
        v = symeig_cpu_forward(matrices);
    assert(v.size() == 2);
    return std::make_tuple(std::move(v[0]), std::move(v[1]));
}

}

#endif
