#ifndef MATH_SYMEIG_H
#define MATH_SYMEIG_H

#include <vector>
#include <torch/script.h>

namespace gpe {
std::vector<torch::Tensor> symeig_cuda_forward(const torch::Tensor& matrices);
std::vector<torch::Tensor> symeig_cpu_forward(const torch::Tensor& matrices);

inline std::vector<torch::Tensor> symeig(const torch::Tensor& matrices) {
    if (matrices.is_cuda())
        return symeig_cuda_forward(matrices);
    else
        return symeig_cpu_forward(matrices);
}

}

#endif
