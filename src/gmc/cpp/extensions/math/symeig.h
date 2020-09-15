#ifndef MATH_SYMEIG_H
#define MATH_SYMEIG_H

#include <vector>
#include <tuple>

#include <torch/script.h>

#include "math/symeig_cpu.h"
#include "math/symeig_cuda.h"

namespace gpe {

inline std::tuple<torch::Tensor, torch::Tensor> symeig(const torch::Tensor& matrices) {
    if (matrices.is_cuda())
        return symeig_cuda_forward(matrices);
    else
        return symeig_cpu_forward(matrices);
}

}

#endif
