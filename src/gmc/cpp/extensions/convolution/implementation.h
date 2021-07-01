#ifndef BVH_MHEM_FIT_IMPLEMENTATION
#define BVH_MHEM_FIT_IMPLEMENTATION
#include <tuple>
#include <torch/script.h>

namespace convolution {
struct ForwardOutput {
    torch::Tensor mixture;

    ForwardOutput clone() {
        return {mixture.clone()};
    }
};

ForwardOutput forward_impl(const at::Tensor& data, const at::Tensor& kernels);

std::pair<at::Tensor, at::Tensor> backward_impl(torch::Tensor grad, const ForwardOutput& forward_out);

}
#endif
