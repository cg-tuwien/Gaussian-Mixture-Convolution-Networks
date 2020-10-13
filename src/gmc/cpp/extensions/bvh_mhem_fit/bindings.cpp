#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "bindings.h"
#include "implementation.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> bvh_mhem_fit_forward(const torch::Tensor& mixture, int n_components_target) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(mixture));
    return bvh_mhem_fit::forward_impl(mixture, n_components_target);
}


std::tuple<torch::Tensor, torch::Tensor> bvh_mhem_fit_backward(const torch::Tensor& grad_output,
                                                           const torch::Tensor& mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& aabbs,
                                                           const torch::Tensor& xes,
                                                           bool requires_grad_mixture, bool requires_grad_xes) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(mixture));
    return bvh_mhem_fit::backward_impl(grad_output, mixture, bvh_nodes, aabbs, xes, requires_grad_mixture, requires_grad_xes);
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cuda_bvh_forward, "cuda_bvh_forward (CUDA)");
  m.def("backward", &cuda_bvh_backward, "cuda_bvh_backward (CUDA)");
}
#endif
