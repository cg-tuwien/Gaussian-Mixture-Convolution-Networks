#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "bindings.h"
#include "implementation.h"

std::vector<torch::Tensor> bvh_mhem_fit_forward(const torch::Tensor& mixture, const BvhMhemFitConfig& config) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (mixture.is_cuda()) {
        assert (device_of(mixture).has_value());
        device_guard.set_device(device_of(mixture).value());
    }
    auto result = bvh_mhem_fit::forward_impl(mixture, config);
    return {result.fitting, result.bvh_mixture, result.bvh_nodes, result.bvh_aabbs, result.bvh_attributes};
}

torch::Tensor bvh_mhem_fit_backward(const torch::Tensor& grad,
                                    const torch::Tensor& fitting_mixture,
                                    const torch::Tensor& target_mixture,
                                    const torch::Tensor& bvh_mixture_inversed, const torch::Tensor& bvh_nodes, const torch::Tensor& bvh_aabbs, const torch::Tensor& bvh_attribs,
                                    const BvhMhemFitConfig& config) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (grad.is_cuda()) {
        assert (device_of(grad).has_value());
        device_guard.set_device(device_of(grad).value());
    }
    return bvh_mhem_fit::backward_impl(grad, bvh_mhem_fit::ForwardOutput{fitting_mixture, target_mixture, bvh_mixture_inversed, bvh_nodes, bvh_aabbs, bvh_attribs}, config);
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cuda_bvh_forward, "cuda_bvh_forward (CUDA)");
  m.def("backward", &cuda_bvh_backward, "cuda_bvh_backward (CUDA)");
}
#endif
