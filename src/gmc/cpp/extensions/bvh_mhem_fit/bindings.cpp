#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "bindings.h"
#include "implementation.h"
#include "bvh_mhem_fit/Config.h"

std::vector<torch::Tensor> bvh_mhem_fit_forward(const torch::Tensor& mixture, int n_components_fitting, int reduction_n) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (mixture.is_cuda()) {
        assert (device_of(mixture).has_value());
        device_guard.set_device(device_of(mixture).value());
    }
    bvh_mhem_fit::Config config = {};
    config.n_components_fitting = unsigned(n_components_fitting);
    config.reduction_n = reduction_n;
    auto result = bvh_mhem_fit::forward_impl(mixture, config);
    return {result.fitting, result.bvh_nodes, result.bvh_attributes};
}

torch::Tensor bvh_mhem_fit_backward(const torch::Tensor& grad,
                                    const torch::Tensor& fitting_mixture,
                                    const torch::Tensor& target_mixture,
                                    const torch::Tensor& bvh_nodes, const torch::Tensor& bvh_attribs,
                                    int n_components_fitting, int reduction_n) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (grad.is_cuda()) {
        assert (device_of(grad).has_value());
        device_guard.set_device(device_of(grad).value());
    }

    bvh_mhem_fit::Config config = {};
    config.n_components_fitting = unsigned(n_components_fitting);
    config.reduction_n = reduction_n;

    return bvh_mhem_fit::backward_impl(grad, bvh_mhem_fit::ForwardOutput{fitting_mixture, target_mixture, bvh_nodes, bvh_attribs}, config);
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &bvh_mhem_fit_forward, "bvh_mhem_fit_forward");
  m.def("backward", &bvh_mhem_fit_backward, "bvh_mhem_fit_backward");
}
#endif
