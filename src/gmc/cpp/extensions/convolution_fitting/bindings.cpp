#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "convolution_fitting/bindings.h"
#include "convolution_fitting/implementation.h"
#include "convolution_fitting/Config.h"

std::vector<torch::Tensor> convolution_fitting_forward(torch::Tensor data, torch::Tensor kernels, int n_components_fitting) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (data.is_cuda()) {
        assert (device_of(data).has_value());
        device_guard.set_device(device_of(data).value());
    }
    convolution_fitting::Config config = {};
    config.n_components_fitting = unsigned(n_components_fitting);
    auto result = convolution_fitting::forward_impl(data, kernels, config);
    return {result.fitting, result.cached_pos_covs, result.nodesobjs, result.fitting_subtrees};
}

std::pair<torch::Tensor, torch::Tensor> convolution_fitting_backward(const torch::Tensor& grad,
                                                                     const torch::Tensor& data, const torch::Tensor& kernels, int n_components_fitting,
                                                                     const torch::Tensor& fitting, const torch::Tensor& cached_pos_covs, const torch::Tensor& nodeobjs, const torch::Tensor& fitting_subtrees) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (grad.is_cuda()) {
        assert (device_of(grad).has_value());
        device_guard.set_device(device_of(grad).value());
    }

    convolution_fitting::Config config = {};
    config.n_components_fitting = unsigned(n_components_fitting);
    return convolution_fitting::backward_impl(grad, data, kernels, convolution_fitting::ForwardOutput{fitting, cached_pos_covs, nodeobjs, fitting_subtrees}, config);
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &convolution_fitting_forward, "convolution_fitting_forward");
  m.def("backward", &convolution_fitting_backward, "convolution_fitting_backward");
}
#endif
