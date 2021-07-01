#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "convolution/bindings.h"
#include "convolution/implementation.h"

std::vector<torch::Tensor> convolution_forward(torch::Tensor data, torch::Tensor kernels) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (data.is_cuda()) {
        assert (device_of(data).has_value());
        device_guard.set_device(device_of(data).value());
    }
    auto result = convolution::forward_impl(data, kernels);
    return {result.mixture};
}

//std::pair<torch::Tensor, torch::Tensor> convolution_backward(const torch::Tensor& grad,
//                                    const torch::Tensor& fitting_mixture,
//                                    const torch::Tensor& target_mixture,
//                                    const torch::Tensor& bvh_nodes, const torch::Tensor& bvh_attribs,
//                                    int n_components_fitting, int reduction_n) {
//    at::cuda::OptionalCUDAGuard device_guard;
//    if (grad.is_cuda()) {
//        assert (device_of(grad).has_value());
//        device_guard.set_device(device_of(grad).value());
//    }

//    convolution::Config config = {};
//    config.n_components_fitting = unsigned(n_components_fitting);
//    config.reduction_n = reduction_n;

////    return convolution::backward_impl(grad, convolution::ForwardOutput{fitting_mixture, target_mixture, bvh_nodes, bvh_attribs}, config);
//    return {};
//}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &convolution_forward, "convolution_forward");
//  m.def("backward", &convolution_backward, "convolution_backward");
}
#endif
