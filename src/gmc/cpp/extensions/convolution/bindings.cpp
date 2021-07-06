#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "convolution/bindings.h"
#include "convolution/implementation.h"

torch::Tensor convolution_forward(torch::Tensor data, torch::Tensor kernels) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (data.is_cuda()) {
        assert (device_of(data).has_value());
        device_guard.set_device(device_of(data).value());
    }
    return convolution::forward_impl(data, kernels);
}

std::pair<torch::Tensor, torch::Tensor> convolution_backward(at::Tensor grad, at::Tensor data, at::Tensor kernels) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (grad.is_cuda()) {
        assert (device_of(grad).has_value());
        device_guard.set_device(device_of(grad).value());
    }

    return convolution::backward_impl(grad, data, kernels);
    return {};
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &convolution_forward, "convolution_forward");
//  m.def("backward", &convolution_backward, "convolution_backward");
}
#endif
