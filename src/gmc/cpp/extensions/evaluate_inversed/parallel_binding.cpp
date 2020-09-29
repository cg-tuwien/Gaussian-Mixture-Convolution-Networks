#include "parallel_binding.h"

#include <cassert>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "parallel_implementation.h"

// We can't use a single implementatino file because the cuda compiler doesn't like pybind11 (i guess)
// Ye, and it certainly doesn't like <torch/extension.h> (depending on the version of pytorch / pybind / cuda / gcc)

torch::Tensor parallel_forward(const torch::Tensor& mixture, const torch::Tensor& xes) {
//    auto guard = gpe::make_device_guard(mixture);
    at::cuda::OptionalCUDAGuard device_guard;
    if (mixture.is_cuda()) {
        assert (device_of(mixture).has_value());
        device_guard.set_device(device_of(mixture).value());
    }
    return parallel_forward_impl(mixture, xes);
}

std::tuple<torch::Tensor, torch::Tensor> parallel_backward(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes,
                                                           bool requires_grad_mixture, bool requires_grad_xes) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (mixture.is_cuda()) {
        assert (device_of(mixture).has_value());
        device_guard.set_device(device_of(mixture).value());
    }
    return parallel_backward_impl(grad_output, mixture, xes, requires_grad_mixture, requires_grad_xes);
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &parallel_forward, "evaluate_inversed parallel forward (CPU and CUDA))");
    m.def("backward", &parallel_backward, "evaluate_inversed parallel backward (CPU and CUDA)");
}
#endif
