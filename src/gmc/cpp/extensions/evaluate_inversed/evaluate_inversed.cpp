#include "evaluate_inversed/evaluate_inversed.h"

#include <cassert>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "implementations.h"

namespace evaluate_inversed {
std::tuple<at::Tensor> parallel_forward(const torch::Tensor& mixture, const torch::Tensor& xes) {
//    auto guard = gpe::make_device_guard(mixture);
    at::cuda::OptionalCUDAGuard device_guard;
    if (mixture.is_cuda()) {
        assert (device_of(mixture).has_value());
        device_guard.set_device(device_of(mixture).value());
        return parallel_forward_optimised_impl(mixture, xes);
    }
    return {parallel_forward_impl(mixture, xes)};
}

std::tuple<torch::Tensor, torch::Tensor> parallel_backward(const torch::Tensor& grad_output,
                                                           const torch::Tensor& mixture,
                                                           const torch::Tensor& xes,
                                                           const std::tuple<torch::Tensor>&,
                                                           bool requires_grad_mixture, bool requires_grad_xes) {
    at::cuda::OptionalCUDAGuard device_guard;
    if (mixture.is_cuda()) {
        assert (device_of(mixture).has_value());
        device_guard.set_device(device_of(mixture).value());
        return parallel_backward_optimised_impl(grad_output, mixture, xes, requires_grad_mixture, requires_grad_xes);
    }
    return parallel_backward_impl(grad_output, mixture, xes, requires_grad_mixture, requires_grad_xes);
}

}
