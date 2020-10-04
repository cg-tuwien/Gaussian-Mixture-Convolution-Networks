#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// I guess(!) we can't use a single implementatino file because the cu file doesn't like pybind11
// I'm really guessing here, didn't test anything, just copied the example from the docs.
// Ye, and I believe that the cuda compiler also doesn't like <torch/extension.h> (depending on the version of pytorch / pybind / cuda / gcc)
// CUDA forward declarations
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cuda_bvh_forward_impl(const torch::Tensor& mixture, const torch::Tensor& xes);
std::tuple<torch::Tensor, torch::Tensor> cuda_bvh_backward_impl(const torch::Tensor& grad_output,
                                                                const torch::Tensor& mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& aabbs,
                                                                const torch::Tensor& xes,
                                                                bool requires_grad_mixture, bool requires_grad_xes);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cuda_bvh_forward(const torch::Tensor& mixture, const torch::Tensor& xes) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(mixture));
    return cuda_bvh_forward_impl(mixture, xes);
}


std::tuple<torch::Tensor, torch::Tensor> cuda_bvh_backward(const torch::Tensor& grad_output,
                                                           const torch::Tensor& mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& aabbs,
                                                           const torch::Tensor& xes,
                                                           bool requires_grad_mixture, bool requires_grad_xes) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(mixture));
    return cuda_bvh_backward_impl(grad_output, mixture, bvh_nodes, aabbs, xes, requires_grad_mixture, requires_grad_xes);
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cuda_bvh_forward, "cuda_bvh_forward (CUDA)");
  m.def("backward", &cuda_bvh_backward, "cuda_bvh_backward (CUDA)");
}
#endif
