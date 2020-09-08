#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// I guess(!) we can't use a single implementatino file because the cu file doesn't like pybind11
// I'm really guessing here, didn't test anything, just copied the example from the docs.
// Ye, and I believe that the cuda compiler also doesn't like <torch/extension.h> (depending on the version of pytorch / pybind / cuda / gcc)
// CUDA forward declarations
//torch::Tensor cuda_parallel_forward_impl(torch::Tensor mixture, torch::Tensor xes);
//std::vector<torch::Tensor> cuda_parallel_backward_impl(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes);


//torch::Tensor cuda_parallel_forward(torch::Tensor mixture, torch::Tensor xes) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(mixture));
//    return cuda_parallel_forward_impl(mixture, xes);
//}


//std::vector<torch::Tensor> cuda_parallel_backward(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes) {
//    const at::cuda::OptionalCUDAGuard device_guard(device_of(mixture));
//    return cuda_parallel_backward_impl(grad_output, mixture, xes, requires_grad_mixture, requires_grad_xes);
//}

//#ifndef GMC_CMAKE_TEST_BUILD
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("forward", &cuda_parallel_forward, "evaluate_inversed forward (CUDA)");
//  m.def("backward", &cuda_parallel_backward, "evaluate_inversed backward (CUDA)");
//}
//#endif
