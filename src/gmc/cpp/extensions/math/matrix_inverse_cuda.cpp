#include "math/symeig_cuda.h"
#include <vector>
#include <algorithm>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>


// I guess(!) we can't use a single implementatino file because the cu file doesn't like pybind11
// I'm really guessing here, didn't test anything, just copied the example from the docs.
// Ye, and I believe that the cuda compiler also doesn't like <torch/extension.h> (depending on the version of pytorch / pybind / cuda / gcc)
// CUDA forward declarations
torch::Tensor matrix_inverse_cuda_forward_impl(const torch::Tensor& matrices);
// torch::Tensor matrix_inverse_cuda_backward_impl(const torch::Tensor& grad_output, const torch::Tensor& matrices);

namespace gpe {
torch::Tensor matrix_inverse_cuda_forward(const torch::Tensor& matrices) {
    const at::cuda::OptionalCUDAGuard device_guard(device_of(matrices));
    return matrix_inverse_cuda_forward_impl(matrices);
}
// torch::Tensor matrix_inverse_cuda_backward(const torch::Tensor& grad_output, const torch::Tensor& inversed_matrices) {
//     const at::cuda::OptionalCUDAGuard device_guard(device_of(inversed_matrices));
//     return -inversed_matrices.matmul(grad_output).matmul(inversed_matrices);
//     return matrix_inverse_cuda_backward_impl(grad_output, inversed_matrices);
// }
}

#ifndef GMC_CMAKE_TEST_BUILD
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &gpe::matrix_inverse_cuda_forward, "matrix_inverse forward (CUDA)");
//   m.def("backward", &gpe::matrix_inverse_cuda_backward, "evaluate_inversed backward (CUDA)");
}
#endif
