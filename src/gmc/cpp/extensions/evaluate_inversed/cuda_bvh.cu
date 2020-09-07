//#include <torch/extension.h>
#include <torch/script.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <algorithm>

#include "common.h"

#ifndef __CUDACC__
constexpr dim3 blockIdx;
constexpr dim3 blockDim;
constexpr dim3 threadIdx;
using std::min;
using std::max;
#endif


torch::Tensor cuda_bvh_forward_impl(torch::Tensor mixture, torch::Tensor xes) {
    using namespace torch::indexing;
    auto n = gm::check_input_and_get_ns(mixture, xes);

    torch::Tensor sum = torch::zeros({n.batch, n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));

    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor");
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA");
    TORCH_CHECK(n.xes < 65535, "number of xes must be smaller than 65535 for CUDA");

    for (uint batch_id = 0; batch_id < n.batch; ++batch_id) {
        for (uint layer_id = 0; layer_id < n.layers; ++layer_id) {
            torch::Tensor current_mixture = mixture.index({batch_id, layer_id});
            TORCH_CHECK(current_mixture.is_contiguous(), "mixtures must be contiguous");

        }
    }


    return sum;
}

//std::vector<torch::Tensor> cuda_parallel_backward_impl(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes) {
//    gm::check_mixture(mixture);
//    auto n = gm::check_input_and_get_ns(mixture, xes);

//    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor")
//    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor");
//    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions");
//    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension");
//    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension");
//    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension");
//    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")


//    torch::Tensor grad_mixture = torch::zeros({n.batch, n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
//    torch::Tensor grad_xes = torch::zeros({n.batch_xes, n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

//    dim3 dimBlock = dim3(128);
//    const dim3 dimGrid = dim3(n.batch * n.layers,
//                              n.xes,
//                              (n.components + dimBlock.z - 1) / dimBlock.z);
////    std::cout << "forward: dimBlock=" << dimBlock.x << "/" << dimBlock.y << "/" << dimBlock.z << ", dimGrid=" << dimGrid.x << "/" << dimGrid.y << "/" << dimGrid.z << std::endl;

//    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "eval_inversed_omp_backward", ([&] {
//        auto mixture_a = mixture.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto xes_a = xes.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_mixture_a = grad_mixture.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_xes_a = grad_xes.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>();
//        auto grad_output_a = grad_output.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();

//        if (n.dims == 2)
//            kernel_backward<scalar_t, 2><<<dimGrid, dimBlock>>>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
//        else
//            kernel_backward<scalar_t, 3><<<dimGrid, dimBlock>>>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
//    }));

//    return {grad_mixture, grad_xes};
//}
