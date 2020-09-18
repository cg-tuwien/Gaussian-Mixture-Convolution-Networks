#include <algorithm>
#include <chrono>
#include <vector>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <torch/script.h>

#include <glm/glm.hpp>

#include "common.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "lbvh/query.h"
#include "lbvh/predicator.h"
#include "math/symeig_cuda.h"

#ifndef __CUDACC__
constexpr dim3 blockIdx;
constexpr dim3 blockDim;
constexpr dim3 threadIdx;
using std::min;
using std::max;

namespace torch {
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};
}

#endif

template<int N_DIMS, typename scalar_t>
std::ostream& operator <<(std::ostream& stream, const Gaussian<N_DIMS, scalar_t>& g) {
    stream << "Gauss[" << g.weight << "; " << g.position[0];
    for (int i = 1; i < N_DIMS; i++)
        stream << "/" << g.position[i];
    stream << "; ";

    for (int i = 0; i < N_DIMS; i++) {
        for (int j = 0; j < N_DIMS; j++) {
            if (i != 0 || j != 0)
                stream << "/";
            stream << g.covariance[i][j];
        }
    }
    stream << "]";
    return stream;
}

template <typename scalar_t, int DIMS>
__global__ void evaluate_inversed_bvh(const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> mixture,
                                      const torch::PackedTensorAccessor32<int16_t, 4, torch::RestrictPtrTraits> nodes,
                                      const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> aabbs,
                                      const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> xes,
                                      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> sums,
                                      const gpe::MixtureAndXesNs n)
{
    using G = Gaussian<DIMS, scalar_t>;
    const auto batch_index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto layer_index = blockIdx.y * blockDim.y + threadIdx.y;
    const auto xes_index = blockIdx.z * blockDim.z + threadIdx.z;

    const auto batch_xes_index = min(batch_index, n.batch_xes - 1);
    const auto layer_xes_index = min(layer_index, n.layers_xes - 1);

//    printf("batch_index=%d, layer_index=%d, batch_xes_index=%d, layer_xes_index=%d, xes_index=%d\n", batch_index, layer_index, batch_xes_index, layer_xes_index, xes_index);
    if (batch_index >= n.batch || layer_index >= n.layers || xes_index >= n.xes)
        return;
//    printf("do batch_index=%d, layer_index=%d, batch_xes_index=%d, layer_xes_index=%d, xes_index=%d\n", batch_index, layer_index, batch_xes_index, layer_xes_index, xes_index);


    unsigned int num_nodes = n.components * 2 + 1;  // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_objects = n.components;        // (# of leaves), the same as the number of objects
    const auto* bvh_nodes = &reinterpret_cast<const lbvh::detail::Node&>(nodes[batch_index][layer_index][0][0]);
    const auto* bvh_aabbs = &reinterpret_cast<const lbvh::Aabb<scalar_t>&>(aabbs[batch_index][layer_index][0][0]);
    const auto* bvh_gaussians = &reinterpret_cast<const G&>(mixture[batch_index][layer_index][0][0]);
    lbvh::detail::basic_device_bvh<scalar_t, G, true> bvh {num_nodes, num_objects, bvh_nodes, bvh_aabbs, bvh_gaussians};

    const auto& x_pos = gpe::vec<2>(xes[batch_xes_index][layer_xes_index][xes_index][0]);
    auto point = float4{x_pos.x, x_pos.y, 0, 0};
    auto& sum = sums[batch_index][layer_index][xes_index];
    auto evaluate = [bvh, &sum, &x_pos] (unsigned index) {
        const auto& g = bvh.objects[index];
        sum += gpe::evaluate_gaussian(x_pos, g.weight, g.position, g.covariance);
    };
    lbvh::query_device_with_fun(bvh, lbvh::inside_aabb(point), evaluate);
}

torch::Tensor cuda_bvh_forward_impl(const at::Tensor& mixture, const at::Tensor& xes) {
    using namespace torch::indexing;
    using LBVH = lbvh::bvh<float, Gaussian<2, float>>;

    auto n = gpe::check_input_and_get_ns(mixture, xes);

    torch::Tensor sum = torch::zeros({n.batch, n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));
    auto sum_a = sum.packed_accessor32<float, 3, torch::RestrictPtrTraits>();

    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor");
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA");
    TORCH_CHECK(n.xes < 65535, "number of xes must be smaller than 65535 for CUDA");
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation");

    TORCH_CHECK(n.dims == 2, "atm only 2d gaussians");
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<float>(), "atm only float");

    auto bvh = LBVH(mixture);
    auto mixture_a = mixture.packed_accessor32<float, 4, torch::RestrictPtrTraits>();
    auto nodes_a = bvh.m_nodes.packed_accessor32<int16_t, 4, torch::RestrictPtrTraits>();
    auto aabbs_a = bvh.m_aabbs.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    // mixture(batch, layer, component, data)
    // xes(batch, layer, n, data)

    auto xes_copy = xes;
    if (n.xes == n.components && n.batch == n.batch_xes && n.layers == n.layers_xes) {
        auto indices = bvh.m_nodes.index({Slice(), Slice(), Slice(bvh.m_n_internal_nodes, None), 3}).to(torch::ScalarType::Long);
        indices = indices.view({n.batch, n.layers, n.components, 1}).expand_as(xes);
//        std::cout << "indices.sizes() = " << indices.sizes() << std::endl;
        xes_copy = torch::gather(xes, 2, indices);
//        std::cout << "xes_copy.sizes() = " << xes_copy.sizes() << std::endl;
    }
//    std::cout << "xes: " << xes << std::endl;
//    std::cout << "xes_copy: " << xes_copy << std::endl;
//    std::cout << "gpe::positions(mixture): " << gpe::positions(mixture) << std::endl;
    const auto xes_a = xes_copy.packed_accessor32<float, 4, torch::RestrictPtrTraits>();

    dim3 dimBlock = dim3(1, 1, LBVH_N_QUERY_THREADS);
    dim3 dimGrid = dim3((n.batch + dimBlock.x - 1) / dimBlock.x,
                        (n.layers + dimBlock.y - 1) / dimBlock.y,
                        (n.xes + dimBlock.z - 1) / dimBlock.z);
//    printf("dimBlock=(%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
//    printf("dimGrid=(%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);
    auto start = std::chrono::system_clock::now();
    evaluate_inversed_bvh<float, 2><<<dimGrid, dimBlock>>>(mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "evaluate_inversed_bvh elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    return sum;
}

//std::vector<torch::Tensor> cuda_parallel_backward_impl(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes) {
//    gpe::check_mixture(mixture);
//    auto n = gpe::check_input_and_get_ns(mixture, xes);

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
