#include "bvh_mhem_fit/implementation.h"
#include <algorithm>
#include <chrono>
#include <vector>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <glm/matrix.hpp>
#include <torch/types.h>

#include "bvh_mhem_fit/implementation_common.cuh"
#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "hacked_accessor.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "lbvh/query.h"
#include "lbvh/predicator.h"
#include "math/symeig_cuda.h"
#include "mixture.h"
#include "parallel_start.h"

namespace bvh_mhem_fit {

namespace  {

template <typename scalar_t, int DIMS>
__host__ __device__ void kernel(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                     const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                     const gpe::PackedTensorAccessor32<scalar_t, 4> mixture,
                                     const gpe::PackedTensorAccessor32<lbvh::detail::Node::index_type_torch, 4> nodes,
                                     const gpe::PackedTensorAccessor32<scalar_t, 4> aabbs,
                                     const gpe::PackedTensorAccessor32<scalar_t, 4> xes,
                                     gpe::PackedTensorAccessor32<scalar_t, 3> sums,
                                     const gpe::MixtureAndXesNs n)
{
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<DIMS, scalar_t>;
    using Lbvh = lbvh::detail::basic_device_bvh<scalar_t, G, true>;
    const auto xes_index = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
    const auto layer_index = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
    const auto batch_index = int(gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z);

    const auto batch_xes_index = gpe::min(batch_index, n.batch_xes - 1);
    const auto layer_xes_index = gpe::min(layer_index, n.layers_xes - 1);

    if (batch_index >= n.batch || layer_index >= n.layers || xes_index >= n.xes)
        return;

    const unsigned int num_nodes = uint(n.components) * 2 + 1;  // (# of internal node) + (# of leaves), 2N+1
    const unsigned int num_objects = uint(n.components);        // (# of leaves), the same as the number of objects
    const auto* bvh_nodes = &reinterpret_cast<const lbvh::detail::Node&>(nodes[batch_index][layer_index][0][0]);
    const auto* bvh_aabbs = &reinterpret_cast<const lbvh::Aabb<scalar_t>&>(aabbs[batch_index][layer_index][0][0]);
    const auto* bvh_gaussians = &reinterpret_cast<const G&>(mixture[batch_index][layer_index][0][0]);
    Lbvh bvh {num_nodes, num_objects, bvh_nodes, bvh_aabbs, bvh_gaussians};

    const auto& x_pos = gpe::vec<DIMS>(xes[batch_xes_index][layer_xes_index][xes_index][0]);
    auto point = lbvh::make_vector_of(x_pos);
    auto& sum = sums[batch_index][layer_index][xes_index];
    auto evaluate = [&bvh, &sum, &x_pos] (unsigned index) {
        const auto& g = bvh.objects[index];
        auto val = gpe::evaluate_gaussian(x_pos, g.weight, g.position, g.covariance);
        sum += val;
    };
    lbvh::query_device_with_fun(bvh, lbvh::inside_aabb(point), evaluate);
}

} // anonymous namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(const at::Tensor& mixture, const at::Tensor& xes) {
    using namespace torch::indexing;
    using LBVH = lbvh::Bvh<float, gpe::Gaussian<2, float>>;

    auto n = gpe::check_input_and_get_ns(mixture, xes);
    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor")
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.xes < 65535, "number of xes must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.dims == 2, "atm only 2d gaussians")
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<float>(), "atm only float")

    auto bvh = LBVH(mixture);
    torch::Tensor sum = torch::zeros({n.batch, n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));

    // mixture(batch, layer, component, data)
    // xes(batch, layer, n, data)

    auto xes_copy = xes;
    const auto use_indirect_xes = n.xes == n.components && n.batch == n.batch_xes && n.layers == n.layers_xes;
    if (use_indirect_xes) {
        auto indices = bvh.m_nodes.index({Slice(), Slice(), Slice(bvh.m_n_internal_nodes, None), 3}).to(torch::ScalarType::Long);
        indices = indices.view({n.batch, n.layers, n.components, 1}).expand_as(xes);
        xes_copy = torch::gather(xes, 2, indices);
    }

    dim3 dimBlock = dim3(LBVH_N_QUERY_THREADS, 1, 1);
    dim3 dimGrid = dim3((uint(n.xes) + dimBlock.x - 1) / dimBlock.x,
                        (uint(n.layers) + dimBlock.y - 1) / dimBlock.y,
                        (uint(n.batch) + dimBlock.z - 1) / dimBlock.z);

    auto sum_c = sum.cpu();
    auto mixture_c = mixture.cpu();
    auto bvh_nodes_c = bvh.m_nodes.cpu();
    auto bvh_aabbs_c = bvh.m_aabbs.cpu();
    xes_copy = xes_copy.cpu();

    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_bvh_backward_impl", ([&] {
//                                   auto sum_a = gpe::accessor<scalar_t, 3>(sum);
//                                   auto mixture_a = gpe::accessor<scalar_t, 4>(mixture);
//                                   auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 4>(bvh.m_nodes);
//                                   auto aabbs_a = gpe::accessor<scalar_t, 4>(bvh.m_aabbs);
                                   auto sum_a = gpe::accessor<scalar_t, 3>(sum_c);
                                   auto mixture_a = gpe::accessor<scalar_t, 4>(mixture_c);
                                   auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 4>(bvh_nodes_c);
                                   auto aabbs_a = gpe::accessor<scalar_t, 4>(bvh_aabbs_c);
                                   const auto xes_a = gpe::accessor<scalar_t, 4>(xes_copy);

                                   if (n.dims == 2) {
                                       auto fun = [mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               kernel<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n);
                                           };
//                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture_c), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               kernel<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n);
                                           };
//                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture_c), dimGrid, dimBlock, fun);
                                   }
                               }));

    sum = sum_c.cuda();
    if (use_indirect_xes) {
        auto indices = bvh.m_nodes.index({Slice(), Slice(), Slice(bvh.m_n_internal_nodes, None), 3}).to(torch::ScalarType::Long);
        indices = inverse_permutation(indices);
        sum = torch::gather(sum, 2, indices);
    }

    GPE_CUDA_ASSERT(cudaPeekAtLastError())
    GPE_CUDA_ASSERT(cudaDeviceSynchronize())

    return std::make_tuple(sum, bvh.m_nodes, bvh.m_aabbs);
}


} // namespace bvh_mhem_fit
