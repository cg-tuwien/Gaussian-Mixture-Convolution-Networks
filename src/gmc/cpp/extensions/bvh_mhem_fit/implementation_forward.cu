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

using node_index_torch_t = lbvh::detail::Node::index_type_torch;

template <typename scalar_t, int N_DIMS, int N_GAUSSIANS_TARGET>
__host__ __device__ void fit_reduce_node(lbvh::detail::Node::index_type id) {

}

template <typename scalar_t, int N_DIMS, int N_GAUSSIANS_TARGET>
__host__ __device__ void iterate_over_nodes(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                            const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                            const gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                                            const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                            const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                            gpe::PackedTensorAccessor32<int, 2> flags,
                                            gpe::PackedTensorAccessor32<node_index_torch_t, 3> tmp_g_container_a,
                                            const gpe::MixtureNs n, const unsigned n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes)
{
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Lbvh = lbvh::detail::basic_device_bvh<scalar_t, G, true>;

    const auto node_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x + n_internal_nodes;
    const auto mixture_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
    if (mixture_id >= n_mixtures || node_id >= n_nodes)
        return;

    // collect Gs int tmp_g_container until N_GAUSSIANS_TARGET * 2 is reached
    // then merge and cont


    const auto* node = &reinterpret_cast<const lbvh::detail::Node&>(nodes[int(mixture_id)][int(node_id)][0]);
    while(node->parent_idx != lbvh::detail::Node::index_type(0xFFFFFFFF)) // means idx == 0
    {
        auto* flag = &reinterpret_cast<int&>(flags[mixture_id][node->parent_idx]);
        const int old = gpe::atomicCAS(flag, 0, 1);
        if(old == 0)
        {
            // this is the first thread entered here.
            // wait the other thread from the other child node.
            return;
        }
        assert(old == 1);
        // here, the flag has already been 1. it means that this
        // thread is the 2nd thread. merge AABB of both childlen.


        auto& current_aabb = reinterpret_cast<const lbvh::Aabb<scalar_t>&>(aabbs[mixture_id][node->parent_idx][0]);
        node = &reinterpret_cast<const lbvh::detail::Node&>(nodes[mixture_id][node->parent_idx][0]);
        const auto& left_aabb = reinterpret_cast<const lbvh::Aabb<scalar_t>&>(aabbs[mixture_id][node->left_idx][0]);
        const auto& right_aabb = reinterpret_cast<const lbvh::Aabb<scalar_t>&>(aabbs[mixture_id][node->right_idx][0]);
    }
}

} // anonymous namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(const at::Tensor& mixture, int n_components_target) {
    using namespace torch::indexing;
    using LBVH = lbvh::Bvh<2, float>;

    constexpr int N_GAUSSIANS_TARGET = 4;

    // todo: flatten mixture for kernel, i.g. nbatch/nlayers/ncomponents/7 => nmixture/ncomponents/7

    auto n = gpe::get_ns(mixture);
    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor")
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.dims == 2, "atm only 2d gaussians")
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<float>(), "atm only float")

    auto bvh = LBVH(mixture);
    auto scratch_mixture = mixture.clone();
    auto n_mixtures = n.batch * n.layers;
    auto n_internal_nodes = bvh.m_n_internal_nodes;
    auto n_nodes = bvh.m_n_nodes;
    auto flag_container = torch::zeros({n_mixtures, n_internal_nodes}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Int));
    auto tmp_g_container = torch::zeros({n_mixtures, n_nodes, N_GAUSSIANS_TARGET},
                                        torch::TensorOptions(mixture.device()).dtype(lbvh::detail::TorchTypeMapper<node_index_torch_t>::id()));
    auto flags_a = gpe::accessor<int, 2>(flag_container);
    auto tmp_g_container_a = gpe::accessor<node_index_torch_t, 3>(tmp_g_container);

    dim3 dimBlock = dim3(32, 1, 1);
    dim3 dimGrid = dim3((uint(bvh.m_n_leaf_nodes) + dimBlock.x - 1) / dimBlock.x,
                        (uint(n.layers) + dimBlock.y - 1) / dimBlock.y,
                        (uint(n.batch) + dimBlock.z - 1) / dimBlock.z);

    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_bvh_backward_impl", ([&] {
                                   auto mixture_a = gpe::accessor<scalar_t, 3>(scratch_mixture);
                                   auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(bvh.m_nodes);
                                   auto aabbs_a = gpe::accessor<scalar_t, 3>(bvh.m_aabbs);

                                   if (n.dims == 2) {
                                       auto fun = [mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a, n, n_mixtures, n_internal_nodes, n_nodes] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               iterate_over_nodes<scalar_t, 2, 4>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                               mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a,
                                                                               n, n_mixtures, n_internal_nodes, n_nodes);
                                           };
//                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a, n, n_mixtures, n_internal_nodes, n_nodes] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               iterate_over_nodes<scalar_t, 3, 4>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                               mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a,
                                                                               n, n_mixtures, n_internal_nodes, n_nodes);
                                           };
//                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                               }));

    GPE_CUDA_ASSERT(cudaPeekAtLastError())
    GPE_CUDA_ASSERT(cudaDeviceSynchronize())

//    return std::make_tuple(sum, bvh.m_nodes, bvh.m_aabbs);
    //todo: return something useful
    return {};
}


} // namespace bvh_mhem_fit
