#include "bvh_mhem_fit/implementation.h"
#include <algorithm>
#include <chrono>
#include <vector>
#include <stdio.h>

#include <cuda.h>
//#include <
#include <cuda_runtime.h>
#include <glm/matrix.hpp>
#include <torch/types.h>

#include "bvh_mhem_fit/implementation_common.cuh"
#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
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
using node_index_t = lbvh::detail::Node::index_type;


template <int REDUCTION_N>
__host__ __device__ int copy_gaussian_ids(gpe::Accessor32<node_index_torch_t, 1> tmp_g_container_source, node_index_torch_t* destination) {
    for (int i = 0; i < REDUCTION_N; ++i) {
        destination[i] = tmp_g_container_source[i];
        if (tmp_g_container_source[i] == -1)
            return i;
    }
    return REDUCTION_N;
}


template <int REDUCTION_N>
__host__ __device__ int collect_child_gaussian_ids(const lbvh::detail::Node* node,
                                                   gpe::Accessor32<node_index_torch_t, 2> tmp_g_container_a,
                                                   node_index_torch_t* destination) {
    auto n_copied = copy_gaussian_ids<REDUCTION_N>(tmp_g_container_a[node->left_idx], destination);
    destination += n_copied;
    n_copied += copy_gaussian_ids<REDUCTION_N>(tmp_g_container_a[node->right_idx], destination);
    return n_copied;
}


template <typename scalar_t, int N_DIMS, int REDUCTION_N>
__host__ __device__ void fit_reduce_node(const lbvh::detail::Node* node,
                                         gpe::Accessor32<node_index_torch_t, 2> tmp_g_container_a,
                                         gpe::Accessor32<scalar_t, 2> mixture_a) {
    // for now (testing) simply select N_GAUSSIANS_TARGET strongest gaussians
    // no stl available in cuda 10.1.
    node_index_torch_t gaussian_ids[REDUCTION_N * 2];
    scalar_t gaussian_v[REDUCTION_N * 2];
    const auto n_input_gaussians = collect_child_gaussian_ids<REDUCTION_N>(node, tmp_g_container_a, gaussian_ids);
    int largest_index = -1;
    scalar_t largest_value = 0;
    for (int i = 0; i < n_input_gaussians; ++i) {
        gaussian_v[i] = gpe::gaussian<N_DIMS>(mixture_a[gaussian_ids[i]]).weight;
        if (gaussian_v[i] > largest_value) {
            largest_value = gaussian_v[i];
            largest_index = i;
        }
    }

    assert(largest_index != -1);
    int currently_largest_i = 0;
    tmp_g_container_a[node->object_idx][currently_largest_i] = node_index_torch_t(largest_index);
    for (int i = 1; i < REDUCTION_N; ++i) {
        scalar_t currently_largest_v = 0;
        currently_largest_i = -1;
        for (int j = 0; j < n_input_gaussians; ++j) {
            auto v = gaussian_v[j];
            if (v > currently_largest_v && (v < largest_value || (v == largest_value && j > largest_index))) {
                currently_largest_v = v;
                currently_largest_i = j;
            }
        }

        tmp_g_container_a[node->object_idx][i] = node_index_torch_t(currently_largest_i);
        largest_value = currently_largest_v;
        largest_index = currently_largest_i;
    }
}


template <int REDUCTION_N>
__host__ __device__ int count_gaussians(const lbvh::detail::Node* node,
                                        gpe::Accessor32<node_index_torch_t, 2> tmp_g_container_a) {
    auto c = tmp_g_container_a[node->object_idx];
    for (int i = 0; i < REDUCTION_N; ++i) {
        if (c[i] == -1)
            return i;
    }
    return REDUCTION_N;
}


template <int N_GAUSSIANS_TARGET>
__host__ __device__ int count_child_gaussians(const lbvh::detail::Node* node,
                                               const gpe::Accessor32<node_index_torch_t, 2>& nodes_a,
                                               gpe::Accessor32<node_index_torch_t, 2> tmp_g_container_a) {
    return count_gaussians<N_GAUSSIANS_TARGET>(reinterpret_cast<const lbvh::detail::Node*>(&nodes_a[node->left_idx][0]), tmp_g_container_a)
           + count_gaussians<N_GAUSSIANS_TARGET>(reinterpret_cast<const lbvh::detail::Node*>(&nodes_a[node->right_idx][0]), tmp_g_container_a);
}


template <typename scalar_t, int N_DIMS, int REDUCTION_N>
__host__ __device__ void iterate_over_nodes(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                            const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                            gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                                            const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                            const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                            gpe::PackedTensorAccessor32<int, 2> flags,
                                            gpe::PackedTensorAccessor32<node_index_torch_t, 3> tmp_g_container_a,
                                            const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes, int n_components_target)
{
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Lbvh = lbvh::detail::basic_device_bvh<scalar_t, G, true>;

    auto node_id = node_index_t(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x + n_internal_nodes);
    const auto mixture_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
    if (mixture_id >= n_mixtures || node_id >= n_nodes)
        return;

    // collect Gs int tmp_g_container until N_GAUSSIANS_TARGET * 2 is reached
    // then merge and cont


    const auto* node = reinterpret_cast<const lbvh::detail::Node*>(&nodes[mixture_id][int(node_id)][0]);
    tmp_g_container_a[mixture_id][node_id][0] = node_index_torch_t(node->object_idx);
    while(node->parent_idx != node_index_t(0xFFFFFFFF)) // means idx == 0
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

        node_id = node->parent_idx;
        node = reinterpret_cast<const lbvh::detail::Node*>(&nodes[mixture_id][node_id][0]);
        if (count_child_gaussians<REDUCTION_N>(node, nodes[mixture_id], tmp_g_container_a[mixture_id]) > REDUCTION_N) {
            fit_reduce_node<scalar_t, N_DIMS, REDUCTION_N>(node, tmp_g_container_a[mixture_id], mixture[mixture_id]);
        }
        else {
            auto* destination = &tmp_g_container_a[mixture_id][node->object_idx][0];
            collect_child_gaussian_ids<REDUCTION_N>(node, tmp_g_container_a[mixture_id], destination);
        }
    }
}

template <typename scalar_t, int N_DIMS, int REDUCTION_N>
__host__ __device__ void collect_result(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                            const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                            const gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                                            gpe::PackedTensorAccessor32<scalar_t, 3> out_mixture,
                                            const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                            const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                            gpe::PackedTensorAccessor32<int, 2> flags,
                                            gpe::PackedTensorAccessor32<node_index_torch_t, 3> tmp_g_container_a,
                                            const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes, int n_components_target)
{
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Lbvh = lbvh::detail::basic_device_bvh<scalar_t, G, true>;

    auto target_component_id = node_index_t(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x + n_internal_nodes);
    const auto mixture_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
    if (mixture_id >= n_mixtures || target_component_id >= n_components_target)
        return;

    // collect merged Gs
    auto n_levels_down = 32 - gpe::clz(uint32_t(n_components_target / REDUCTION_N));

    const auto* node = reinterpret_cast<const lbvh::detail::Node*>(&nodes[mixture_id][int(0)][0]);

    //todo cathch: that'll fail if the tree is very unbalanced -> abort and pad gaussians with zeroes;
    for (int i = 0; i < n_levels_down; ++i) {
        auto decision_bit = 1 << (n_levels_down - i);
        if (target_component_id & decision_bit)
            node = node = reinterpret_cast<const lbvh::detail::Node*>(&nodes[mixture_id][node->left_idx][0]);
        else
            node = node = reinterpret_cast<const lbvh::detail::Node*>(&nodes[mixture_id][node->right_idx][0]);
    }
}


} // anonymous namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(at::Tensor mixture, int n_components_target = 32) {
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

    const auto n_mixtures = n.batch * n.layers;
    const auto bvh = LBVH(mixture);
    mixture = mixture.view({n_mixtures, n.components, -1});
    auto scratch_mixture = mixture.clone();
    const auto n_internal_nodes = bvh.m_n_internal_nodes;
    const auto n_nodes = bvh.m_n_nodes;
    auto flag_container = torch::zeros({n_mixtures, n_internal_nodes}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Int));
    auto tmp_g_container = -1 * torch::ones({n_mixtures, n_nodes, N_GAUSSIANS_TARGET},
                                            torch::TensorOptions(mixture.device()).dtype(lbvh::detail::TorchTypeMapper<node_index_torch_t>::id()));
    auto flags_a = gpe::accessor<int, 2>(flag_container);
    auto tmp_g_container_a = gpe::accessor<node_index_torch_t, 3>(tmp_g_container);


    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(mixture.scalar_type(), n.dims, ([&] {
                                    dim3 dimBlock = dim3(32, 1, 1);
                                    dim3 dimGrid = dim3((uint(bvh.m_n_leaf_nodes) + dimBlock.x - 1) / dimBlock.x,
                                                        (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
                                                        (uint(1) + dimBlock.z - 1) / dimBlock.z);

                                   auto mixture_a = gpe::accessor<scalar_t, 3>(scratch_mixture);
                                   auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(bvh.m_nodes);
                                   auto aabbs_a = gpe::accessor<scalar_t, 3>(bvh.m_aabbs);

                                   auto fun = [mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a, n, n_mixtures, n_internal_nodes, n_nodes, n_components_target] __host__ __device__
                                       (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                           iterate_over_nodes<scalar_t, N_DIMS, 4>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                                   mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a,
                                                                                   n, n_mixtures, n_internal_nodes, n_nodes, n_components_target);
                                       };
                                   gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   GPE_CUDA_ASSERT(cudaPeekAtLastError())
                                   GPE_CUDA_ASSERT(cudaDeviceSynchronize())
                               }));


    auto out_mixture = torch::zeros({n_mixtures, n.components, mixture.size(-1)}, torch::TensorOptions(mixture.device()).dtype(mixture.dtype()));
    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(mixture.scalar_type(), n.dims, ([&] {
                                            dim3 dimBlock = dim3(32, 1, 1);
                                            dim3 dimGrid = dim3((uint(n_components_target) + dimBlock.x - 1) / dimBlock.x,
                                                                (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
                                                                (uint(1) + dimBlock.z - 1) / dimBlock.z);

                                            auto mixture_a = gpe::accessor<scalar_t, 3>(scratch_mixture);
                                            auto out_mixture_a = gpe::accessor<scalar_t, 3>(out_mixture);
                                            auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(bvh.m_nodes);
                                            auto aabbs_a = gpe::accessor<scalar_t, 3>(bvh.m_aabbs);

                                            auto fun = [mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a, n, n_mixtures, n_internal_nodes, n_nodes, n_components_target]
                                                __host__ __device__
                                                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                                    collect_result<scalar_t, N_DIMS, 4>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                                        mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a,
                                                                                        n, n_mixtures, n_internal_nodes, n_nodes, n_components_target);
                                                };
                                            gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                        }));

    GPE_CUDA_ASSERT(cudaPeekAtLastError())
    GPE_CUDA_ASSERT(cudaDeviceSynchronize())

    return std::make_tuple(out_mixture.view({n.batch, n.layers, n.components, -1}), bvh.m_nodes, bvh.m_aabbs);
}


} // namespace bvh_mhem_fit

