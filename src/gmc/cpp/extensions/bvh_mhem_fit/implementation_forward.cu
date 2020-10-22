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
#include "math/matrix.h"
#include "math/scalar.h"
#include "math/symeig_cuda.h"
#include "mixture.h"
#include "parallel_start.h"

#define EXECUTION_DEVICES __host__ __device__

namespace bvh_mhem_fit {

namespace  {

using node_index_torch_t = lbvh::detail::Node::index_type_torch;
using node_index_t = lbvh::detail::Node::index_type;
using gaussian_index_t = uint16_t;
using gaussian_index_torch_t = int16_t;



template <int REDUCTION_N>
EXECUTION_DEVICES int copy_gaussian_ids(gpe::Accessor32<node_index_torch_t, 1> tmp_g_container_source, gaussian_index_t* destination) {
    for (int i = 0; i < REDUCTION_N; ++i) {
        destination[i] = gaussian_index_t(tmp_g_container_source[i]);
        if (tmp_g_container_source[i] == -1)
            return i;
    }
    return REDUCTION_N;
}


template <int REDUCTION_N>
EXECUTION_DEVICES int collect_child_gaussian_ids(const lbvh::detail::Node* node,
                                                   gpe::Accessor32<node_index_torch_t, 2> tmp_g_container_a,
                                                   gaussian_index_t* destination) {
    auto n_copied = copy_gaussian_ids<REDUCTION_N>(tmp_g_container_a[node->left_idx], destination);
    destination += n_copied;
    n_copied += copy_gaussian_ids<REDUCTION_N>(tmp_g_container_a[node->right_idx], destination);
    return n_copied;
}


template <typename scalar_t, int N_DIMS, int REDUCTION_N>
EXECUTION_DEVICES void fit_reduce_node(const lbvh::detail::Node* node,
                                       gpe::Accessor32<node_index_torch_t, 2> tmp_g_container_a,
                                       gpe::Accessor32<scalar_t, 2> mixture_a,
                                       const gpe::MixtureNs& n) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    // for now (testing) simply select N_GAUSSIANS_TARGET strongest gaussians
    // no stl available in cuda 10.1.
    gaussian_index_t cached_gaussian_ids[REDUCTION_N * 2];
    const auto n_input_gaussians = collect_child_gaussian_ids<REDUCTION_N>(node, tmp_g_container_a, cached_gaussian_ids);

    // compute distance matrix
    float distances[REDUCTION_N * 2][REDUCTION_N * 2];
    for (int i = 0; i < n_input_gaussians; ++i) {
        const G& g_i = gpe::gaussian<N_DIMS>(mixture_a[int(cached_gaussian_ids[i])]);
        distances[i][i] = 0;
        for (int j = i + 1; j < n_input_gaussians; ++j) {
            const G& g_j = gpe::gaussian<N_DIMS>(mixture_a[int(cached_gaussian_ids[j])]);
            // todo: min of KL divergencies is probably a better distance
            float distance = gpe::sign(g_i.weight) == gpe::sign(g_j.weight) ? 0 : std::numeric_limits<scalar_t>::infinity();
            assert(std::isnan(distance) == false);
            distance += float(gpe::squared_norm(g_i.position - g_j.position));
            assert(std::isnan(distance) == false);
            distances[i][j] = distance;
            distances[j][i] = distance;
        }
    }

    using cache_index_t = uint16_t;

    cache_index_t subgraphs[REDUCTION_N * 2][REDUCTION_N * 2];
    for (int i = 0; i < REDUCTION_N * 2; ++i) {
        subgraphs[i][0] = cache_index_t(i);
        for (int j = 1; j < REDUCTION_N * 2; ++j) {
            subgraphs[i][j] = cache_index_t(-1);
        }
    }

    int n_subgraphs = n_input_gaussians;
    auto merge_subgraphs = [&](int a, int b) {
        assert (a != b);
        auto a_ = gpe::min(a, b);
        auto b_ = gpe::max(a, b);

        auto a_end = 1;
        while (subgraphs[a_][a_end] != cache_index_t(-1)) {
            assert(a_end < REDUCTION_N * 2);
            ++a_end;
        }
        auto b_current = 0;
        while (subgraphs[b_][b_current] != cache_index_t(-1)) {
            assert(b_current < REDUCTION_N * 2);
            subgraphs[a_][a_end] = subgraphs[b_][b_current];
            // todo: i think the following line can be removed in the final version
            subgraphs[b_][b_current] = cache_index_t(-1);
            ++a_end;
            assert(a_end <= REDUCTION_N * 2);
            ++b_current;
        }
        --n_subgraphs;
    };
    auto subgraph_of = [&](cache_index_t id) {
        for (int i = 0; i < REDUCTION_N * 2; ++i) {
            auto current = 0;
            while (subgraphs[i][current] != cache_index_t(-1)) {
                if (subgraphs[i][current] == id)
                    return i;
                ++current;
                assert(current < REDUCTION_N * 2);
            }
        }
        assert(false);
        return -1;
    };
    auto shortest_edge = [&](cache_index_t* a, cache_index_t* b) {
        *a = cache_index_t(-1);
        *b = cache_index_t(-1);
        float shortest_length = std::numeric_limits<float>::infinity();
        for (cache_index_t i = 0; i < n_input_gaussians; ++i) {
            for (cache_index_t j = i + 1; j < n_input_gaussians; ++j) {
                if (distances[i][j] < shortest_length) {
                    *a = i;
                    *b = j;
                    shortest_length = distances[i][j];
                }
            }
        }
        assert(*a != cache_index_t(-1));
        assert(*b != cache_index_t(-1));
        assert(shortest_length != std::numeric_limits<float>::infinity());
    };

    while (n_subgraphs > REDUCTION_N) {
        cache_index_t a;
        cache_index_t b;
        shortest_edge(&a, &b);
        distances[a][b] = std::numeric_limits<float>::infinity();
        auto subgraph_a = subgraph_of(a);
        auto subgraph_b = subgraph_of(b);
        if (subgraph_a != subgraph_b) {
            merge_subgraphs(subgraph_a, subgraph_b);
        }
    }


    auto find_next_subgraph = [&](int subgraph_id) {
        while(subgraphs[++subgraph_id][0] == cache_index_t(-1)) {
            assert(subgraph_id < REDUCTION_N * 2);
        }
        return subgraph_id;
    };

    auto reduce_subgraph = [&](int subgraph_id) {
        assert(subgraph_id >= 0);
        assert(subgraph_id < REDUCTION_N * 2);

        G new_gaussian = {scalar_t(0), typename G::pos_t(0), typename G::cov_t(0)};
        int n_merge = 0;
        unsigned current = 0;
        while (subgraphs[subgraph_id][current] != cache_index_t(-1)) {
            assert(current < REDUCTION_N * 2);
            cache_index_t cache_id = subgraphs[subgraph_id][current];
            assert(cache_id < REDUCTION_N * 2);
            gaussian_index_t gaussian_id = cached_gaussian_ids[cache_id];
            assert(gaussian_id < n.components);

            G current_g = gpe::gaussian<N_DIMS>(mixture_a[gaussian_id]);
            new_gaussian.weight += current_g.weight;
            new_gaussian.position += current_g.position;
            new_gaussian.covariance += glm::inverse(current_g.covariance);
            ++n_merge;
            ++current;
        }
        new_gaussian.weight /= scalar_t(n_merge);
        new_gaussian.position /= scalar_t(n_merge);
        new_gaussian.covariance = glm::inverse(new_gaussian.covariance /  scalar_t(n_merge));
        return new_gaussian;
    };

    int subgraph_id = -1;
    for (int i = 0; i < REDUCTION_N; ++i) {
        subgraph_id = find_next_subgraph(subgraph_id);
        gaussian_index_t new_gaussian_index = cached_gaussian_ids[subgraphs[subgraph_id][0]];
        assert(new_gaussian_index < n.components);
        gpe::gaussian<N_DIMS>(mixture_a[int(new_gaussian_index)]) = reduce_subgraph(subgraph_id);
        tmp_g_container_a[node->object_idx][i] = gaussian_index_torch_t(new_gaussian_index);
    }
}


template <int REDUCTION_N>
EXECUTION_DEVICES int count_gaussians(const lbvh::detail::Node* node,
                                        gpe::Accessor32<node_index_torch_t, 2> tmp_g_container_a) {
    auto c = tmp_g_container_a[node->object_idx];
    for (int i = 0; i < REDUCTION_N; ++i) {
        if (c[i] == -1)
            return i;
    }
    return REDUCTION_N;
}


template <int N_GAUSSIANS_TARGET>
EXECUTION_DEVICES int count_child_gaussians(const lbvh::detail::Node* node,
                                               const gpe::Accessor32<node_index_torch_t, 2>& nodes_a,
                                               gpe::Accessor32<node_index_torch_t, 2> tmp_g_container_a) {
    return count_gaussians<N_GAUSSIANS_TARGET>(reinterpret_cast<const lbvh::detail::Node*>(&nodes_a[node->left_idx][0]), tmp_g_container_a)
           + count_gaussians<N_GAUSSIANS_TARGET>(reinterpret_cast<const lbvh::detail::Node*>(&nodes_a[node->right_idx][0]), tmp_g_container_a);
}

template <typename scalar_t, int N_DIMS, int REDUCTION_N>
EXECUTION_DEVICES void iterate_over_nodes(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                          const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                          gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                                          const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                          const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                          gpe::PackedTensorAccessor32<int, 2> flags,
                                          gpe::PackedTensorAccessor32<gaussian_index_torch_t, 3> tmp_g_container_a,
                                          gpe::PackedTensorAccessor32<int32_t, 3> scratch_container,
                                          const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes, int n_components_target)
{
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Lbvh = lbvh::detail::basic_device_bvh<scalar_t, G, true>;

    auto node_id = node_index_t(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x + n_internal_nodes);
    const auto mixture_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
    if (mixture_id >= n_mixtures || node_id >= n_nodes)
        return;

    scratch_container[mixture_id][node_id][0] = 1;
    reinterpret_cast<float&>(scratch_container[mixture_id][node_id][1]) = gpe::integrate_inversed(gpe::gaussian<N_DIMS>(mixture[mixture_id][node_id - n_internal_nodes]));


    // collect Gs int tmp_g_container
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
        scratch_container[mixture_id][node_id][0] = scratch_container[mixture_id][node->left_idx][0] + scratch_container[mixture_id][node->right_idx][0];
        reinterpret_cast<float&>(scratch_container[mixture_id][node_id][1]) = reinterpret_cast<float&>(scratch_container[mixture_id][node->left_idx][1])
                                                                               + reinterpret_cast<float&>(scratch_container[mixture_id][node->right_idx][1]);
        if (count_child_gaussians<REDUCTION_N>(node, nodes[mixture_id], tmp_g_container_a[mixture_id]) > REDUCTION_N) {
            fit_reduce_node<scalar_t, N_DIMS, REDUCTION_N>(node, tmp_g_container_a[mixture_id], mixture[mixture_id]);
        }
        else {
            gaussian_index_t* destination = reinterpret_cast<gaussian_index_t*>(&tmp_g_container_a[mixture_id][node->object_idx][0]);
            auto n_copied = collect_child_gaussian_ids<REDUCTION_N>(node, nodes[mixture_id], tmp_g_container_a[mixture_id], destination);
            assert(n_copied <= REDUCTION_N);
        }
    }
}

template <typename scalar_t, int N_DIMS, int REDUCTION_N, int N_MAX_TARGET_COMPS = 128>
EXECUTION_DEVICES void collect_result(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                      const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                      const gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                                      gpe::PackedTensorAccessor32<scalar_t, 3> out_mixture,
                                      const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                      const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                      gpe::PackedTensorAccessor32<int, 2> flags,
                                      gpe::PackedTensorAccessor32<gaussian_index_torch_t, 3> tmp_g_container_a,
                                      gpe::PackedTensorAccessor32<int32_t, 3> scratch_container,
                                      const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes, int n_components_target)
{
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Lbvh = lbvh::detail::basic_device_bvh<scalar_t, G, true>;

    assert(n_components_target % REDUCTION_N == 0);
    static_assert (N_MAX_TARGET_COMPS % REDUCTION_N == 0, "N_MAX_TARGET_COMPS must be divisible by REDUCTION_N");

    const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
    if (mixture_id >= n_mixtures)
        return;

    constexpr int CACH_SIZE = N_MAX_TARGET_COMPS / REDUCTION_N;
    float selectedNodesRating[CACH_SIZE]; // up to 32 * REDUCTION_N gaussians
    node_index_t selectedNodes[CACH_SIZE];
    for (int i = 0; i < CACH_SIZE; ++i)
        selectedNodes[i] = node_index_t(-1);

    int n_selected_nodes = 0;
    auto compute_rating = [&](node_index_t node_id) {
        assert(node_id < n_nodes);
        return std::abs(reinterpret_cast<const float&>(scratch_container[mixture_id][node_id][1]));
    };
    auto cach_id_with_highest_rating = [&]() {
        float rating = -1;
        int best_node_id = -1;
        for (int i = 0; i < n_selected_nodes; ++i) {
            if (selectedNodesRating[i] > rating) {
                rating = selectedNodesRating[i];
                best_node_id = i;
            }
        }
        assert(best_node_id != -1);
        return best_node_id;
    };
    auto set_cache = [&](int cache_location, node_index_t node_id) {
        assert(node_id < n_nodes);
        assert(cache_location < CACH_SIZE);
        selectedNodes[cache_location] = node_id;
        selectedNodesRating[cache_location] = compute_rating(node_id);
    };

    set_cache(0, 0);
    n_selected_nodes++;

    while (n_selected_nodes < n_components_target / REDUCTION_N) {
        auto best_node_cache_id = cach_id_with_highest_rating();
        auto best_node_id = selectedNodes[best_node_cache_id];
        assert(best_node_id < n_nodes);
        auto best_node = reinterpret_cast<const lbvh::detail::Node*>(&nodes[mixture_id][best_node_id][0]);
        set_cache(best_node_cache_id, best_node->left_idx);
        set_cache(n_selected_nodes, best_node->right_idx);
        n_selected_nodes++;
        assert(n_selected_nodes < CACH_SIZE);
    }

    assert(n_selected_nodes == n_components_target / REDUCTION_N);
    for (int i = 0; i < n_selected_nodes; ++i) {
        for (int j = 0; j < REDUCTION_N; ++j) {
            // copy gaussians to their final location in out_mixture
            // tmp_g_container contains the addresses of the gaussians; we misuse object_idx in internal nodes for addressing that list of addresses.
            // first n_level_down bits of target_component_id were used for traversing the bvh, last reduction_n bits are used to select the current G
            auto node_id = selectedNodes[i];
            auto component_id = node_index_t(tmp_g_container_a[mixture_id][node_id][j]);
            assert(component_id < n.components);
            if (component_id != node_index_t(-1))
                gpe::gaussian<N_DIMS>(out_mixture[mixture_id][i * REDUCTION_N + j]) = gpe::gaussian<N_DIMS>(mixture[mixture_id][component_id]);;
        }
    }
}


} // anonymous namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(at::Tensor mixture, int n_components_target = 32) {
    using namespace torch::indexing;
    using LBVH = lbvh::Bvh<2, float>;

    constexpr int REDUCTION_N = 2;

    // todo: flatten mixture for kernel, i.g. nbatch/nlayers/ncomponents/7 => nmixture/ncomponents/7

    auto n = gpe::get_ns(mixture);
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.dims == 2, "atm only 2d gaussians")
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<float>(), "atm only float")

    const auto n_mixtures = n.batch * n.layers;
    const auto bvh = LBVH(mixture);
    const auto n_internal_nodes = bvh.m_n_internal_nodes;
    const auto n_nodes = bvh.m_n_nodes;
    mixture = mixture.view({n_mixtures, n.components, -1});
    auto flat_bvh_nodes = bvh.m_nodes.view({n_mixtures, n_nodes, -1});
    auto flat_bvh_aabbs = bvh.m_aabbs.view({n_mixtures, n_nodes, -1});
    auto scratch_mixture = mixture.clone();
    auto flag_container = torch::zeros({n_mixtures, n_internal_nodes}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Int));

    // scratch_container: int n_gaussians; float integral_sum;
    auto scratch_container = torch::zeros({n_mixtures, n_nodes, 2}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Int));
    auto tmp_g_container = -1 * torch::ones({n_mixtures, n_nodes, REDUCTION_N},
                                            torch::TensorOptions(mixture.device()).dtype(lbvh::detail::TorchTypeMapper<node_index_torch_t>::id()));
    auto flags_a = gpe::accessor<int, 2>(flag_container);
    auto tmp_g_container_a = gpe::accessor<gaussian_index_torch_t, 3>(tmp_g_container);
    auto scratch_container_a = gpe::accessor<int32_t, 3>(scratch_container);


    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(mixture.scalar_type(), n.dims, ([&] {
                                    dim3 dimBlock = dim3(32, 1, 1);
                                    dim3 dimGrid = dim3((uint(bvh.m_n_leaf_nodes) + dimBlock.x - 1) / dimBlock.x,
                                                        (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
                                                        (uint(1) + dimBlock.z - 1) / dimBlock.z);

                                   auto mixture_a = gpe::accessor<scalar_t, 3>(scratch_mixture);
                                   auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(flat_bvh_nodes);
                                   auto aabbs_a = gpe::accessor<scalar_t, 3>(flat_bvh_aabbs);

                                   auto fun = [mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a, scratch_container_a, n, n_mixtures, n_internal_nodes, n_nodes, n_components_target] EXECUTION_DEVICES
                                       (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                           iterate_over_nodes<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                                   mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a, scratch_container_a,
                                                                                   n, n_mixtures, n_internal_nodes, n_nodes, n_components_target);
                                       };
                                   gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                               }));

    auto out_mixture = torch::zeros({n_mixtures, n_components_target, mixture.size(-1)}, torch::TensorOptions(mixture.device()).dtype(mixture.dtype()));
    // make it valid, in case something doesn't get filled (due to an inbalance of the tree or just not enough elements)
    gpe::covariances(out_mixture) = torch::eye(n.dims, torch::TensorOptions(mixture.device()).dtype(mixture.dtype()));
    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(mixture.scalar_type(), n.dims, ([&] {
                                            dim3 dimBlock = dim3(32, 1, 1);
                                            dim3 dimGrid = dim3((uint(n_mixtures) + dimBlock.x - 1) / dimBlock.x, 1, 1);

                                            auto mixture_a = gpe::accessor<scalar_t, 3>(scratch_mixture);
                                            auto out_mixture_a = gpe::accessor<scalar_t, 3>(out_mixture);
                                            auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(flat_bvh_nodes);
                                            auto aabbs_a = gpe::accessor<scalar_t, 3>(flat_bvh_aabbs);

                                            auto fun = [mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a, scratch_container_a, n, n_mixtures, n_internal_nodes, n_nodes, n_components_target]
                                                EXECUTION_DEVICES
                                                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                                    collect_result<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                                        mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, tmp_g_container_a, scratch_container_a,
                                                                                        n, n_mixtures, n_internal_nodes, n_nodes, n_components_target);
                                                };
                                            gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                        }));

    GPE_CUDA_ASSERT(cudaPeekAtLastError())
    GPE_CUDA_ASSERT(cudaDeviceSynchronize())

    return std::make_tuple(out_mixture.view({n.batch, n.layers, n_components_target, -1}), bvh.m_nodes, bvh.m_aabbs);
}


} // namespace bvh_mhem_fit

