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
#include "containers.h"
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
using Node  = lbvh::detail::Node;


template <typename scalar_t>
struct UIntOfSize {
    using type = uint32_t;
};

template <>
struct UIntOfSize<double> {
    using type = uint64_t;
};

template<typename scalar_t, int N_DIMS, int REDUCTION_N>
struct AugmentedBvh
{
    using aabb_type  = lbvh::Aabb<scalar_t>;
    using Gaussian_type = gpe::Gaussian<N_DIMS, scalar_t>;
    struct NodeAttributes {
        using UIntType = typename UIntOfSize<scalar_t>::type;
        gpe::Vector<Gaussian_type, REDUCTION_N, UIntType> gaussians;
        scalar_t gm_integral;
        UIntType n_child_leaves;
        // when adding an attribute, remember to update the line
        // auto node_attributes = torch::zeros({n_mixtures, n_nodes, REDUCTION_N * mixture.size(-1) + 3}, torch::TensorOptions(mixture.device()).dtype(mixture.scalar_type()));
    };
//    static_assert (alignof (Gaussian_type) == 4, "adsf");
    static_assert (sizeof (NodeAttributes) <= sizeof(scalar_t) * (REDUCTION_N * (1 + N_DIMS + N_DIMS * N_DIMS) + 3), "NodeAttribute is too large and won't fit into the torch::Tensor");
    static_assert (sizeof (NodeAttributes) == sizeof(scalar_t) * (REDUCTION_N * (1 + N_DIMS + N_DIMS * N_DIMS) + 3), "NodeAttribute has unexpected size (it could be smaller, no problem, just unexpected)");

    const unsigned n_internal_nodes;
    const unsigned n_leaves;
    const unsigned n_nodes;
    const int32_t _padding = 0;

    const Node* nodes;                         // size = n_nodes
    const aabb_type* aabbs;                         // size = n_nodes
    Gaussian_type* gaussians;                       // size = n_leaves
    NodeAttributes* per_node_attributes;            // size = n_nodes

    EXECUTION_DEVICES
    AugmentedBvh(int mixture_id,
                 const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                 const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                 gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                 gpe::PackedTensorAccessor32<scalar_t, 3> node_attributes,
                 const gpe::MixtureNs n, const unsigned n_internal_nodes, const unsigned n_nodes)
        : n_internal_nodes(n_internal_nodes), n_leaves(unsigned(n.components)), n_nodes(n_nodes),
          nodes(reinterpret_cast<const Node*>(&nodes[mixture_id][0][0])),
          aabbs(reinterpret_cast<const aabb_type*>(&aabbs[mixture_id][0][0])),
          gaussians(reinterpret_cast<Gaussian_type*>(&mixture[mixture_id][0][0])),
          per_node_attributes(reinterpret_cast<NodeAttributes*>(&node_attributes[mixture_id][0][0]))
    {}

    EXECUTION_DEVICES unsigned count_per_node_gaussians_of_children(const lbvh::detail::Node* node) const {
        assert(node->left_idx != node_index_t(-1));
        assert(node->right_idx != node_index_t(-1));
        return this->per_node_attributes[node->left_idx].gaussians.size() + this->per_node_attributes[node->right_idx].gaussians.size();
    }

    EXECUTION_DEVICES gpe::Vector<Gaussian_type, REDUCTION_N * 2> collect_child_gaussians(const lbvh::detail::Node* node) const {
        assert(node->left_idx != node_index_t(-1));
        assert(node->right_idx != node_index_t(-1));
        gpe::Vector<Gaussian_type, REDUCTION_N * 2> retval;
        retval.push_all_back(per_node_attributes[node->left_idx].gaussians);
        retval.push_all_back(per_node_attributes[node->right_idx].gaussians);
        return retval;
    }

    EXECUTION_DEVICES node_index_t node_id(const lbvh::detail::Node* node) {
        auto id = node_index_t(node - nodes);
        assert(id < n_nodes);
        return id;
    }
};

// todo: min of KL divergencies is probably a better distance
template <typename scalar_t, int N_DIMS>
EXECUTION_DEVICES scalar_t centroid_distance(const gpe::Gaussian<N_DIMS, scalar_t>& a, const gpe::Gaussian<N_DIMS, scalar_t>& b) {
    if (gpe::sign(a.weight) != gpe::sign(b.weight))
        return std::numeric_limits<scalar_t>::infinity();
    return gpe::squared_norm(a.position - b.position);
}


template<typename scalar_t, int N_DIMS, uint32_t N_GAUSSIANS_CAPACITY, typename DisparityFunction>
EXECUTION_DEVICES
gpe::Vector2d<scalar_t, N_GAUSSIANS_CAPACITY> compute_disparity_matrix(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS_CAPACITY>& gaussians,
                                                                        DisparityFunction disparity_fun) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    gpe::Vector2d<scalar_t, N_GAUSSIANS_CAPACITY> matrix;
    matrix.resize(gaussians.size());
    for (unsigned i = 0; i < matrix.size(); ++i) {
        matrix[i].resize(gaussians.size());
    }

    for (unsigned i = 0; i < gaussians.size(); ++i) {
        const G& g_i = gaussians[i];
        matrix[i][i] = 0;
        for (unsigned j = i + 1; j < gaussians.size(); ++j) {
            const G& g_j = gaussians[j];
            scalar_t disparity = disparity_fun(g_i, g_j);
            assert(std::isnan(disparity) == false);
            // todo: non-symmetric disparities, like kl-divergence
            matrix[i][j] = disparity;
            matrix[j][i] = disparity;
        }
    }
    return matrix;
}

template <uint32_t N_CLUSTERS, typename scalar_t, int N_DIMS, uint32_t N_INPUT>
EXECUTION_DEVICES
gpe::Array<gpe::Vector<gaussian_index_t, N_INPUT - N_CLUSTERS + 1>, N_CLUSTERS> clusterise(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_INPUT>& gaussians,
                                                                                           const gpe::Vector2d<scalar_t, N_INPUT>& disparities) {
    // this is a greedy smallest spanning subtrees algorithm
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    static_assert (N_CLUSTERS <= N_INPUT, "N output clusters must be larger than n input");
    assert(N_CLUSTERS <= gaussians.size());

    gpe::Vector2d<gaussian_index_t, N_INPUT> subgraphs;
    for (unsigned i = 0; i < gaussians.size(); ++i) {
        subgraphs.push_back({i});
    }
    unsigned n_subgraphs = subgraphs.size();

    auto merge_subgraphs = [&](unsigned a, unsigned b) {
        assert (a != b);
        auto a_ = gpe::min(a, b);
        auto b_ = gpe::max(a, b);

        subgraphs[a_].push_all_back(subgraphs[b_]);
        subgraphs[b_].clear();
        --n_subgraphs;
    };
    auto subgraph_of = [&](gaussian_index_t id) {
        for (unsigned i = 0; i < subgraphs.size(); ++i) {
            for (unsigned j = 0; j < subgraphs[i].size(); ++j) {
                if (subgraphs[i][j] == id)
                    return i;
            }
        }
        assert(false);
        return unsigned(-1);
    };

    gpe::BitSet<N_INPUT * N_INPUT> invalid_edges;
    auto shortest_edge = [&](gaussian_index_t* a, gaussian_index_t* b) {
        *a = gaussian_index_t(-1);
        *b = gaussian_index_t(-1);
        scalar_t shortest_length = std::numeric_limits<scalar_t>::infinity();
        for (gaussian_index_t i = 0; i < gaussians.size(); ++i) {
            for (gaussian_index_t j = i + 1; j < gaussians.size(); ++j) {
                if (!invalid_edges.isSet(i * gaussians.size() + j) && disparities[i][j] < shortest_length) {
                    *a = i;
                    *b = j;
                    shortest_length = disparities[i][j];
                }
            }
        }
        assert(*a != gaussian_index_t(-1));
        assert(*b != gaussian_index_t(-1));
        assert(shortest_length != std::numeric_limits<scalar_t>::infinity());
    };

    while (n_subgraphs > N_CLUSTERS) {
        gaussian_index_t a;
        gaussian_index_t b;
        shortest_edge(&a, &b);
        assert(a < b);
        invalid_edges.set1(a * gaussians.size() + b);
        auto subgraph_a = subgraph_of(a);
        auto subgraph_b = subgraph_of(b);
        if (subgraph_a != subgraph_b) {
            merge_subgraphs(subgraph_a, subgraph_b);
        }
    }

    auto find_next_subgraph = [&](unsigned subgraph_id) {
        while(subgraphs[++subgraph_id].size() == 0) {
            assert(subgraph_id < N_INPUT);
        }
        assert(subgraph_id < N_INPUT);
        return subgraph_id;
    };

    unsigned subgraph_id = unsigned(-1);
    assert(n_subgraphs == N_CLUSTERS);
    gpe::Array<gpe::Vector<gaussian_index_t, N_INPUT - N_CLUSTERS + 1>, N_CLUSTERS> retval;
    for (unsigned i = 0; i < N_CLUSTERS; ++i) {
        subgraph_id = find_next_subgraph(subgraph_id);
        retval[i].push_all_back(subgraphs[subgraph_id]);
    }

    return retval;
}

template <typename scalar_t, int N_DIMS, uint32_t N_GAUSSIANS_CAPACITY, uint32_t N_MAX_CLUSTER_ELEMENTS>
EXECUTION_DEVICES
gpe::Gaussian<N_DIMS, scalar_t> averageCluster(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS_CAPACITY>& mixture,
                                               const gpe::Vector<gaussian_index_t, N_MAX_CLUSTER_ELEMENTS>& cluster_indices) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    G new_gaussian = {scalar_t(0), typename G::pos_t(0), typename G::cov_t(0)};

    for (unsigned i = 0; i < cluster_indices.size(); ++i) {
        auto gaussian_id = cluster_indices[i];
        const auto& gaussian = mixture[gaussian_id];

        assert(new_gaussian.weight == 0 || gpe::sign(new_gaussian.weight) == gpe::sign(gaussian.weight)); // can't merge positive and negative gaussian
        new_gaussian.weight += gaussian.weight;
        new_gaussian.position += gaussian.weight * gaussian.position;
        assert(glm::determinant(gaussian.covariance) > 0);
        new_gaussian.covariance += gaussian.weight * glm::inverse(gaussian.covariance);
    }
    if (new_gaussian.weight == 0) {
        new_gaussian.covariance = typename G::cov_t(1.0);
        assert(glm::determinant(new_gaussian.covariance) > 0);
    }
    else {
        new_gaussian.position /= new_gaussian.weight;
        new_gaussian.covariance /= new_gaussian.weight;
        new_gaussian.weight /= scalar_t(cluster_indices.size());
        assert(glm::determinant(new_gaussian.covariance) > 0);
        new_gaussian.covariance = glm::inverse(new_gaussian.covariance);
        assert(glm::determinant(new_gaussian.covariance) > 0);
    }
    assert(std::isnan(new_gaussian.weight) == false);
    assert(std::isnan(glm::dot(new_gaussian.position, new_gaussian.position)) == false);
    assert(std::isnan(glm::determinant(new_gaussian.covariance)) == false);

    return new_gaussian;
};

template <typename scalar_t, int N_DIMS, int REDUCTION_N>
EXECUTION_DEVICES void fit_reduce_node(AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>& bvh,
                                       const lbvh::detail::Node* node) {
    using Abvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    // for now (testing) simply select N_GAUSSIANS_TARGET strongest gaussians
    // no stl available in cuda 10.1.
    const auto gaussians = bvh.collect_child_gaussians(node);

    const auto disparity_matrix = compute_disparity_matrix(gaussians, centroid_distance<scalar_t, N_DIMS>);   // returns gpe::Vector<gpe::Vector>
    const auto clustering = clusterise<REDUCTION_N>(gaussians, disparity_matrix);                            // returns gpe::Array<gpe::Vector>
    assert(clustering.size() == REDUCTION_N);

    typename Abvh::NodeAttributes& destination_attribute = bvh.per_node_attributes[bvh.node_id(node)];
    for (unsigned i = 0; i < REDUCTION_N; ++i) {
        destination_attribute.gaussians.push_back(averageCluster(gaussians, clustering[i]));
    }
}


template <typename scalar_t, int N_DIMS, int REDUCTION_N>
EXECUTION_DEVICES void iterate_over_nodes(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                          const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                          gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                                          const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                          const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                          gpe::PackedTensorAccessor32<int, 2> flags,
                                          gpe::PackedTensorAccessor32<scalar_t, 3> node_attributes,
                                          const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes, int n_components_target)
{
    GPE_UNUSED(gpe_gridDim)
    GPE_UNUSED(n_components_target);
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;

    auto node_id = node_index_t(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x + n_internal_nodes);
    const auto mixture_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
    if (mixture_id >= n_mixtures || node_id >= n_nodes)
        return;

    Bvh bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>(mixture_id, nodes, aabbs, mixture, node_attributes, n, n_internal_nodes, n_nodes);
    {
        const G& leaf_gaussian = bvh.gaussians[node_id - n_internal_nodes];
        bvh.per_node_attributes[node_id].gaussians.push_back(leaf_gaussian);
        bvh.per_node_attributes[node_id].n_child_leaves = 1;
        bvh.per_node_attributes[node_id].gm_integral = gpe::integrate_inversed(leaf_gaussian);
    }

    // collect Gs in per_node_gaussians
    const Node* node = &bvh.nodes[node_id];
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
        node = &bvh.nodes[node_id];
        bvh.per_node_attributes[node_id].n_child_leaves = bvh.per_node_attributes[node->left_idx].n_child_leaves + bvh.per_node_attributes[node->right_idx].n_child_leaves;
        bvh.per_node_attributes[node_id].gm_integral = bvh.per_node_attributes[node->left_idx].gm_integral + bvh.per_node_attributes[node->right_idx].gm_integral;

        auto gaussian_count = bvh.count_per_node_gaussians_of_children(node);
        if (gaussian_count > REDUCTION_N) {
            fit_reduce_node<scalar_t, N_DIMS, REDUCTION_N>(bvh, node);
        }
        else {
            bvh.per_node_attributes[node_id].gaussians.push_all_back(bvh.collect_child_gaussians(node));

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
                                      gpe::PackedTensorAccessor32<scalar_t, 3> node_attributes,
                                      const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes, int n_components_target)
{
    GPE_UNUSED(gpe_gridDim)
    GPE_UNUSED(flags);
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;

    assert(n_components_target % REDUCTION_N == 0);
    static_assert (N_MAX_TARGET_COMPS % REDUCTION_N == 0, "N_MAX_TARGET_COMPS must be divisible by REDUCTION_N");

    const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
    if (mixture_id >= n_mixtures)
        return;

    Bvh bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>(mixture_id, nodes, aabbs, mixture, node_attributes, n, n_internal_nodes, n_nodes);
    constexpr int CACH_SIZE = N_MAX_TARGET_COMPS / REDUCTION_N;
    scalar_t selectedNodesRating[CACH_SIZE];
    node_index_t selectedNodes[CACH_SIZE];
    for (int i = 0; i < CACH_SIZE; ++i)
        selectedNodes[i] = node_index_t(-1);

    int n_selected_nodes = 0;
    auto compute_rating = [&](node_index_t node_id) {
        assert(node_id < n_nodes);
        return std::abs(bvh.per_node_attributes[node_id].gm_integral);
    };
    auto cach_id_with_highest_rating = [&]() {
        scalar_t rating = -1;
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
        if (best_node_id < n_internal_nodes) {
            const auto& best_node = bvh.nodes[best_node_id];
            set_cache(best_node_cache_id, best_node.left_idx);
            set_cache(n_selected_nodes, best_node.right_idx);
            n_selected_nodes++;
        }
        else {
            // best_node_id is a leaf node, it can't be further expanded.; ignore it.
            selectedNodesRating[best_node_cache_id] = -1;
        }
        assert(n_selected_nodes < CACH_SIZE);
    }

    // copy gaussians to their final location in out_mixture
    assert(n_selected_nodes == n_components_target / REDUCTION_N);
    for (int i = 0; i < n_selected_nodes; ++i) {
        auto node_id = selectedNodes[i];
        typename Bvh::NodeAttributes& destination_attribute = bvh.per_node_attributes[node_id];

        for (unsigned j = 0; j < destination_attribute.gaussians.size(); ++j) {
            gpe::gaussian<N_DIMS>(out_mixture[mixture_id][i * REDUCTION_N + int(j)]) = destination_attribute.gaussians[j];
        }
        // todo: log or something if destination_attribute.n_gaussians != REDUCTION_N. we are loosing slots for more gaussians; adapt node rating function.
    }
}


} // anonymous namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(at::Tensor mixture, int n_components_target = 32) {
    using namespace torch::indexing;
    using LBVH = lbvh::Bvh<2, float>;

    constexpr int REDUCTION_N = 8;

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
    auto node_attributes = torch::zeros({n_mixtures, n_nodes, REDUCTION_N * mixture.size(-1) + 3}, torch::TensorOptions(mixture.device()).dtype(mixture.scalar_type()));


    GPE_DISPATCH_FLOATING_TYPES_AND_DIM(mixture.scalar_type(), n.dims, ([&] {
                                   dim3 dimBlock = dim3(32, 1, 1);
                                   dim3 dimGrid = dim3((uint(bvh.m_n_leaf_nodes) + dimBlock.x - 1) / dimBlock.x,
                                                       (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
                                                       (uint(1) + dimBlock.z - 1) / dimBlock.z);


                                   auto mixture_a = gpe::accessor<scalar_t, 3>(scratch_mixture);
                                   auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(flat_bvh_nodes);
                                   auto aabbs_a = gpe::accessor<scalar_t, 3>(flat_bvh_aabbs);
                                   auto node_attributes_a = gpe::accessor<scalar_t, 3>(node_attributes);

                                   auto fun = [mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, n_components_target] EXECUTION_DEVICES
                                       (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                           iterate_over_nodes<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                                   mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
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
                                            auto node_attributes_a = gpe::accessor<scalar_t, 3>(node_attributes);

                                            auto fun = [mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, n_components_target]
                                                EXECUTION_DEVICES
                                                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                                    collect_result<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                                        mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                                                        n, n_mixtures, n_internal_nodes, n_nodes, n_components_target);
                                                };
                                            gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                        }));

    GPE_CUDA_ASSERT(cudaPeekAtLastError())
    GPE_CUDA_ASSERT(cudaDeviceSynchronize())

    return std::make_tuple(out_mixture.view({n.batch, n.layers, n_components_target, -1}), bvh.m_nodes, bvh.m_aabbs);
}


} // namespace bvh_mhem_fit

