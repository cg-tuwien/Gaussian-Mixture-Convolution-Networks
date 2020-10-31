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

template <typename scalar_t, int N_DIMS, int N_FITTING_COMPONENTS>
EXECUTION_DEVICES scalar_t likelihood(const gpe::Gaussian<N_DIMS, scalar_t>& target, const gpe::Gaussian<N_DIMS, scalar_t>& fitting) {
    // Continuous projection for fast L 1 reconstruction: Equation 9
    scalar_t normal_amplitude = gpe::gaussian_amplitude(fitting.covariance);
    scalar_t a = gpe::evaluate(target.position, normal_amplitude, fitting.position, fitting.covariance);
    auto c = glm::inverse(fitting.covariance) * target.covariance;
    scalar_t b = gpe::exp(scalar_t(-0.5) * gpe::trace(c));
    scalar_t target_normal_amplitudes = gpe::gaussian_amplitude(target.covariance);
    scalar_t wi_bar = N_FITTING_COMPONENTS * target.weight / target_normal_amplitudes;
    // pow(0, 0) gives nan in cuda with fast math
    return gpe::pow(gpe::abs(a * b) + scalar_t(0.000000001), wi_bar);
}

template <typename scalar_t, int N_DIMS>
EXECUTION_DEVICES scalar_t kl_divergence(const gpe::Gaussian<N_DIMS, scalar_t>& target, const gpe::Gaussian<N_DIMS, scalar_t>& fitting) {
    auto p_diff = target.position - fitting.position;

    auto target_cov = target.covariance;
    auto fitting_cov = fitting.covariance;
//    auto inversed_target_cov = glm::inverse(target.covariance);
    auto inversed_fitting_cov = glm::inverse(fitting.covariance);

    // mahalanobis_factor = mahalanobis distance squared
    auto mahalanobis_factor = glm::dot(p_diff, inversed_fitting_cov * p_diff);
    auto trace = gpe::trace(inversed_fitting_cov * target_cov);
    auto logarithm = gpe::log(glm::determinant(target_cov) / glm::determinant(fitting_cov));
    return scalar_t(0.5) * (mahalanobis_factor + trace - N_DIMS - logarithm);
}

template <uint32_t N_CLUSTERS, typename scalar_t, int N_DIMS, uint32_t N_INPUT>
EXECUTION_DEVICES
gpe::Array<gpe::Vector<gaussian_index_t, N_INPUT - N_CLUSTERS + 1>, N_CLUSTERS> clusterise(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_INPUT>& gaussians,
                                                                                           const gpe::Vector2d<scalar_t, N_INPUT>& disparities) {
    // this is a greedy smallest spanning subtrees algorithm
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    static_assert (N_CLUSTERS <= N_INPUT, "N output clusters must be larger than n input");
    assert(N_CLUSTERS <= gaussians.size());
    assert(!gpe::reduce(disparities, false, [](bool o, scalar_t v) { return o || gpe::isnan(v); }));

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

    auto shortest_edge = [&disparities, &gaussians](gaussian_index_t* a, gaussian_index_t* b, gpe::BitSet<N_INPUT * N_INPUT>* invalid_edges) {
        *a = gaussian_index_t(-1);
        *b = gaussian_index_t(-1);
        scalar_t shortest_length = std::numeric_limits<scalar_t>::infinity();
        for (gaussian_index_t i = 0; i < gaussians.size(); ++i) {
            for (gaussian_index_t j = i + 1; j < gaussians.size(); ++j) {
                if (!invalid_edges->isSet(i * gaussians.size() + j) && disparities[i][j] < shortest_length) {
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
        shortest_edge(&a, &b, &invalid_edges);
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

    assert(cluster_indices.size() > 0);

    for (unsigned i = 0; i < cluster_indices.size(); ++i) {
        auto gaussian_id = cluster_indices[i];
        const auto& gaussian = mixture[gaussian_id];

        assert(new_gaussian.weight == 0 || gpe::sign(new_gaussian.weight) == gpe::sign(gaussian.weight)); // can't merge positive and negative gaussian
        new_gaussian.weight += gaussian.weight;
        new_gaussian.position += gaussian.weight * gaussian.position;
        assert(glm::determinant(gaussian.covariance) > 0);
        new_gaussian.covariance += gaussian.weight * gaussian.covariance;
    }
    if (gpe::abs(new_gaussian.weight) < scalar_t(0.00000000001)) {
        new_gaussian.covariance = typename G::cov_t(1.0);
        assert(glm::determinant(new_gaussian.covariance) > 0);
    }
    else {
        new_gaussian.position /= new_gaussian.weight;
        new_gaussian.covariance /= new_gaussian.weight;
        new_gaussian.weight /= scalar_t(cluster_indices.size());
        assert(glm::determinant(new_gaussian.covariance) > 0);
    }
    assert(std::isnan(new_gaussian.weight) == false);
    assert(std::isnan(glm::dot(new_gaussian.position, new_gaussian.position)) == false);
    assert(std::isnan(glm::determinant(new_gaussian.covariance)) == false);

    return new_gaussian;
};

template <typename scalar_t, int N_DIMS, uint32_t N_GAUSSIANS, typename VectorSizeType>
EXECUTION_DEVICES
scalar_t integrate_abs_mixture(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS, VectorSizeType>& mixture) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    scalar_t abs_integral = gpe::reduce(mixture, scalar_t(0), [](scalar_t i, const G& g) {
        scalar_t ci = gpe::integrate(g);
        return i + gpe::abs(ci);
    });

    return abs_integral;
}


template <typename scalar_t, int N_DIMS, unsigned N_GAUSSIANS>
EXECUTION_DEVICES
    gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS> normalise_mixture(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS>& mixture, scalar_t* abs_integral_ptr = nullptr) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    scalar_t abs_integral = integrate_abs_mixture(mixture);
    abs_integral = gpe::max(scalar_t(0.05), abs_integral);
    if (abs_integral_ptr)
        *abs_integral_ptr = abs_integral;

    return gpe::transform(mixture, [abs_integral](const G& g) { return G{g.weight / abs_integral, g.position, g.covariance}; });
}

#define GPE_DISPARITY_METHOD 2

template <typename scalar_t, int N_DIMS, int REDUCTION_N>
EXECUTION_DEVICES
void fit_reduce_node(AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>& bvh,
                     const lbvh::detail::Node* node) {
    using Abvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using pos_t = typename G::pos_t;
    using cov_t = typename G::cov_t;

    namespace fun = gpe::functors;


    auto has_nan = [](const auto& vec) {
        return gpe::reduce(vec, false, [](bool o, auto v) { return o || gpe::isnan(v); });
    };

    constexpr scalar_t kl_div_threshold = scalar_t(2.0);
    constexpr unsigned N_FITTING = REDUCTION_N;
    constexpr unsigned N_TARGET = REDUCTION_N * 2;

    const gpe::Vector<G, N_TARGET> target = bvh.collect_child_gaussians(node);
    scalar_t abs_integral;
    const gpe::Vector<G, N_TARGET> target_double_gmm = normalise_mixture(target, &abs_integral);

#if GPE_DISPARITY_METHOD == 0
    auto disparity_matrix = gpe::outer_product(target_double_gmm, target_double_gmm, centroid_distance<scalar_t, N_DIMS>);   // returns gpe::Vector<gpe::Vector>
#elif GPE_DISPARITY_METHOD == 1
    auto disparity_matrix = gpe::outer_product(target_double_gmm, target_double_gmm, likelihood<scalar_t, N_DIMS, N_TARGET>);   // returns gpe::Vector<gpe::Vector>
    for (unsigned i = 0; i < disparity_matrix.size(); ++i) {
        for (unsigned j = i + 1; j < disparity_matrix[i].size(); ++j) {
            disparity_matrix[i][j] = gpe::min(-disparity_matrix[i][j], -disparity_matrix[j][i]);
        }
    }
#else
    auto disparity_matrix = gpe::outer_product(target_double_gmm, target_double_gmm, kl_divergence<scalar_t, N_DIMS>);   // returns gpe::Vector<gpe::Vector>
    for (unsigned i = 0; i < disparity_matrix.size(); ++i) {
        for (unsigned j = i + 1; j < disparity_matrix[i].size(); ++j) {
            disparity_matrix[i][j] = gpe::min(disparity_matrix[i][j], disparity_matrix[j][i]);
        }
    }
#endif

    const auto clustering = clusterise<N_FITTING>(target_double_gmm, disparity_matrix);                             // returns gpe::Array<gpe::Vector>
    assert(clustering.size() == N_FITTING);

//    typename Abvh::NodeAttributes& destination_attribute = bvh.per_node_attributes[bvh.node_id(node)];
//    for (unsigned i = 0; i < N_FITTING; ++i) {
//        destination_attribute.gaussians.push_back(averageCluster(target, clustering[i]));
//    }
//    return;
    gpe::Vector<G, N_FITTING> fitting;
    for (unsigned i = 0; i < N_FITTING; ++i) {
        fitting.push_back(averageCluster(target_double_gmm, clustering[i]));
    }
    const gpe::Vector<G, N_FITTING> fitting_double_gmm = normalise_mixture(fitting);

    const gpe::Vector2d<scalar_t, N_TARGET, N_FITTING> likelihood_matrix = gpe::outer_product(target_double_gmm, fitting_double_gmm, likelihood<scalar_t, N_DIMS, N_FITTING>);

    // todo: test modification of clamp matrix: at least one row element or x percent row elements should be 1.
    //       rational: otherwise certain target gaussians are not covered at all.
    auto clamp_matrix = gpe::outer_product(target_double_gmm, fitting_double_gmm, [kl_div_threshold](auto target, auto fitting) {
        return (gpe::sign(fitting.weight) == gpe::sign(target.weight) && kl_divergence<scalar_t, N_DIMS>(target, fitting) < kl_div_threshold) ? scalar_t(1) : scalar_t(0);
    });

    const gpe::Vector2d<scalar_t, N_TARGET, N_FITTING> clamped_likelihood_matrix = gpe::cwise_fun(likelihood_matrix, clamp_matrix, [] EXECUTION_DEVICES (scalar_t a, scalar_t b) { return a * b; });


//    likelihoods = likelihoods * (gm.weights(fitting_double_gmm).abs() / gm.normal_amplitudes(gm.covariances(fitting_double_gmm))).view(n_batch, n_layers, 1, n_components_fitting)
    const auto pure_fitting_weights = gpe::transform(fitting_double_gmm, [](const G& g) { return gpe::abs(g.weight) / gpe::gaussian_amplitude(g.covariance); });
    assert(!has_nan(pure_fitting_weights));
    const gpe::Vector2d<scalar_t, N_TARGET, N_FITTING> weighted_likelihood_matrix = gpe::cwise_fun(pure_fitting_weights, clamped_likelihood_matrix, fun::times<scalar_t>);
    assert(!has_nan(weighted_likelihood_matrix));


//    likelihoods_sum = likelihoods.sum(3, keepdim=True)
    const gpe::Vector<scalar_t, N_TARGET> weighted_likelihood_sum = gpe::reduce_rows(weighted_likelihood_matrix, scalar_t(0), fun::plus<scalar_t>);
    const gpe::Vector<scalar_t, N_TARGET> weighted_likelihood_sum_clamped = gpe::transform(weighted_likelihood_sum, [](scalar_t v) { return gpe::max(v, scalar_t(0.00000000000001)); });
//    responsibilities = likelihoods / (likelihoods_sum + 0.00001)
    const gpe::Vector2d<scalar_t, N_TARGET, N_FITTING> responsibilities_1 = gpe::cwise_fun(weighted_likelihood_matrix, weighted_likelihood_sum_clamped, fun::divided_AbyB<scalar_t>);

//    responsibilities = responsibilities * (gm.weights(target_double_gmm).abs() / gm.normal_amplitudes(gm.covariances(target_double_gmm))).unsqueeze(-1)
    const auto pure_target_weights = gpe::transform(target_double_gmm, [](const G& g) { return gpe::abs(g.weight) / gpe::gaussian_amplitude(g.covariance); });
    assert(!has_nan(pure_target_weights));

    const gpe::Vector2d<scalar_t, N_TARGET, N_FITTING> responsibilities_2 = gpe::cwise_fun(responsibilities_1, pure_target_weights, fun::times<scalar_t>);

//    assert not torch.any(torch.isnan(responsibilities))
    assert(!has_nan(responsibilities_2));

//    newWeights = torch.sum(responsibilities, 2)
    gpe::Vector<scalar_t, N_FITTING> newWeights = gpe::reduce_cols(responsibilities_2, scalar_t(0), fun::plus<scalar_t>);
//    assert not torch.any(torch.isnan(newWeights))
    assert(!has_nan(newWeights));

//    responsibilities = responsibilities / (newWeights + 0.00001).view(n_batch, n_layers, 1, n_components_fitting)
    const gpe::Vector2d<scalar_t, N_TARGET, N_FITTING> responsibilities_3 = gpe::cwise_fun(
                                                                                gpe::transform(newWeights, [](auto w) { return gpe::max(w, scalar_t(0.00000000000001)); }),
                                                                                responsibilities_2,
                                                                                fun::divided_BbyA<scalar_t>);
//    assert not torch.any(torch.isnan(responsibilities))
    assert(!has_nan(responsibilities_3));
//    assert torch.all(responsibilities >= 0)
    assert(!gpe::reduce(responsibilities_3, false, [](bool o, scalar_t v) { return o || v < 0; }));

//    newPositions = torch.sum(responsibilities.unsqueeze(-1) * gm.positions(target_double_gmm).view(n_batch, n_layers, n_components_target, 1, n_dims), 2)
    gpe::Vector2d<pos_t, N_TARGET, N_FITTING> weightedPositions = gpe::cwise_fun(responsibilities_3, target, [](scalar_t r, const G& g) { return r * g.position; });
    gpe::Vector<pos_t, N_FITTING> newPositions = gpe::reduce_cols(weightedPositions, pos_t(0), fun::plus<pos_t>);
//    assert not torch.any(torch.isnan(newPositions))
    assert(!has_nan(newPositions));

//    posDiffs = gm.positions(target_double_gmm).view(n_batch, n_layers, n_components_target, 1, n_dims, 1) - newPositions.view(n_batch, n_layers, 1, n_components_fitting, n_dims, 1)
    gpe::Vector<pos_t, N_TARGET> targetPositions = gpe::transform(target, [](const G& g){ return g.position; });
    gpe::Vector2d<pos_t, N_TARGET, N_FITTING> posDiffs = gpe::outer_product(targetPositions, newPositions, fun::minus<pos_t>);
//    assert not torch.any(torch.isnan(posDiffs))
    assert(!has_nan(posDiffs));

/*
    newCovariances = (torch.sum(responsibilities.unsqueeze(-1).unsqueeze(-1) *
                                (gm.covariances(target_double_gmm).view(n_batch, n_layers, n_components_target, 1, n_dims, n_dims) + posDiffs.matmul(posDiffs.transpose(-1, -2))), 2))
*/
    const gpe::Vector2d<cov_t, N_TARGET, N_FITTING> posDiffsOuter = gpe::transform(posDiffs, [](const pos_t& p) { return glm::outerProduct(p, p); });
    gpe::Vector<cov_t, N_TARGET> targetCovs = gpe::transform(target, [](const G& g){ return g.covariance; });
    const gpe::Vector2d<cov_t, N_TARGET, N_FITTING> unweightedCovs = gpe::cwise_fun(posDiffsOuter, targetCovs, fun::plus<cov_t>);

    const gpe::Vector2d<cov_t, N_TARGET, N_FITTING> weightedCovs = gpe::cwise_fun(responsibilities_3, unweightedCovs, [](scalar_t r, const cov_t& cov) { return r * cov; });
    gpe::Vector<cov_t, N_FITTING> newCovariances = gpe::reduce_cols(weightedCovs, cov_t(0), fun::plus<cov_t>);

//    newCovariances = newCovariances + (newWeights < 0.0001).unsqueeze(-1).unsqueeze(-1) * torch.eye(n_dims, device=device).view(1, 1, 1, n_dims, n_dims) * 0.0001
    newCovariances = gpe::cwise_fun(newCovariances, newWeights, [](cov_t cov, scalar_t w) {
        if (w < scalar_t(0.0001))
            cov += cov_t(1) * scalar_t(0.0001);
        return cov;
    });

//    assert not torch.any(torch.isnan(newCovariances))
    assert(!has_nan(newCovariances));

//    normal_amplitudes = gm.normal_amplitudes(newCovariances)
    gpe::Vector<scalar_t, N_FITTING> normal_amplitudes = gpe::transform(newCovariances, [](const cov_t& cov) { return gpe::gaussian_amplitude(cov); });
    assert(!has_nan(normal_amplitudes));

//    fitting_double_gmm = gm.pack_mixture(newWeights.contiguous() * normal_amplitudes * gm.weights(fitting_double_gmm).sign(), newPositions.contiguous(), newCovariances.contiguous())
    typename Abvh::NodeAttributes& destination_attribute = bvh.per_node_attributes[bvh.node_id(node)];
    for (unsigned i = 0; i < fitting.size(); ++i) {
        destination_attribute.gaussians.push_back(G{newWeights[i] * normal_amplitudes[i] * abs_integral,
                                                    newPositions[i],
                                                    newCovariances[i]});
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
        bvh.per_node_attributes[node_id].gm_integral = gpe::integrate(leaf_gaussian);
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

    constexpr int REDUCTION_N = 4;

    // todo: flatten mixture for kernel, i.g. nbatch/nlayers/ncomponents/7 => nmixture/ncomponents/7

    auto n = gpe::get_ns(mixture);
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.dims == 2, "atm only 2d gaussians")
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<float>(), "atm only float")

    const auto n_mixtures = n.batch * n.layers;
    const auto bvh = LBVH(gpe::mixture_with_inversed_covariances(mixture).contiguous());
    const auto n_internal_nodes = bvh.m_n_internal_nodes;
    const auto n_nodes = bvh.m_n_nodes;
    mixture = mixture.view({n_mixtures, n.components, -1}).contiguous();
    auto flat_bvh_nodes = bvh.m_nodes.view({n_mixtures, n_nodes, -1});
    auto flat_bvh_aabbs = bvh.m_aabbs.view({n_mixtures, n_nodes, -1});
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


                                   auto mixture_a = gpe::accessor<scalar_t, 3>(mixture);
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

                                            auto mixture_a = gpe::accessor<scalar_t, 3>(mixture);
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


    return std::make_tuple(out_mixture.view({n.batch, n.layers, n_components_target, -1}), bvh.m_nodes, bvh.m_aabbs);
}


} // namespace bvh_mhem_fit

