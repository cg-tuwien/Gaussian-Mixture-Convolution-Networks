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

// todo:
// - when fitting 1024 components (more than input), mse doesn't go to 0, although the rendering looks good.
// - in collect_result, run a new fitting with the most important node to fill up the remaining gaussian slots
// - kl-divergenece filter: at least one should pass (select from clustering), otherwise we loose mass
// - check integration mass again (after kl-div filter)
// - debug 2nd and 3rd layer: why are we lossing these important gaussians, especially when N_REDUCE is larger (8)?

namespace bvh_mhem_fit {

namespace  {

using node_index_torch_t = lbvh::detail::Node::index_type_torch;
using node_index_t = lbvh::detail::Node::index_type;
using gaussian_index_t = uint16_t;
using gaussian_index_torch_t = int16_t;
using Node  = lbvh::detail::Node;

template <typename scalar_t = float>
struct Epsilon {
    static constexpr scalar_t value = scalar_t(0.0000000000000001);
    static EXECUTION_DEVICES scalar_t clip(scalar_t v) { return gpe::max(v, value); }
};

template<>
struct Epsilon<double> {
    static constexpr double value = 0.000000000000000000001;
    static EXECUTION_DEVICES double clip(double v) { return gpe::max(v, value); }
};


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

    EXECUTION_DEVICES gpe::Vector<Gaussian_type, REDUCTION_N * 2> collect_child_gaussians(const lbvh::detail::Node* node, scalar_t weight_threshold) const {
        assert(node->left_idx != node_index_t(-1));
        assert(node->right_idx != node_index_t(-1));
        gpe::Vector<Gaussian_type, REDUCTION_N * 2> retval;
        auto condition = [weight_threshold](const Gaussian_type& g) { return gpe::abs(g.weight) >= weight_threshold; };
        retval.push_back_if(per_node_attributes[node->left_idx].gaussians, condition);
        retval.push_back_if(per_node_attributes[node->right_idx].gaussians, condition);
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

// numerical problems when N_VIRTUAL_POINTS is large: a*b for instance 0.001, wi_bar becomes 5.6 -> bad things
// that depends on cov magnitude => better normalise mixture to have covs in the magnitude of the identity
template <typename scalar_t, int N_DIMS, int N_VIRTUAL_POINTS = 4>
EXECUTION_DEVICES scalar_t likelihood(const gpe::Gaussian<N_DIMS, scalar_t>& target, const gpe::Gaussian<N_DIMS, scalar_t>& fitting) {
    // Continuous projection for fast L 1 reconstruction: Equation 9
    scalar_t normal_amplitude = gpe::gaussian_amplitude(fitting.covariance);
    scalar_t a = gpe::evaluate(target.position, normal_amplitude, fitting.position, fitting.covariance);
    auto c = glm::inverse(fitting.covariance) * target.covariance;
    scalar_t b = gpe::exp(scalar_t(-0.5) * gpe::trace(c));
    scalar_t target_normal_amplitudes = gpe::gaussian_amplitude(target.covariance);
    scalar_t wi_bar = N_VIRTUAL_POINTS * target.weight / target_normal_amplitudes;
    // pow(0, 0) gives nan in cuda with fast math
    return gpe::pow(Epsilon<scalar_t>::clip(a * b), wi_bar);
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


// todo: gaussian is not necessary in the parameter list. only used for size and size can we get from disparities as well;
template <uint32_t N_CLUSTERS, typename scalar_t, uint32_t N_INPUT>
EXECUTION_DEVICES
gpe::Array<gpe::Vector<gaussian_index_t, N_INPUT - N_CLUSTERS + 1>, N_CLUSTERS> clusterise(const gpe::Vector2d<scalar_t, N_INPUT>& disparities) {
    // this is a greedy smallest spanning subtrees algorithm
    static_assert (N_CLUSTERS <= N_INPUT, "N output clusters must be larger than n input");
    assert(N_CLUSTERS <= disparities.size());
    assert(!gpe::reduce(disparities, false, [](bool o, scalar_t v) { return o || gpe::isnan(v); }));

    gpe::Vector2d<gaussian_index_t, N_INPUT> subgraphs;
    for (unsigned i = 0; i < disparities.size(); ++i) {
        subgraphs.push_back({i});
    }
    unsigned n_subgraphs = subgraphs.size();

    auto merge_subgraphs = [&](unsigned a, unsigned b) {
        assert (a != b);
        auto a_ = gpe::min(a, b);
        auto b_ = gpe::max(a, b);

        subgraphs[a_].push_back(subgraphs[b_]);
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

    auto shortest_edge = [&disparities](gaussian_index_t* a, gaussian_index_t* b, gpe::BitSet<N_INPUT * N_INPUT>* invalid_edges) {
        *a = gaussian_index_t(-1);
        *b = gaussian_index_t(-1);
        scalar_t shortest_length = std::numeric_limits<scalar_t>::infinity();
        for (gaussian_index_t i = 0; i < disparities.size(); ++i) {
            for (gaussian_index_t j = i + 1; j < disparities.size(); ++j) {
                if (!invalid_edges->isSet(i * disparities.size() + j) && disparities[i][j] < shortest_length) {
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
        invalid_edges.set1(a * disparities.size() + b);
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
        retval[i].push_back(subgraphs[subgraph_id]);
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
    if (gpe::abs(new_gaussian.weight) < Epsilon<scalar_t>::value) {
        new_gaussian.covariance = typename G::cov_t(1.0);
        assert(glm::determinant(new_gaussian.covariance) > 0);
    }
    else {
        new_gaussian.position /= new_gaussian.weight;
        new_gaussian.covariance /= new_gaussian.weight;
        // no good (?): very large weight G + small weight G should result in large G and not 1/2 large G
        // subsequently we'll rescale the whole mixture anyways
//        new_gaussian.weight /= scalar_t(cluster_indices.size());
        assert(glm::determinant(new_gaussian.covariance) > 0);
    }
    assert(std::isnan(new_gaussian.weight) == false);
    assert(std::isnan(glm::dot(new_gaussian.position, new_gaussian.position)) == false);
    assert(std::isnan(glm::determinant(new_gaussian.covariance)) == false);

    return new_gaussian;
};

template <typename scalar_t, int N_DIMS, uint32_t N_GAUSSIANS_CAPACITY, uint32_t N_MAX_CLUSTER_ELEMENTS>
EXECUTION_DEVICES
gpe::Gaussian<N_DIMS, scalar_t> averageCluster_corrected(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS_CAPACITY>& mixture,
                                               const gpe::Vector<gaussian_index_t, N_MAX_CLUSTER_ELEMENTS>& cluster_indices) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    G new_gaussian = {scalar_t(0), typename G::pos_t(0), typename G::cov_t(0)};

    assert(cluster_indices.size() > 0);

    for (unsigned i = 0; i < cluster_indices.size(); ++i) {
        auto gaussian_id = cluster_indices[i];
        const auto& gaussian = mixture[gaussian_id];

        assert(new_gaussian.weight == 0 || gpe::sign(new_gaussian.weight) == gpe::sign(gaussian.weight)); // can't merge positive and negative gaussian
        auto weight = gaussian.weight;
        weight /= gpe::gaussian_amplitude(gaussian.covariance);
        new_gaussian.weight += weight;
        new_gaussian.position += weight * gaussian.position;
        assert(glm::determinant(gaussian.covariance) > 0);
        new_gaussian.covariance += weight * gaussian.covariance;
    }
    if (gpe::abs(new_gaussian.weight) < Epsilon<scalar_t>::value) {
        new_gaussian.covariance = typename G::cov_t(1.0);
        assert(glm::determinant(new_gaussian.covariance) > 0);
    }
    else {
        new_gaussian.position /= new_gaussian.weight;
        new_gaussian.covariance /= new_gaussian.weight;
        new_gaussian.weight *= gpe::gaussian_amplitude(new_gaussian.covariance);
        // no good (?): very large weight G + small weight G should result in large G and not 1/2 large G
        // subsequently we'll rescale the whole mixture anyways
//        new_gaussian.weight /= scalar_t(cluster_indices.size());
        assert(glm::determinant(new_gaussian.covariance) > 0);
    }
    assert(std::isnan(new_gaussian.weight) == false);
    assert(std::isnan(glm::dot(new_gaussian.position, new_gaussian.position)) == false);
    assert(std::isnan(glm::determinant(new_gaussian.covariance)) == false);

    return new_gaussian;
};

template <typename scalar_t, int N_DIMS, uint32_t N_GAUSSIANS_CAPACITY, uint32_t N_MAX_CLUSTER_ELEMENTS>
EXECUTION_DEVICES
gpe::Gaussian<N_DIMS, scalar_t> maxOfCluster(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS_CAPACITY>& mixture,
                                             const gpe::Vector<gaussian_index_t, N_MAX_CLUSTER_ELEMENTS>& cluster_indices) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    G new_gaussian = {scalar_t(0), typename G::pos_t(0), typename G::cov_t(0)};
    scalar_t max_abs_weight = 0;
    assert(cluster_indices.size() > 0);

    for (unsigned i = 0; i < cluster_indices.size(); ++i) {
        auto gaussian_id = cluster_indices[i];
        const auto& gaussian = mixture[gaussian_id];
        if (gpe::abs(gaussian.weight) > max_abs_weight) {
            max_abs_weight = gpe::abs(gaussian.weight);
            new_gaussian = gaussian;
        }
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
    abs_integral = Epsilon<scalar_t>::clip(abs_integral);
    if (abs_integral_ptr)
        *abs_integral_ptr = abs_integral;

    return gpe::transform(mixture, [abs_integral](const G& g) { return G{g.weight / abs_integral, g.position, g.covariance}; });
}

#define GPE_DISPARITY_METHOD 2
template <unsigned N_FITTING, typename scalar_t, int N_DIMS, unsigned N_TARGET>
EXECUTION_DEVICES
    gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_FITTING> fit_initial(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_TARGET>& target_double_gmm) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;

//    // enable for testing the tree walking without em
//    // change factor computation below to "auto factor = abs_integral / Epsilon<scalar_t>::clip(result_integral);"
//    scalar_t abs_integral;
//    const gpe::Vector<G, N_TARGET> target_double_gmm = normalise_mixture(target, &abs_integral);

#if GPE_DISPARITY_METHOD == 0
    auto disparity_matrix = gpe::outer_product(target_double_gmm, target_double_gmm, centroid_distance<scalar_t, N_DIMS>);   // returns gpe::Vector<gpe::Vector>
#elif GPE_DISPARITY_METHOD == 1
    auto disparity_matrix = gpe::outer_product(target_double_gmm, target_double_gmm, likelihood<scalar_t, N_DIMS>);   // returns gpe::Vector<gpe::Vector>
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

    const auto clustering = clusterise<N_FITTING>(disparity_matrix);                             // returns gpe::Array<gpe::Vector>
    assert(clustering.size() == N_FITTING);

    gpe::Vector<G, N_FITTING> result;
    for (unsigned i = 0; i < N_FITTING; ++i) {
        result.push_back(averageCluster(target_double_gmm, clustering[i]));
    }
    scalar_t result_integral = integrate_abs_mixture(result);
    // result_integral should be approx 1, since we shouldn't fit on zero mixtures anymore and incoming target_double_gmm is normalised;
    assert(result_integral >= scalar_t(0.1));
    auto factor = scalar_t(1.0) / result_integral;
    for (unsigned i = 0; i < N_FITTING; ++i) {
        result[i].weight *= factor;
    }
    return result;
}

template <unsigned N_FITTING, typename scalar_t, int N_DIMS, unsigned N_TARGET>
EXECUTION_DEVICES
gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_FITTING> fit_em(gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_TARGET> target, const scalar_t kl_div_threshold = scalar_t(2.0)) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using pos_t = typename G::pos_t;
    using cov_t = typename G::cov_t;

    namespace fun = gpe::functors;

    auto has_nan = [](const auto& vec) {
        return gpe::reduce(vec, false, [](bool o, auto v) { return o || gpe::isnan(v); });
    };

    scalar_t abs_integral;
    const auto target_double_gmm = normalise_mixture(target, &abs_integral);
    const auto fitting_double_gmm = fit_initial<N_FITTING>(target_double_gmm);

    const auto likelihood_matrix = gpe::outer_product(target_double_gmm, fitting_double_gmm, likelihood<scalar_t, N_DIMS>);
    // todo: test modification of clamp matrix: at least one row element or x percent row elements should be 1.
    //       rational: otherwise certain target gaussians are not covered at all and resulting fitting is missing mass.
    //       could assign gaussian from the cluster; modulo zero weight gaussians
    const auto kldiv_sign_matrix = gpe::outer_product(target_double_gmm, fitting_double_gmm, [](auto target, auto fitting) {
//        return int(1);
        return (gpe::sign(fitting.weight) == gpe::sign(target.weight)) ? kl_divergence<scalar_t, N_DIMS>(target, fitting) : scalar_t(0);
    });

    auto clamp_matrix = gpe::transform(kldiv_sign_matrix, [kl_div_threshold](scalar_t v) { return v < kl_div_threshold ? scalar_t(1) : scalar_t(0); });
    for (unsigned target_id = 0; target_id < clamp_matrix.size(); ++target_id) {
        auto& row = kldiv_sign_matrix[target_id];
        unsigned best_fitting_id = unsigned(-1);
        scalar_t smallest_value = std::numeric_limits<scalar_t>::infinity();
        for (unsigned fitting_id = 0; fitting_id < row.size(); ++fitting_id) {
            if (row[fitting_id] < smallest_value) {
                smallest_value = row[fitting_id];
                best_fitting_id = fitting_id;
            }
        }
        assert(best_fitting_id < N_FITTING);
        clamp_matrix[target_id][best_fitting_id] = scalar_t(1);  // no change if largest value was > kl_div_threshold.
    }

    const auto clamped_likelihood_matrix = gpe::cwise_fun(likelihood_matrix, clamp_matrix, fun::times<scalar_t>);

    const auto pure_fitting_weights = gpe::transform(fitting_double_gmm, [](const G& g) { return gpe::abs(g.weight) / gpe::gaussian_amplitude(g.covariance); });
    const auto weighted_likelihood_matrix = gpe::cwise_fun(pure_fitting_weights, clamped_likelihood_matrix, fun::times<scalar_t>);
    const auto weighted_likelihood_sum = gpe::reduce_rows(weighted_likelihood_matrix, scalar_t(0), fun::plus<scalar_t>);
    const auto weighted_likelihood_sum_clamped = gpe::transform(weighted_likelihood_sum, Epsilon<scalar_t>::clip);
    const auto responsibilities_1 = gpe::cwise_fun(weighted_likelihood_matrix, weighted_likelihood_sum_clamped, fun::divided_AbyB<scalar_t>);
    assert(!has_nan(responsibilities_1));

    const auto pure_target_weights = gpe::transform(target_double_gmm, [](const G& g) { return gpe::abs(g.weight) / gpe::gaussian_amplitude(g.covariance); });
    const auto responsibilities_2 = gpe::cwise_fun(responsibilities_1, pure_target_weights, fun::times<scalar_t>);
    assert(!has_nan(responsibilities_2));

    const auto newWeights = gpe::reduce_cols(responsibilities_2, scalar_t(0), fun::plus<scalar_t>);

    const auto responsibilities_3 = gpe::cwise_fun(gpe::transform(newWeights, Epsilon<scalar_t>::clip), responsibilities_2, fun::divided_BbyA<scalar_t>);
    assert(!has_nan(responsibilities_3));
    assert(!gpe::reduce(responsibilities_3, false, [](bool o, scalar_t v) { return o || v < 0; }));

    const auto weightedPositions = gpe::cwise_fun(responsibilities_3, target, [](scalar_t r, const G& g) { return r * g.position; });
    const auto newPositions = gpe::reduce_cols(weightedPositions, pos_t(0), fun::plus<pos_t>);
    assert(!has_nan(newPositions));

    const auto targetPositions = gpe::transform(target, [](const G& g){ return g.position; });
    const auto posDiffs = gpe::outer_product(targetPositions, newPositions, fun::minus<pos_t>);
    const auto posDiffsOuter = gpe::transform(posDiffs, [](const pos_t& p) { return glm::outerProduct(p, p); });
    const auto targetCovs = gpe::transform(target, [](const G& g){ return g.covariance; });
    const auto unweightedCovs = gpe::cwise_fun(posDiffsOuter, targetCovs, fun::plus<cov_t>);
    const auto weightedCovs = gpe::cwise_fun(responsibilities_3, unweightedCovs, [](scalar_t r, const cov_t& cov) { return r * cov; });
    auto newCovariances = gpe::reduce_cols(weightedCovs, cov_t(0), fun::plus<cov_t>);
    newCovariances = gpe::cwise_fun(newCovariances, newWeights, [](cov_t cov, scalar_t w) {
        if (w < Epsilon<scalar_t>::value)
            cov += cov_t(1) * Epsilon<scalar_t>::value;
        return cov;
    });
    assert(!has_nan(newCovariances));

    const auto normal_amplitudes = gpe::transform(newCovariances, [](const cov_t& cov) { return gpe::gaussian_amplitude(cov); });
    assert(!has_nan(normal_amplitudes));

    gpe::Vector<G, N_FITTING> result;
    for (unsigned i = 0; i < N_FITTING; ++i) {
        result.push_back(G{newWeights[i] * normal_amplitudes[i] * abs_integral,
                           newPositions[i],
                           newCovariances[i]});
    }

    assert(false);
    assert(gpe::abs(abs_integral - integrate_abs_mixture(result)) < scalar_t(0.0001));
    return result;
}



template <typename scalar_t, int N_DIMS, int REDUCTION_N>
EXECUTION_DEVICES void iterate_over_nodes(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                          const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                          gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                                          const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                          const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                          gpe::PackedTensorAccessor32<int, 2> flags,
                                          gpe::PackedTensorAccessor32<scalar_t, 3> node_attributes,
                                          const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes, unsigned n_components_target)
{
    GPE_UNUSED(gpe_gridDim)
    GPE_UNUSED(n_components_target)
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

        auto child_gaussians = bvh.collect_child_gaussians(node, Epsilon<scalar_t>::value);
        if (child_gaussians.size() > REDUCTION_N) {
            bvh.per_node_attributes[node_id].gaussians = fit_em<REDUCTION_N>(child_gaussians);
        }
        else {
            bvh.per_node_attributes[node_id].gaussians.push_back(child_gaussians);

        }
    }
}

template <typename scalar_t, int N_DIMS, int REDUCTION_N, int N_MAX_TARGET_COMPS = 1024>
EXECUTION_DEVICES void collect_result(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                      const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                      const gpe::PackedTensorAccessor32<scalar_t, 3> mixture,
                                      gpe::PackedTensorAccessor32<scalar_t, 3> out_mixture,
                                      const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                      const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                      gpe::PackedTensorAccessor32<int, 2> flags,
                                      gpe::PackedTensorAccessor32<scalar_t, 3> node_attributes,
                                      const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes, unsigned n_components_target)
{
    GPE_UNUSED(gpe_gridDim)
    GPE_UNUSED(flags);
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;

    assert(n_components_target % REDUCTION_N == 0);
    assert(n_components_target <= N_MAX_TARGET_COMPS);
    static_assert (N_MAX_TARGET_COMPS % REDUCTION_N == 0, "N_MAX_TARGET_COMPS must be divisible by REDUCTION_N");

    const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
    if (mixture_id >= n_mixtures)
        return;

    Bvh bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>(mixture_id, nodes, aabbs, mixture, node_attributes, n, n_internal_nodes, n_nodes);

    gpe::Vector<scalar_t, N_MAX_TARGET_COMPS> selectedNodesRating;
    gpe::Vector<node_index_t, N_MAX_TARGET_COMPS> selectedNodes;

    unsigned n_selected_components = 0;
    auto compute_rating = [&](node_index_t node_id) {
        assert(node_id < n_nodes);
        // todo: will break with negative weights, should compute sum of abs integrals / seperately positive and negative integrals
        if (bvh.per_node_attributes[node_id].gaussians.size() < REDUCTION_N)
            return scalar_t(-2); // -2 so it's safely below -1 from cach_id_with_highest_rating
        else
            return std::abs(bvh.per_node_attributes[node_id].gm_integral);
    };
    auto cach_id_with_highest_rating = [&]() {
        scalar_t rating = -1;
        unsigned best_node_id = unsigned(-1);
        for (unsigned i = 0; i < selectedNodes.size(); ++i) {
            if (selectedNodesRating[i] > rating) {
                rating = selectedNodesRating[i];
                best_node_id = i;
            }
        }
        // can become unsigned(-1) when no selectable node remains
        return best_node_id;
    };
    selectedNodes.push_back(0); // root node
    selectedNodesRating.push_back(compute_rating(0));
    n_selected_components = bvh.per_node_attributes[0].gaussians.size();

    while (n_selected_components < n_components_target - REDUCTION_N)  {
        auto best_node_cache_id = cach_id_with_highest_rating();
        if (best_node_cache_id >= selectedNodes.size())
            break;  // ran out of nodes
        auto best_node_id = selectedNodes[best_node_cache_id];
        assert(best_node_id < n_nodes);
        const auto& best_descend_node = bvh.nodes[best_node_id];
        assert(best_node_id < n_internal_nodes); // we should have only internal nodes at this point as cach_id_with_highest_rating() returns 0xffff.. if the node is not full.

        selectedNodes[best_node_cache_id] = best_descend_node.left_idx;
        selectedNodesRating[best_node_cache_id] = compute_rating(best_descend_node.left_idx);

        selectedNodes.push_back(best_descend_node.right_idx);
        selectedNodesRating.push_back(compute_rating(best_descend_node.right_idx));
        n_selected_components = n_selected_components - REDUCTION_N + bvh.per_node_attributes[best_descend_node.left_idx].gaussians.size() + bvh.per_node_attributes[best_descend_node.right_idx].gaussians.size();
    }

//    if (n_selected_components < n_components_target) {
//        printf("n_selected_components = %d / %d\n", n_selected_components, n_components_target);
//    }

    // copy gaussians to their final location in out_mixture
    unsigned write_position = 0;
    for (unsigned i = 0; i < selectedNodes.size(); ++i) {
        auto node_id = selectedNodes[i];
        typename Bvh::NodeAttributes& destination_attribute = bvh.per_node_attributes[node_id];

        for (unsigned j = 0; j < destination_attribute.gaussians.size(); ++j) {
            assert(write_position < n_components_target);
            gpe::gaussian<N_DIMS>(out_mixture[mixture_id][int(write_position++)]) = destination_attribute.gaussians[j];
        }
    }
}


} // anonymous namespace


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> forward_impl(at::Tensor mixture, unsigned n_components_target = 32) {
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

