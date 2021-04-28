#include "bvh_mhem_fit_alpha/implementation.h"
#include <stdio.h>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "bvh_mhem_fit_alpha/implementation_common.h"
#include "common.h"
#include "cuda_operations.h"
#include "cuda_qt_creator_definitinos.h"
#include "hacked_accessor.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "parallel_start.h"
#include "util/algorithms.h"
#include "util/autodiff.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/glm.h"
#include "util/mixture.h"
#include "util/scalar.h"

// todo:
// - in collect_result, run a new fitting with the most important node to fill up the remaining gaussian slots

namespace bvh_mhem_fit_alpha {

namespace  {

// todo: min of KL divergencies is probably a better distance
template <typename scalar_t, int N_DIMS>
EXECUTION_DEVICES scalar_t cluster_centroid_distance(const gpe::Gaussian<N_DIMS, scalar_t>& a, const gpe::Gaussian<N_DIMS, scalar_t>& b) {
    if (gpe::sign(a.weight) != gpe::sign(b.weight))
        return std::numeric_limits<scalar_t>::infinity();
    if (gpe::abs(a.weight) < gpe::Epsilon<scalar_t>::large || gpe::abs(b.weight) < gpe::Epsilon<scalar_t>::large)
        return std::numeric_limits<scalar_t>::infinity();
    return gpe::squared_norm(a.position - b.position);
}


template <uint32_t N_CLUSTERS, typename scalar_t, uint32_t N_INPUT>
EXECUTION_DEVICES
gpe::Array<gpe::Vector<gaussian_index_t, N_INPUT - N_CLUSTERS + 1>, N_CLUSTERS> clusterise_using_heap(const gpe::Array2d<scalar_t, N_INPUT>& disparities,
                                                                                                      const gpe::Vector<gaussian_index_t, N_INPUT>& valid_gaussians) {
    // this is a greedy smallest spanning subtrees algorithm
    static_assert (N_CLUSTERS <= N_INPUT, "N output clusters must be larger than n input");
    assert(N_CLUSTERS <= disparities.size());
    assert(!gpe::reduce(disparities, false, [](bool o, scalar_t v) { return o || gpe::isnan(v); }));
    const auto n_gaussians = valid_gaussians.size();

    gpe::Vector2d<gaussian_index_t, N_INPUT> subgraphs;
    for (gaussian_index_t i = 0; i < valid_gaussians.size(); ++i) {
        subgraphs.push_back({valid_gaussians[i]});
    }
    unsigned n_subgraphs = subgraphs.size();
    // make disparities into an array
    // first put all the overflow gaussians into cluster 0 (they are zero weight, so it doesn't matter which
//    for (gaussian_index_t i = n_gaussians; i < N_INPUT; ++i) {
//        subgraphs[0].push_back({i});
//    }
    // then copy the disparities, filling up with infty (so they won't get selected)
    struct DisparityData {
        scalar_t disparity;
        gaussian_index_t idx_a;
        gaussian_index_t idx_b;
        EXECUTION_DEVICES
        bool operator <= (const DisparityData& other) const { return disparity <= other.disparity; }
    };

    gpe::ArrayHeap<DisparityData, (N_INPUT * N_INPUT - N_INPUT) / 2> disparity_heap;
    const auto invalid_disparity = DisparityData{std::numeric_limits<scalar_t>::infinity(), gaussian_index_t(-1), gaussian_index_t(-1)};
    unsigned n_disparities = 0;
    for (gaussian_index_t i = 0; i < n_gaussians; ++i) {
        for (gaussian_index_t j = i + 1; j < n_gaussians; ++j) {
            disparity_heap.m_data[n_disparities] = DisparityData{disparities[valid_gaussians[i]][valid_gaussians[j]], valid_gaussians[i], valid_gaussians[j]};
            ++n_disparities;
        }
    }
    // set remaining disparities to infinity, so they won't be selected.
    for (unsigned i = n_disparities; i < (N_INPUT * N_INPUT - N_INPUT) / 2; ++i) {
        disparity_heap.m_data[i] = invalid_disparity;
    }
    disparity_heap.build();

    auto merge_subgraphs = [&](unsigned a, unsigned b) {
        assert (a != b);
        assert(a < n_gaussians);
        assert(b < n_gaussians); // smaller than n_gaussians in target

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

    while (n_subgraphs > N_CLUSTERS) {
        auto current_dispairty = disparity_heap.replaceRoot(invalid_disparity);
        assert(current_dispairty.disparity != invalid_disparity.disparity);
        auto subgraph_a = subgraph_of(current_dispairty.idx_a);
        auto subgraph_b = subgraph_of(current_dispairty.idx_b);
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
        retval[i].push_back_if(subgraphs[subgraph_id], [=](gaussian_index_t idx) { return idx < n_gaussians; });
    }

    return retval;
}


template <typename scalar_t, int N_DIMS, uint32_t N_GAUSSIANS, uint32_t N_MAX_CLUSTER_ELEMENTS>
EXECUTION_DEVICES
gaussian_index_t cluster_select_maxWeight(const gpe::Array<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS>& mixture,
                                          const gpe::Vector<gaussian_index_t, N_MAX_CLUSTER_ELEMENTS>& cluster_indices) {
    gaussian_index_t selected_index = gaussian_index_t(-1);
    scalar_t max_abs = 0;
    assert(cluster_indices.size() > 0);

    for (unsigned i = 0; i < cluster_indices.size(); ++i) {
        auto gaussian_id = cluster_indices[i];
        assert(gaussian_id < mixture.size());
        const auto& weight = mixture[gaussian_id].weight;
        if (gpe::abs(weight) > max_abs) {
            max_abs = gpe::abs(weight);
            selected_index = gaussian_id;
        }
    }
    assert(selected_index != gaussian_index_t(-1));
    return selected_index;
};


template <unsigned N_FITTING, typename scalar_t, int N_DIMS, unsigned N_TARGET>
EXECUTION_DEVICES
gpe::Array<gaussian_index_t, N_FITTING> fit_initial(const gpe::Array<gpe::Gaussian<N_DIMS, scalar_t>, N_TARGET>& target, const Config& config) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;

    auto disparity_matrix = gpe::outer_product(target, target, cluster_centroid_distance<gradless_scalar_t, N_DIMS>);

    gpe::Vector<gaussian_index_t, N_TARGET> valid_gaussians;
    for (gaussian_index_t i = 0; i < N_TARGET; ++i) {
        if (gpe::abs(target[i].weight) >= gpe::Epsilon<scalar_t>::large)
            valid_gaussians.push_back(i);
    }
    const auto clustering = clusterise_using_heap<N_FITTING>(disparity_matrix, valid_gaussians);                             // returns gpe::Array<gpe::Vector>
    assert(clustering.size() == N_FITTING);

    gpe::Array<gaussian_index_t, N_FITTING> result;
    for (unsigned i = 0; i < N_FITTING; ++i) {
        result[i] = (cluster_select_maxWeight(target, clustering[i]));
    }
    return result;
}

template <unsigned N_FITTING, typename scalar_t, int N_DIMS, unsigned N_TARGET>
EXECUTION_DEVICES
gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_FITTING> fit_em(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_TARGET>& target,
                                                               GradientCacheData<scalar_t, N_FITTING, N_FITTING * 2>* gradient_cache_data,
                                                               const Config& config) {
    static_assert (N_FITTING * 2 == N_TARGET, "UNEXPECTED N_TARGET or N_FITTING");
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using pos_t = typename G::pos_t;
    using cov_t = typename G::cov_t;
    using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;
    using GradlessG = gpe::Gaussian<N_DIMS, gradless_scalar_t>;
    using gradless_pos_t = typename GradlessG::pos_t;
    using gradless_cov_t = typename GradlessG::cov_t;
    #ifndef __CUDACC__
    static_assert (std::is_same_v<gradless_scalar_t, float> || std::is_same_v<gradless_scalar_t, double>, "gradless should be either float or double");
    #endif

    namespace fun = gpe::functors;

    auto has_nan = [](const auto& vec) {
        return gpe::reduce(vec, false, [](bool o, auto v) { return o || gpe::isnan(v); });
    };

    const auto target_mixture = gpe::to_array(target, G{0, pos_t(0), cov_t(1)});

    const auto target_weights = gpe::transform(target_mixture, [](const G& g){ return scalar_t(g.weight); });
    const auto target_positions = gpe::transform(target_mixture, [](const G& g){ return g.position; });
    const auto target_covariances = gpe::transform(target_mixture, [](const G& g){ return g.covariance; });

    const auto target_component_integrals = gpe::transform(target_mixture, gpe::integrate<scalar_t, N_DIMS>);

    // todo: fix this for mixtures containing negative weights
    const scalar_t target_integral = gpe::reduce(target_component_integrals, scalar_t(0), fun::plus<scalar_t, scalar_t, scalar_t>);
    const scalar_t target_clipped_integral = gpe::Epsilon<scalar_t>::clip(target_integral);

    const auto target_int1_weights = gpe::cwise_fun(target_weights, target_clipped_integral, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);  // target_int1_weights equal grad // target_weights not equal grad
    const auto initial_indices = fit_initial<N_FITTING>(gpe::removeGrad(target_mixture), config);
    gradient_cache_data->initial_indices = initial_indices;
    const auto initial_mixture = gpe::select(target_mixture, initial_indices);
    const auto initial_weights = gpe::transform(gpe::select(target_weights, initial_indices), [](scalar_t v) { return v; });
    const auto initial_positions = gpe::select(target_positions, initial_indices);
    const auto initial_covariances = gpe::select(target_covariances, initial_indices);
    const auto initial_component_integrals = gpe::transform(initial_mixture, gpe::integrate<scalar_t, N_DIMS>);
    const auto initial_integral = gpe::reduce(initial_component_integrals, scalar_t(0), fun::plus<scalar_t>);
    const auto initial_clipped_integral = gpe::Epsilon<scalar_t>::clip(initial_integral);
    const auto initial_int1_weights = gpe::cwise_fun(initial_weights, initial_clipped_integral, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    const auto initial_int1_mixture = gpe::pack_mixture(initial_int1_weights, initial_positions, initial_covariances);
    const auto initial_gaussian_amplitudes = gpe::transform(initial_covariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    const auto initial_pure_weights = gpe::cwise_fun(initial_int1_weights, initial_gaussian_amplitudes, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);

    const auto target_int1_mixture = gpe::pack_mixture(target_int1_weights, target_positions, target_covariances);

    const auto likelihood_matrix = gpe::outer_product(target_int1_mixture, initial_int1_mixture, gpe::likelihood<scalar_t, N_DIMS>);
    const auto kldiv_sign_matrix = gpe::outer_product(gpe::removeGrad(target_int1_mixture), gpe::removeGrad(initial_int1_mixture), [](auto target, auto fitting) {
        return (gpe::sign(fitting.weight) == gpe::sign(target.weight)) ? gpe::kl_divergence<gradless_scalar_t, N_DIMS>(target, fitting) : gradless_scalar_t(0);
    });

    auto kl_div_threshold = gradless_scalar_t(config.em_kl_div_threshold);
    auto clamp_matrix = gpe::transform(kldiv_sign_matrix, [kl_div_threshold](gradless_scalar_t v) { return v < kl_div_threshold ? gradless_scalar_t(1) : gradless_scalar_t(0); });
    for (unsigned target_id = 0; target_id < clamp_matrix.size(); ++target_id) {
        auto& row = kldiv_sign_matrix[target_id];
        unsigned best_fitting_id = unsigned(-1);
        auto smallest_value = std::numeric_limits<gradless_scalar_t>::infinity();
        for (unsigned fitting_id = 0; fitting_id < row.size(); ++fitting_id) {
            if (row[fitting_id] < smallest_value) {
                smallest_value = row[fitting_id];
                best_fitting_id = fitting_id;
            }
        }
        assert(best_fitting_id < N_FITTING);
        clamp_matrix[target_id][best_fitting_id] = gradless_scalar_t(1);  // no change if largest value was > kl_div_threshold.
    }

    const auto weighted_likelihood_matrix = gpe::cwise_fun(initial_pure_weights, likelihood_matrix, fun::times<scalar_t, scalar_t, scalar_t>);
    const auto weighted_likelihood_matrix_clipped = gpe::transform(weighted_likelihood_matrix, gpe::Epsilon<scalar_t>::clip);
    const auto weighted_likelihood_clamped_matrix = gpe::cwise_fun(weighted_likelihood_matrix_clipped, clamp_matrix, fun::times<scalar_t, scalar_t, scalar_t>);
    const auto weighted_likelihood_sum = gpe::reduce_rows(weighted_likelihood_clamped_matrix, scalar_t(0), fun::plus<scalar_t, scalar_t, scalar_t>);
    const auto responsibilities_1 = gpe::cwise_fun(weighted_likelihood_clamped_matrix, weighted_likelihood_sum, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    assert(!has_nan(responsibilities_1));

    const auto target_gaussian_amplitudes = gpe::transform(target_covariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    const auto pure_target_weights = gpe::cwise_fun(target_int1_weights, target_gaussian_amplitudes, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    const auto responsibilities_2 = gpe::cwise_fun(responsibilities_1, pure_target_weights, fun::times<scalar_t, scalar_t, scalar_t>);
    gradient_cache_data->responsibilities_2 = gpe::removeGrad(responsibilities_2);
    assert(!has_nan(responsibilities_2));

    const auto fitting_pure_weights = gpe::reduce_cols(responsibilities_2, scalar_t(0), fun::plus<scalar_t, scalar_t, scalar_t>);
    const auto clippedFittingWeights = gpe::transform(fitting_pure_weights, gpe::Epsilon<scalar_t>::clip);

    const auto responsibilities_3 = gpe::cwise_fun(clippedFittingWeights, responsibilities_2, fun::divided_BbyA<scalar_t, scalar_t, scalar_t>);
    gradient_cache_data->responsibilities_3 = gpe::removeGrad(responsibilities_3);
    assert(!has_nan(responsibilities_3));
    assert(!gpe::reduce(responsibilities_3, false, [](bool o, scalar_t v) { return o || v < 0; }));

    const auto weightedPositions = gpe::cwise_fun(responsibilities_3, target_positions, fun::times<scalar_t, pos_t, pos_t>);
    const auto fittingPositions = gpe::reduce_cols(weightedPositions, pos_t(0), fun::plus<pos_t, pos_t, pos_t>);
    assert(!has_nan(fittingPositions));

    const auto posDiffs = gpe::outer_product(target_positions, fittingPositions, fun::minus<pos_t>);
    const auto posDiffsOuter = gpe::cwise_fun(posDiffs, posDiffs, gpe::outerProduct<scalar_t, N_DIMS>);
    const auto unweightedCovs = gpe::cwise_fun(posDiffsOuter, target_covariances, fun::plus<cov_t, cov_t, cov_t>);

    const auto weightedCovs = gpe::cwise_fun(responsibilities_3, unweightedCovs, fun::times<scalar_t, cov_t, cov_t>);

    auto fittingCovariances = gpe::reduce_cols(weightedCovs, cov_t(0), fun::plus<cov_t, cov_t, cov_t>);
    fittingCovariances = gpe::cwise_fun(fittingCovariances, fitting_pure_weights, [](cov_t cov, scalar_t w) {  // no influence on gradient.
        if (w < gpe::Epsilon<scalar_t>::large)
            return cov_t(1);
        return cov;
    });
    assert(!has_nan(fittingCovariances));

    const auto fitting_normal_amplitudes = gpe::transform(fittingCovariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    const auto fitting_int1_weights = gpe::cwise_fun(fitting_pure_weights, fitting_normal_amplitudes, fun::times<scalar_t, scalar_t, scalar_t>);
//    const auto finalFittingWeights = gpe::transform(int1_final_fitting_weights, [&abs_integral](scalar_t v) { return scalar_t(v * abs_integral); });
    const auto fitting_weights = gpe::cwise_fun(fitting_int1_weights, target_clipped_integral, fun::times<scalar_t, scalar_t, scalar_t>);

//    float grad_weights[] = {0.7f, 1.3f};
    gradless_scalar_t grad_weights[] = {1.0, 1.0};
//    for (unsigned i = 0; i < 4; ++i) {
//        for (unsigned j = 0; j < 2; ++j) {
//            autodiff::Variable<float>(fittingCovariances[j][0][0]).expr->propagate(1);
//            autodiff::Variable<float>(fittingCovariances[j][0][1]).expr->propagate(1);
//            autodiff::Variable<float>(fittingCovariances[j][1][0]).expr->propagate(1);
//            autodiff::Variable<float>(fittingCovariances[j][1][1]).expr->propagate(1);
//        }
//    }
    bool test_propagate_grad = false;
    gpe::Vector<G, N_FITTING> result;
    for (unsigned i = 0; i < N_FITTING; ++i) {
        result.push_back(G{fitting_weights[i],
                           fittingPositions[i],
                           fittingCovariances[i]});
        if (test_propagate_grad) {
            autodiff::Variable<gradless_scalar_t>(fitting_weights[i]).expr->propagate(1);

            autodiff::Variable<gradless_scalar_t>(fittingPositions[i][0]).expr->propagate(1);
            autodiff::Variable<gradless_scalar_t>(fittingPositions[i][1]).expr->propagate(1);

            autodiff::Variable<gradless_scalar_t>(fittingCovariances[i][0][0]).expr->propagate(1);
            autodiff::Variable<gradless_scalar_t>(fittingCovariances[i][0][1]).expr->propagate(1);
            autodiff::Variable<gradless_scalar_t>(fittingCovariances[i][1][0]).expr->propagate(1);
            autodiff::Variable<gradless_scalar_t>(fittingCovariances[i][1][1]).expr->propagate(1);
        }
    }
//#ifdef GPE_AUTODIFF
//    auto grad_initial_mixture = gpe::extractGrad(initial_mixture);
//    auto grad_initial_weights = gpe::extractGrad(initial_weights);
//    auto grad_initial_int1_weights = gpe::extractGrad(initial_int1_weights);
//    auto grad_initial_positions = gpe::extractGrad(initial_positions);
//    auto grad_initial_covariances = gpe::extractGrad(initial_covariances);
//    auto grad_target_weights = gpe::extractGrad(target_weights);
//    auto grad_target_int1_weights = gpe::extractGrad(target_int1_weights);
//    auto grad_target_positions = gpe::extractGrad(target_positions);
//    auto grad_target_covariances = gpe::extractGrad(target_covariances);
//    auto grad_target_mixture = gpe::extractGrad(target_mixture);
//    auto grad_target_component_integrals = gpe::extractGrad(target_component_integrals);
//    auto grad_target_int1_mixture = gpe::extractGrad(target_int1_mixture);
//    auto grad_initial_gaussian_amplitudes = gpe::extractGrad(initial_gaussian_amplitudes);

//    auto grad_likelihood_matrix = gpe::extractGrad(likelihood_matrix);
//    auto grad_weighted_likelihood_matrix = gpe::extractGrad(weighted_likelihood_matrix);
//    auto grad_weighted_likelihood_sum = gpe::extractGrad(weighted_likelihood_sum);
//    auto grad_responsibilities_1 = gpe::extractGrad(responsibilities_1);
//    auto grad_target_gaussian_amplitudes = gpe::extractGrad(target_gaussian_amplitudes);
//    auto grad_responsibilities_2 = gpe::extractGrad(responsibilities_2);
//    auto grad_fitting_pure_weights = gpe::extractGrad(fitting_pure_weights);
//    auto grad_responsibilities_3 = gpe::extractGrad(responsibilities_3);

//    auto grad_fittingPositions = gpe::extractGrad(fittingPositions);
//    auto grad_fittingCovariances = gpe::extractGrad(fittingCovariances);
//    auto grad_weightedCovs = gpe::extractGrad(weightedCovs);
//    auto grad_unweightedCovs = gpe::extractGrad(unweightedCovs);
//    auto grad_posDiffsOuter = gpe::extractGrad(posDiffsOuter);
//    auto grad_fitting_normal_amplitudes = gpe::extractGrad(fitting_normal_amplitudes);
//#endif



//    if (gpe::abs(abs_integral - gpe::reduce(result, scalar_t(0), [](scalar_t i, const G& g) { return i + gpe::abs(gpe::integrate(g)); })) >= scalar_t(0.0001)) {
//        auto intabsmixres = gpe::reduce(result, scalar_t(0), [](scalar_t i, const G& g) { return i + gpe::abs(gpe::integrate(g)); });
//        printf("target:\n");
//        for (const auto& g : target_double_gmm) {
//            gpe::printGaussian(g);
//        }
//        printf("initial fitting:\n");
//        for (const auto& g : fitting_double_gmm) {
//            gpe::printGaussian(g);
//        }
//        printf("result:\n");
//        for (const auto& g : result) {
//            gpe::printGaussian(g);
//        }
//#ifndef __CUDA_ARCH__
//        fflush(stdout);
//#endif
//        assert(false);
//    }
    assert(gpe::abs(target_clipped_integral - gpe::reduce(result, scalar_t(0), [](scalar_t i, const G& g) { return i + gpe::abs(gpe::integrate(g)); })) < gradless_scalar_t(0.0001));
    return result;
}

template <typename scalar_t, int N_DIMS, int REDUCTION_N>
EXECUTION_DEVICES
void iterate_over_nodes(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                        const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                        const gpe::PackedTensorAccessor32<gpe::Gaussian<N_DIMS, scalar_t>, 2> mixture,
                        const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                        const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                        gpe::PackedTensorAccessor32<int, 2> flags,
                        gpe::PackedTensorAccessor32<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes, 2> node_attributes,
                        const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes,
                        const Config& config) {
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;

    assert(gpe_blockDim.y == 1);
    assert(gpe_blockDim.z == 1);
    const auto mixture_id = int(gpe_blockIdx.y);
    assert(mixture_id < n_mixtures);

    Bvh bvh = Bvh(mixture_id, nodes, aabbs, mixture, node_attributes, n, n_internal_nodes, n_nodes);

    const unsigned leaves_per_thread = (unsigned(n.components) + gpe_blockDim.x - 1) / gpe_blockDim.x;
    const unsigned begin_leaf = leaves_per_thread * gpe_threadIdx.x;
    const unsigned end_leaf = gpe::min(begin_leaf + leaves_per_thread, unsigned(n.components));
    unsigned current_leaf = begin_leaf;

    auto is_second_thread = [&flags, mixture_id](node_index_t index) {
        auto* flag = &reinterpret_cast<int&>(flags[mixture_id][index]);
        auto old = gpe::atomicCAS(flag, 0, 1);
        return bool(old);
    };

    // go bottom up through all nodes
    while(current_leaf < end_leaf)
    {
        const auto leaf_node_id = node_index_t(current_leaf + n_internal_nodes);
        assert(leaf_node_id < n_nodes);
        const Node* node = &bvh.nodes[leaf_node_id];

        const G& leaf_gaussian = bvh.gaussians[node->object_idx];
        bvh.per_node_attributes[leaf_node_id].gaussians.push_back(leaf_gaussian);
        bvh.per_node_attributes[leaf_node_id].n_child_leaves = 1;
        bvh.per_node_attributes[leaf_node_id].gm_integral = gpe::integrate(leaf_gaussian);
        current_leaf++;

        while (node->parent_idx != node_index_t(0xFFFFFFFF) && is_second_thread(node->parent_idx)) {
            auto node_id = node->parent_idx;
            node = &bvh.nodes[node_id];
            bvh.per_node_attributes[node_id].n_child_leaves = bvh.per_node_attributes[node->left_idx].n_child_leaves + bvh.per_node_attributes[node->right_idx].n_child_leaves;
            bvh.per_node_attributes[node_id].gm_integral = bvh.per_node_attributes[node->left_idx].gm_integral + bvh.per_node_attributes[node->right_idx].gm_integral;

            auto child_gaussians = bvh.collect_child_gaussians(node, gpe::Epsilon<scalar_t>::large);
            if (child_gaussians.size() > REDUCTION_N) {
                bvh.per_node_attributes[node_id].gaussians = fit_em<REDUCTION_N>(child_gaussians, &(bvh.per_node_attributes[node_id].gradient_cache_data), config);
            }
            else {
                bvh.per_node_attributes[node_id].gaussians.push_back(child_gaussians);
            }
        }
    }
}

template <typename scalar_t, int N_DIMS, int REDUCTION_N, int N_MAX_TARGET_COMPS = 1024>
EXECUTION_DEVICES void collect_result(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                      const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                      const gpe::PackedTensorAccessor32<gpe::Gaussian<N_DIMS, scalar_t>, 2> mixture,
                                      gpe::PackedTensorAccessor32<gpe::Gaussian<N_DIMS, scalar_t>, 2> out_mixture,
                                      const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                      const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                      gpe::PackedTensorAccessor32<int, 2> flags,
                                      gpe::PackedTensorAccessor32<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes, 2> node_attributes,
                                      const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes,
                                      const Config& config)
{
    GPE_UNUSED(gpe_gridDim)
    GPE_UNUSED(flags)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;

    assert(config.n_components_fitting % REDUCTION_N == 0);
    assert(config.n_components_fitting <= N_MAX_TARGET_COMPS);
    static_assert (N_MAX_TARGET_COMPS % REDUCTION_N == 0, "N_MAX_TARGET_COMPS must be divisible by REDUCTION_N");

    const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
    if (mixture_id >= n_mixtures)
        return;

    Bvh bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>(mixture_id, nodes, aabbs, mixture, node_attributes, n, n_internal_nodes, n_nodes);

    gpe::Vector<scalar_t, N_MAX_TARGET_COMPS> selectedNodesRating;
    gpe::Vector<node_index_t, N_MAX_TARGET_COMPS> selectedNodes;

    unsigned n_selected_components = 0;
    auto compute_rating = [&](node_index_t node_id) -> scalar_t {
        assert(node_id < n_nodes);
        // todo: will break with negative weights, should compute sum of abs integrals / seperately positive and negative integrals
        if (bvh.per_node_attributes[node_id].gaussians.size() < REDUCTION_N)
            return -2; // -2 so it's safely below -1 from cach_id_with_highest_rating
        else
            return gpe::abs(bvh.per_node_attributes[node_id].gm_integral);
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

    while (n_selected_components < config.n_components_fitting - REDUCTION_N)  {
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
            assert(write_position < config.n_components_fitting);
            out_mixture[mixture_id][int(write_position++)] = destination_attribute.gaussians[j];
        }
    }
    for (unsigned i = write_position; i < config.n_components_fitting; ++i) {
        out_mixture[mixture_id][int(i)] = G{0, (typename G::pos_t)(0), (typename G::cov_t)(1)};
    }
}


} // anonymous namespace


template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
ForwardOutput forward_impl_t(at::Tensor mixture, const Config& config) {
    using namespace torch::indexing;
    using LBVH = lbvh::Bvh<N_DIMS, scalar_t>;

    // todo: flatten mixture for kernel, i.g. nbatch/nlayers/ncomponents/7 => nmixture/ncomponents/7

    auto n = gpe::get_ns(mixture);
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.components < 65535, "number of components must be smaller than 65535 for morton code computation")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")

    const auto n_mixtures = n.batch * n.layers;
    const LBVH bvh = LBVH(gpe::mixture_with_inversed_covariances(mixture).contiguous(), config.bvh_config);
    const auto n_internal_nodes = bvh.m_n_internal_nodes;
    const auto n_nodes = bvh.m_n_nodes;
    mixture = mixture.view({n_mixtures, n.components, -1}).contiguous();
    auto flat_bvh_nodes = bvh.m_nodes.view({n_mixtures, n_nodes, -1});
    auto flat_bvh_aabbs = bvh.m_aabbs.view({n_mixtures, n_nodes, -1});
    auto flag_container = torch::zeros({n_mixtures, n_internal_nodes}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Int));

    auto flags_a = gpe::accessor<int, 2>(flag_container);
    auto node_attributes = torch::zeros({n_mixtures, n_nodes, sizeof(typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes)}, torch::TensorOptions(mixture.device()).dtype(torch::ScalarType::Byte));

    auto mixture_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 2, scalar_t>(mixture);
    auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(flat_bvh_nodes);
    auto aabbs_a = gpe::accessor<scalar_t, 3>(flat_bvh_aabbs);
    auto node_attributes_a = gpe::struct_accessor<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes, 2, uint8_t>(node_attributes);

    {
        dim3 dimBlock = dim3(32, 1, 1);
        dim3 dimGrid = dim3(uint(1),
                            (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
                            (uint(1) + dimBlock.z - 1) / dimBlock.z);

        auto fun = [mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
            iterate_over_nodes<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                              mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                              n, n_mixtures, n_internal_nodes, n_nodes,
                                                              config);
        };
        gpe::start_serial(gpe::device(mixture), dimGrid, dimBlock, fun);
    }

    auto out_mixture = torch::empty({n_mixtures, config.n_components_fitting, mixture.size(-1)}, torch::TensorOptions(mixture.device()).dtype(mixture.dtype()));
    auto out_mixture_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 2, scalar_t>(out_mixture);

    // make it valid, in case something doesn't get filled (due to an inbalance of the tree or just not enough elements)
    {
        dim3 dimBlock = dim3(32, 1, 1);
        dim3 dimGrid = dim3((uint(n_mixtures) + dimBlock.x - 1) / dimBlock.x, 1, 1);

        auto fun = [mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
            collect_result<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                          mixture_a, out_mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                          n, n_mixtures, n_internal_nodes, n_nodes,
                                                          config);
        };
        gpe::start_serial(gpe::device(mixture), dimGrid, dimBlock, fun);
    }

    return ForwardOutput{out_mixture.view({n.batch, n.layers, config.n_components_fitting, -1}),
                         mixture.view({n.batch, n.layers, n.components, -1}),
                         bvh.m_mixture, bvh.m_nodes, bvh.m_aabbs, node_attributes};
}


} // namespace bvh_mhem_fit

