#include "bvh_mhem_fit/implementation.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "bvh_mhem_fit/implementation_common.h"
#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
#include "hacked_accessor.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "parallel_start.h"
#include "ParallelStack.h"
#include "util/algorithms.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/grad/algorithms.h"
#include "util/grad/glm.h"
#include "util/grad/mixture.h"
#include "util/mixture.h"


// todo:
// - in collect_result, run a new fitting with the most important node to fill up the remaining gaussian slots

namespace bvh_mhem_fit {

namespace  {


template <unsigned N_FITTING, typename scalar_t, int N_DIMS, unsigned N_TARGET, typename size_type>
EXECUTION_DEVICES
gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_TARGET> grad_em(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_TARGET>& target,
                                                               const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_FITTING, size_type>& fitting,
                                                               const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_FITTING, size_type>& fitting_grad,
                                                               const GradientCacheData<scalar_t, N_FITTING, N_FITTING * 2>& gradient_cache_data,
                                                               const BvhMhemFitConfig& config) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using pos_t = typename G::pos_t;
    using cov_t = typename G::cov_t;

    namespace fun = gpe::functors;
    namespace gradfun = gpe::grad::functors;

    auto has_nan = [](const auto& vec) {
        return gpe::reduce(vec, false, [](bool o, auto v) { return o || gpe::isnan(v); });
    };

    // input and result
    auto target_array = gpe::to_array(target, G{0, pos_t(0), cov_t(1)});
    auto fitting_array = gpe::to_array(fitting, G{0, pos_t(0), cov_t(1)});
    auto fitting_grad_array = gpe::to_array(fitting_grad, G{0, pos_t(0), cov_t(0)});

    const auto grad_finalFittingWeights = gpe::transform(fitting_grad_array, [](const G& g){ return g.weight; });
    const auto grad_fittingPositions = gpe::transform(fitting_grad_array, [](const G& g){ return g.position; });
    auto grad_fittingCovariances = gpe::transform(fitting_grad_array, [](const G& g){ return g.covariance; });

    // forward cached
    const auto finalFittingWeights = gpe::transform(fitting_array, [](const G& g){ return g.weight; });
    const auto fittingPositions = gpe::transform(fitting_array, [](const G& g){ return g.position; });
    const auto fittingCovariances = gpe::transform(fitting_array, [](const G& g){ return g.covariance; });

    const auto targetWeights = gpe::transform(target_array, [](const G& g){ return g.weight; });
    const auto targetPositions = gpe::transform(target_array, [](const G& g){ return g.position; });
    const auto targetCovs = gpe::transform(target_array, [](const G& g){ return g.covariance; });

    const auto& responsibilities_1 = gradient_cache_data.responsibilities_1; // N_TARGET x N_FITTING
    const auto& responsibilities_2 = gradient_cache_data.responsibilities_2; // N_TARGET x N_FITTING
    const auto& responsibilities_3 = gradient_cache_data.responsibilities_3; // N_TARGET x N_FITTING

    // temps
    const scalar_t abs_integral = gpe::Epsilon<scalar_t>::clip(gpe::reduce(target_array, scalar_t(0), [](scalar_t i, const G& g) { return i + gpe::abs(gpe::integrate(g)); }));
    const auto posDiffs = gpe::outer_product(targetPositions, fittingPositions, fun::minus<pos_t>);
    const auto posDiffsOuter = gpe::transform(posDiffs, [](const pos_t& p) { return glm::outerProduct(p, p); });
    const auto unweightedCovs = gpe::cwise_fun(posDiffsOuter, targetCovs, fun::plus<cov_t>);
    const auto weightedCovs = gpe::cwise_fun(responsibilities_3, unweightedCovs, fun::times<scalar_t, cov_t>);
    const auto fittingWeights = gpe::reduce_cols(responsibilities_2, scalar_t(0), fun::plus<scalar_t>);
    const auto normal_amplitudes = gpe::transform(fittingCovariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    const auto int1_final_fitting_weights = gpe::cwise_fun(fittingWeights, normal_amplitudes, fun::times<scalar_t>);
    const auto clippedFittingWeights = gpe::transform(fittingWeights, gpe::Epsilon<scalar_t>::clip);
    const auto target_gaussian_amplitudes = gpe::transform(targetCovs, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    const auto pure_target_weights = gpe::cwise_fun(targetWeights, target_gaussian_amplitudes, fun::divided_AbyB<scalar_t>);


    // todo: make grad variable for every forward variable
    // make a small function to add Grad class onto forward variable grad (probably inside that struct, so we have gpe::grad::cwise_fun().addTo(grad1, grad2)
    // walk back to front and don't forget anything.

    // grad variables
    scalar_t grad_abs_integral = 0;
    std::decay_t<decltype (targetWeights)>              grad_targetWeights              {};
    std::decay_t<decltype (targetPositions)>            grad_targetPositions            {};
    std::decay_t<decltype (targetCovs)>                 grad_targetCovs                 {};

    std::decay_t<decltype (int1_final_fitting_weights)> grad_int1_final_fitting_weights {};
    std::decay_t<decltype (normal_amplitudes)>          grad_normal_amplitudes          {};
    std::decay_t<decltype (weightedCovs)>               grad_weightedCovs               {};
    std::decay_t<decltype (unweightedCovs)>             grad_unweightedCovs             {};
    std::decay_t<decltype (responsibilities_1)>         grad_responsibilities_1         {};
    std::decay_t<decltype (responsibilities_2)>         grad_responsibilities_2         {};
    std::decay_t<decltype (responsibilities_3)>         grad_responsibilities_3         {};
    std::decay_t<decltype (clippedFittingWeights)>      grad_clippedFittingWeights      {};
    std::decay_t<decltype (fittingWeights)>             grad_fittingWeights             {};
    std::decay_t<decltype (pure_target_weights)>        grad_pure_target_weights        {};
    std::decay_t<decltype (posDiffsOuter)>              grad_posDiffsOuter              {};
    std::decay_t<decltype (posDiffs)>                   grad_posDiffs                   {};
    std::decay_t<decltype (target_gaussian_amplitudes)> grad_target_gaussian_amplitudes {};


    // walk gradient back

    // const auto finalFittingWeights = gpe::transform(int1_final_fitting_weights, [abs_integral](scalar_t v) { return v * abs_integral; });
    gpe::grad::transform(int1_final_fitting_weights, grad_finalFittingWeights, [abs_integral, &grad_abs_integral](scalar_t v, scalar_t g) {
            grad_abs_integral += v * g;
            return g * abs_integral;
    }).addTo(&grad_int1_final_fitting_weights);

    // const auto int1_final_fitting_weights = gpe::cwise_fun(fittingWeights, normal_amplitudes, fun::times<scalar_t>);
    gpe::grad::cwise_fun(fittingWeights, normal_amplitudes, grad_int1_final_fitting_weights, gradfun::times<scalar_t>).addTo(&grad_fittingWeights, &grad_normal_amplitudes);

    // const auto normal_amplitudes = gpe::transform(fittingCovariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    gpe::grad::transform(fittingCovariances, grad_normal_amplitudes, gpe::grad::gaussian_amplitude<scalar_t, N_DIMS>).addTo(&grad_fittingCovariances);

    //auto fittingCovariances = gpe::reduce_cols(weightedCovs, cov_t(0), fun::plus<cov_t>);
    gpe::grad::sum_cols(weightedCovs, grad_fittingCovariances).addTo(&grad_weightedCovs);

    // const auto weightedCovs = gpe::cwise_fun(responsibilities_3, gpe::removeGrad(unweightedCovs), fun::times<scalar_t, gradless_cov_t>);
    gpe::grad::cwise_fun(responsibilities_3, unweightedCovs, grad_weightedCovs, gradfun::times<scalar_t, N_DIMS>).addTo(&grad_responsibilities_3, &grad_unweightedCovs);

    // const auto unweightedCovs = gpe::cwise_fun(posDiffsOuter, targetCovs, fun::plus<cov_t>);
    gpe::grad::cwise_fun(posDiffsOuter, targetCovs, grad_unweightedCovs, gradfun::plus<cov_t>).addTo(&grad_posDiffsOuter, &grad_targetCovs);

    // const auto posDiffsOuter = gpe::transform(posDiffs, [](const pos_t& p) { return glm::outerProduct(p, p); });

    // const auto posDiffs = gpe::outer_product(targetPositions, fittingPositions, fun::minus<pos_t>);

    // const auto fittingPositions = gpe::reduce_cols(weightedPositions, pos_t(0), fun::plus<pos_t>);

    // const auto weightedPositions = gpe::cwise_fun(responsibilities_3, targetPositions, fun::times<scalar_t, pos_t>);

    // const auto responsibilities_3 = gpe::cwise_fun(clippedFittingWeights, gpe::removeGrad(responsibilities_2), fun::divided_BbyA<scalar_t>);
    gpe::grad::cwise_fun(clippedFittingWeights, responsibilities_2, grad_responsibilities_3, gradfun::divided_BbyA<scalar_t>).addTo(&grad_clippedFittingWeights, &grad_responsibilities_2);

    // const auto clippedFittingWeights = gpe::transform(fittingWeights, gpe::Epsilon<scalar_t>::clip);
    // todo: clip gradient
    gpe::cwise_ref_fun(&grad_clippedFittingWeights, &grad_fittingWeights, [](const auto& gcw, auto& gw) { gw += gcw; });

    // const auto fittingWeights = gpe::reduce_cols(responsibilities_2, scalar_t(0), fun::plus<scalar_t>);
    gpe::grad::sum_cols(responsibilities_2, grad_fittingWeights).addTo(&grad_responsibilities_2);

    // const auto responsibilities_2 = gpe::cwise_fun(responsibilities_1, pure_target_weights, fun::times<scalar_t>);
    gpe::grad::cwise_fun(responsibilities_1, pure_target_weights, grad_responsibilities_2, gradfun::times<scalar_t>).addTo(&grad_responsibilities_1, &grad_pure_target_weights);

    // const auto pure_target_weights = gpe::cwise_fun(targetWeights, target_gaussian_amplitudes, fun::divided_AbyB<scalar_t>);
    gpe::grad::cwise_fun(targetWeights, target_gaussian_amplitudes, grad_pure_target_weights, gradfun::divided_AbyB<scalar_t>).addTo(&grad_targetWeights, &grad_target_gaussian_amplitudes);

    // const auto target_gaussian_amplitudes = gpe::transform(targetCovs, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    gpe::grad::transform(targetCovs, grad_target_gaussian_amplitudes, gpe::grad::gaussian_amplitude<scalar_t, N_DIMS>).addTo(&grad_targetCovs);

    // const auto responsibilities_1 = gpe::cwise_fun(weighted_likelihood_clamped_matrix, weighted_likelihood_sum, fun::divided_AbyB<gradless_scalar_t>);
//    gpe::grad::cwise_fun(weighted_likelihood_clamped_matrix, weighted_likelihood_sum, grad_responsibilities_1, gradfun::divided_AbyB<gradless_scalar_t>)
//            .addTo(grad_weighted_likelihood_clamped_matrix, grad_weighted_likelihood_sum);

    gpe::Vector<G, N_TARGET> target_grad{};
    for (unsigned i = 0; i < N_TARGET; ++i) {
        target_grad.push_back(G{grad_targetWeights[i] / abs_integral,
                                pos_t(0),
                                cov_t(0)});
    }

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
    return target_grad;
}

template <typename scalar_t, int N_DIMS, int REDUCTION_N>
EXECUTION_DEVICES
void trickle_down_grad(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                       const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                       gpe::PackedTensorAccessor32<scalar_t, 3> target_grad,
                       gpe::PackedTensorAccessor32<gpe::Gaussian<N_DIMS, scalar_t>, 2> mixture,
                       const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                       const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                       gpe::PackedTensorAccessor32<int, 2> flags,
                       gpe::PackedTensorAccessor32<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes, 2> node_attributes,
                       const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes,
                       const BvhMhemFitConfig& config) {
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;

    assert(gpe_blockDim.y == 1);
    assert(gpe_blockDim.z == 1);
    const auto mixture_id = int(gpe_blockIdx.y);
    assert(mixture_id < n_mixtures);

    Bvh bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>(mixture_id, nodes, aabbs, mixture, node_attributes, n, n_internal_nodes, n_nodes);
    #ifndef __CUDA_ARCH__
    std::vector<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes> node_attributes_debug;
    std::copy(bvh.per_node_attributes, bvh.per_node_attributes + n_nodes, std::back_inserter(node_attributes_debug));
    #endif

    gpe::ParallelStack<node_index_t, 32 * 32, 0> stack;
    {
        gpe::Vector<node_index_t, 32> top_stack;
        top_stack.push_back(0);
        while (top_stack.size()) {
            auto node_id = top_stack.pop_back();
            assert(node_id < n_nodes);
            if (bvh.per_node_attributes[node_id].grad.size() == 0) {
                top_stack.push_back(bvh.nodes[node_id].left_idx);
                top_stack.push_back(bvh.nodes[node_id].right_idx);
            }
            else {
                stack.push(node_id, gpe_threadIdx.x == 0, gpe_threadIdx.x);
            }
        }
    }

//    #continue here:
//    #top down traversal using stack. stop at leaves, then write into target_grad. skip the first few nodes until we see a populated grad field. then backprop through em code.

    // go top down through all nodes with grad
    while(stack.contains_elements(gpe_threadIdx.x))
    {
        node_index_t current_index = node_index_t(-1);
        if (!stack.pop(&current_index, gpe_threadIdx.x))
            continue;

        const Node* node = &bvh.nodes[current_index];
        if (current_index >= n_internal_nodes) {
            // leaf node
            reinterpret_cast<G&>(target_grad[mixture_id][current_index - n_internal_nodes][0]) = bvh.per_node_attributes[current_index].grad[0];
            continue;
        }

        auto child_gaussians = bvh.collect_child_gaussians(node, gpe::Epsilon<scalar_t>::large);
        if (child_gaussians.size() > REDUCTION_N) {
            auto child_grads = grad_em<REDUCTION_N>(child_gaussians,
                                                    bvh.per_node_attributes[current_index].gaussians,
                                                    bvh.per_node_attributes[current_index].grad,
                                                    bvh.per_node_attributes[current_index].gradient_cache_data,
                                                    config);
            bvh.distribute_gradient_on_children(node, child_grads, gpe::Epsilon<scalar_t>::large);
        }
        else {
            bvh.distribute_gradient_on_children(node, bvh.per_node_attributes[current_index].grad, gpe::Epsilon<scalar_t>::large);
        }

        stack.push(bvh.nodes[current_index].left_idx, true, gpe_threadIdx.x);
        stack.push(bvh.nodes[current_index].right_idx, true, gpe_threadIdx.x);
    }
}

// todo: this one can be refactored out. almost the same functino is used in forward and backward pass.
// todo: test
template <typename scalar_t, int N_DIMS, int REDUCTION_N, int N_MAX_TARGET_COMPS = 1024>
EXECUTION_DEVICES void distribute_grad(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                                      const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                                      const gpe::PackedTensorAccessor32<gpe::Gaussian<N_DIMS, scalar_t>, 2> mixture,
                                      gpe::PackedTensorAccessor32<scalar_t, 3> grad_fitting,
                                      const gpe::PackedTensorAccessor32<node_index_torch_t, 3> nodes,
                                      const gpe::PackedTensorAccessor32<scalar_t, 3> aabbs,
                                      gpe::PackedTensorAccessor32<int, 2> flags,
                                      gpe::PackedTensorAccessor32<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes, 2> node_attributes,
                                      const gpe::MixtureNs n, const int n_mixtures, const unsigned n_internal_nodes, const unsigned n_nodes,
                                      const BvhMhemFitConfig& config)
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

    Bvh bvh = Bvh(mixture_id, nodes, aabbs, mixture, node_attributes, n, n_internal_nodes, n_nodes);

    gpe::Vector<scalar_t, N_MAX_TARGET_COMPS> selectedNodesRating;
    gpe::Vector<node_index_t, N_MAX_TARGET_COMPS> selectedNodes;

    unsigned n_selected_components = 0;
    auto compute_rating = [&](node_index_t node_id) {
        assert(node_id < n_nodes);
        // todo: will break with negative weights, should compute sum of abs integrals / seperately positive and negative integrals
        if (bvh.per_node_attributes[node_id].gaussians.size() < REDUCTION_N)
            return scalar_t(-2); // -2 so it's safely below -1 from cach_id_with_highest_rating
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

    // copy grad to their starting posiion in the tree.
    unsigned read_position = 0;
    for (unsigned i = 0; i < selectedNodes.size(); ++i) {
        auto node_id = selectedNodes[i];
        typename Bvh::NodeAttributes& destination_attribute = bvh.per_node_attributes[node_id];

        for (unsigned j = 0; j < destination_attribute.gaussians.size(); ++j) {
            assert(read_position < config.n_components_fitting);
            destination_attribute.grad.push_back(gpe::gaussian<N_DIMS>(grad_fitting[mixture_id][int(read_position++)]));
        }
    }
}


} // anonymous namespace


template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
at::Tensor backward_impl_t(at::Tensor grad, const ForwardOutput& forward_out, const BvhMhemFitConfig& config) {
    using namespace torch::indexing;
    using LBVH = lbvh::Bvh<N_DIMS, scalar_t>;

    // todo: flatten mixture for kernel, i.g. nbatch/nlayers/ncomponents/7 => nmixture/ncomponents/7

    auto n = gpe::get_ns(forward_out.target);
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.components < 65535, "number of components must be smaller than 65535 for morton code computation")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
    TORCH_CHECK(n.dims == 2, "atm only 2d gaussians (because of eigenvector decomposition)")
    TORCH_CHECK(grad.dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")

    const auto n_mixtures = n.batch * n.layers;
    const auto bvh = LBVH(gpe::mixture_with_inversed_covariances(forward_out.bvh_mixture).contiguous(), forward_out.bvh_nodes, forward_out.bvh_aabbs);
    const auto n_internal_nodes = bvh.m_n_internal_nodes;
    const auto n_nodes = bvh.m_n_nodes;
    const auto mixture_view = forward_out.target.view({n_mixtures, n.components, -1}).contiguous();
    const auto grad_view = grad.view({n_mixtures, config.n_components_fitting, -1}).contiguous();
    auto flat_bvh_nodes = bvh.m_nodes.view({n_mixtures, n_nodes, -1});
    auto flat_bvh_aabbs = bvh.m_aabbs.view({n_mixtures, n_nodes, -1});
    auto flag_container = torch::zeros({n_mixtures, n_internal_nodes}, torch::TensorOptions(mixture_view.device()).dtype(torch::ScalarType::Int));

    auto flags_a = gpe::accessor<int, 2>(flag_container);
    auto node_attributes = forward_out.bvh_attributes.view({n_mixtures, n_nodes, -1});

    auto mixture_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 2, scalar_t>(mixture_view);
    auto grad_a = gpe::accessor<scalar_t, 3>(grad_view);
    auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(flat_bvh_nodes);
    auto aabbs_a = gpe::accessor<scalar_t, 3>(flat_bvh_aabbs);
    auto node_attributes_a = gpe::struct_accessor<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes, 2, uint8_t>(node_attributes);

    {
        // distribute the fitting gradient using the same algorithm amoung the nodes.
        dim3 dimBlock = dim3(32, 1, 1);
        dim3 dimGrid = dim3((uint(n_mixtures) + dimBlock.x - 1) / dimBlock.x, 1, 1);

        auto fun = [mixture_a, grad_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
                __host__ __device__
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
            distribute_grad<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                          mixture_a, grad_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                          n, n_mixtures, n_internal_nodes, n_nodes,
                                                          config);
        };
        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture_view), dimGrid, dimBlock, fun);
    }

    auto target_gradient = torch::zeros_like(mixture_view);
    auto target_gradient_a = gpe::accessor<scalar_t, 3>(target_gradient);
    {
        dim3 dimBlock = dim3(32, 1, 1);
        dim3 dimGrid = dim3(uint(1),
                            (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
                            (uint(1) + dimBlock.z - 1) / dimBlock.z);

        auto fun = [target_gradient_a, mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config] __host__ __device__
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
            trickle_down_grad<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                             target_gradient_a,
                                                             mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                             n, n_mixtures, n_internal_nodes, n_nodes,
                                                             config);
        };
        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture_view), dimGrid, dimBlock, fun);
    }



    return target_gradient.view_as(forward_out.target);
}

} // namespace bvh_mhem_fit

