#include "bvh_mhem_fit_alpha/implementation.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "bvh_mhem_fit_alpha/implementation_common.h"
#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
#include "hacked_accessor.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "parallel_start.h"
#include "util/algorithms.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/grad/algorithms.h"
#include "util/grad/glm.h"
#include "util/grad/gaussian.h"
#include "util/grad/mixture.h"
#include "util/mixture.h"


// todo:
// - in collect_result, run a new fitting with the most important node to fill up the remaining gaussian slots

namespace bvh_mhem_fit_alpha {

namespace  {

// todo: refactor:
// only one fit_em function (not fit_em + grad_em). parametrised via template, computes grad only if the template param says so. use the bvh node as function parameter => does it's own collection of child gaussians?
template <unsigned N_FITTING, typename scalar_t, int N_DIMS, unsigned N_TARGET, typename size_type>
EXECUTION_DEVICES
gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_TARGET> grad_em(const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_TARGET>& target,
                                                               const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_FITTING, size_type>& fitting,
                                                               const gpe::Vector<gpe::Gaussian<N_DIMS, scalar_t>, N_FITTING, size_type>& fitting_grad,
                                                               const GradientCacheData<scalar_t, N_FITTING, N_FITTING * 2>& gradient_cache_data,
                                                               const Config& config) {
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using pos_t = typename G::pos_t;
    using cov_t = typename G::cov_t;

    namespace fun = gpe::functors;
    namespace gradfun = gpe::grad::functors;

    auto has_nan = [](const auto& vec) {
        return gpe::reduce(vec, false, [](bool o, auto v) { return o || gpe::isnan(v); });
    };

    // input and result
    auto target_mixture = gpe::to_array(target, G{0, pos_t(0), cov_t(1)});
    auto fitting_array = gpe::to_array(fitting, G{0, pos_t(0), cov_t(1)});
    auto fitting_grad_array = gpe::to_array(fitting_grad, G{0, pos_t(0), cov_t(0)});

    const auto grad_fitting_weights = gpe::transform(fitting_grad_array, [](const G& g){ return g.weight; });
    auto grad_fittingPositions = gpe::transform(fitting_grad_array, [](const G& g){ return g.position; });
    auto grad_fittingCovariances = gpe::transform(fitting_grad_array, [](const G& g){ return g.covariance; });

    // forward cached
    const auto fitting_weights = gpe::transform(fitting_array, [](const G& g){ return g.weight; });
    const auto fittingPositions = gpe::transform(fitting_array, [](const G& g){ return g.position; });
    const auto fittingCovariances = gpe::transform(fitting_array, [](const G& g){ return g.covariance; });

    const auto target_weights = gpe::transform(target_mixture, [](const G& g){ return g.weight; });
    const auto target_positions = gpe::transform(target_mixture, [](const G& g){ return g.position; });
    const auto target_covariances = gpe::transform(target_mixture, [](const G& g){ return g.covariance; });

    const auto& initial_indices = gradient_cache_data.initial_indices;
    const auto& responsibilities_2 = gradient_cache_data.responsibilities_2; // N_TARGET x N_FITTING
    const auto& responsibilities_3 = gradient_cache_data.responsibilities_3; // N_TARGET x N_FITTING

    // forward not cached
    const auto target_component_integrals = gpe::transform(target_mixture, gpe::integrate<scalar_t, N_DIMS>);
    const scalar_t target_integral = gpe::reduce(target_component_integrals, scalar_t(0), fun::plus<scalar_t>);
    const scalar_t target_clipped_integral = gpe::Epsilon<scalar_t>::clip(target_integral);
    const auto target_int1_weights = gpe::cwise_fun(target_weights, target_clipped_integral, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    const auto target_int1_mixture = gpe::pack_mixture(target_int1_weights, target_positions, target_covariances);

    const auto initial_mixture = gpe::select(target_mixture, initial_indices);
    const auto initial_weights = gpe::select(target_weights, initial_indices);
    const auto initial_positions = gpe::select(target_positions, initial_indices);
    const auto initial_covariances = gpe::select(target_covariances, initial_indices);
    const auto initial_component_integrals = gpe::transform(initial_mixture, gpe::integrate<scalar_t, N_DIMS>);
    const auto initial_integral = gpe::reduce(initial_component_integrals, scalar_t(0), fun::plus<scalar_t>);
    const auto initial_clipped_integral = gpe::Epsilon<scalar_t>::clip(initial_integral);
    const auto initial_int1_weights = gpe::cwise_fun(initial_weights, initial_clipped_integral, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    const auto initial_int1_mixture = gpe::pack_mixture(initial_int1_weights, initial_positions, initial_covariances);
    const auto initial_gaussian_amplitudes = gpe::transform(initial_covariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    const auto initial_pure_weights = gpe::cwise_fun(initial_int1_weights, initial_gaussian_amplitudes, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);

    const auto likelihood_matrix = gpe::outer_product(target_int1_mixture, initial_int1_mixture, gpe::likelihood<scalar_t, N_DIMS>);
    const auto kldiv_sign_matrix = gpe::outer_product(target_int1_mixture, initial_int1_mixture, [](auto target, auto fitting) {
        return (gpe::sign(fitting.weight) == gpe::sign(target.weight)) ? gpe::kl_divergence<scalar_t, N_DIMS>(target, fitting) : scalar_t(0);
    });

    const auto kl_div_threshold = config.em_kl_div_threshold;
    auto clamp_matrix = gpe::transform(kldiv_sign_matrix, [kl_div_threshold](scalar_t v) { return v < kl_div_threshold ? scalar_t(1) : scalar_t(0); });
    for (unsigned target_id = 0; target_id < clamp_matrix.size(); ++target_id) {
        auto& row = kldiv_sign_matrix[target_id];
        unsigned best_fitting_id = unsigned(-1);
        auto smallest_value = std::numeric_limits<scalar_t>::infinity();
        for (unsigned fitting_id = 0; fitting_id < row.size(); ++fitting_id) {
            if (row[fitting_id] < smallest_value) {
                smallest_value = row[fitting_id];
                best_fitting_id = fitting_id;
            }
        }
        assert(best_fitting_id < N_FITTING);
        clamp_matrix[target_id][best_fitting_id] = scalar_t(1);  // no change if largest value was > kl_div_threshold.
    }

    const auto weighted_likelihood_matrix = gpe::cwise_fun(initial_pure_weights, likelihood_matrix, fun::times<scalar_t, scalar_t, scalar_t>);
    const auto weighted_likelihood_matrix_clipped = gpe::transform(weighted_likelihood_matrix, gpe::Epsilon<scalar_t>::clip);
    const auto weighted_likelihood_clamped_matrix = gpe::cwise_fun(weighted_likelihood_matrix_clipped, clamp_matrix, fun::times<scalar_t, scalar_t, scalar_t>);
    const auto weighted_likelihood_sum = gpe::reduce_rows(weighted_likelihood_clamped_matrix, scalar_t(0), fun::plus<scalar_t, scalar_t, scalar_t>);
    const auto responsibilities_1 = gpe::cwise_fun(weighted_likelihood_clamped_matrix, weighted_likelihood_sum, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);

    const auto posDiffs = gpe::outer_product(target_positions, fittingPositions, fun::minus<pos_t>);
    const auto posDiffsOuter = gpe::transform(posDiffs, [](const pos_t& p) { return glm::outerProduct(p, p); });
    const auto unweightedCovs = gpe::cwise_fun(posDiffsOuter, target_covariances, fun::plus<cov_t>);
    const auto weightedCovs = gpe::cwise_fun(responsibilities_3, unweightedCovs, fun::times<scalar_t, cov_t>);
    const auto fitting_pure_weights = gpe::reduce_cols(responsibilities_2, scalar_t(0), fun::plus<scalar_t>);
    const auto fitting_normal_amplitudes = gpe::transform(fittingCovariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    const auto fitting_int1_weights = gpe::cwise_fun(fitting_pure_weights, fitting_normal_amplitudes, fun::times<scalar_t>);
    const auto clippedFittingWeights = gpe::transform(fitting_pure_weights, gpe::Epsilon<scalar_t>::clip);
    const auto target_gaussian_amplitudes = gpe::transform(target_covariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);

    const auto pure_target_weights = gpe::cwise_fun(target_int1_weights, target_gaussian_amplitudes, fun::divided_AbyB<scalar_t>);
    const auto weightedPositions = gpe::cwise_fun(responsibilities_3, target_positions, fun::times<scalar_t, pos_t>);


    // todo: make grad variable for every forward variable
    // make a small function to add Grad class onto forward variable grad (probably inside that struct, so we have gpe::grad::cwise_fun().addTo(grad1, grad2)
    // walk back to front and don't forget anything.

    // grad variables
    scalar_t grad_target_clipped_integral = 0;
    std::decay_t<decltype (target_weights                           )> grad_target_weights                          {};
    std::decay_t<decltype (target_positions                         )> grad_target_positions                        {};
    std::decay_t<decltype (target_covariances                       )> grad_target_covariances                      {};
    std::decay_t<decltype (target_integral                          )> grad_target_integral                         {};
    std::decay_t<decltype (target_mixture                           )> grad_target_mixture                          {};
    std::decay_t<decltype (target_int1_mixture                      )> grad_target_int1_mixture                     {};
    std::decay_t<decltype (target_component_integrals               )> grad_target_component_integrals              {};

    std::decay_t<decltype (initial_mixture                          )> grad_initial_mixture                         {};
    std::decay_t<decltype (initial_weights                          )> grad_initial_weights                         {};
    std::decay_t<decltype (initial_positions                        )> grad_initial_positions                       {};
    std::decay_t<decltype (initial_covariances                      )> grad_initial_covariances                     {};
    std::decay_t<decltype (initial_component_integrals              )> grad_initial_component_integrals             {};
    std::decay_t<decltype (initial_integral                         )> grad_initial_integral                        {};
    std::decay_t<decltype (initial_clipped_integral                 )> grad_initial_clipped_integral                {};
    std::decay_t<decltype (initial_int1_weights                     )> grad_initial_int1_weights                    {};
    std::decay_t<decltype (initial_int1_mixture                     )> grad_initial_int1_mixture                    {};
    std::decay_t<decltype (initial_gaussian_amplitudes              )> grad_initial_gaussian_amplitudes             {};
    std::decay_t<decltype (initial_pure_weights                     )> grad_initial_pure_weights                    {};

    std::decay_t<decltype (fitting_int1_weights                     )> grad_fitting_int1_weights                    {};
    std::decay_t<decltype (fitting_normal_amplitudes                )> grad_fitting_normal_amplitudes               {};
    std::decay_t<decltype (weightedCovs                             )> grad_weightedCovs                            {};
    std::decay_t<decltype (unweightedCovs                           )> grad_unweightedCovs                          {};
    std::decay_t<decltype (responsibilities_1                       )> grad_responsibilities_1                      {};
    std::decay_t<decltype (responsibilities_2                       )> grad_responsibilities_2                      {};
    std::decay_t<decltype (responsibilities_3                       )> grad_responsibilities_3                      {};
    std::decay_t<decltype (clippedFittingWeights                    )> grad_clippedFittingWeights                   {};
    std::decay_t<decltype (fitting_pure_weights                     )> grad_fitting_pure_weights                    {};
    std::decay_t<decltype (target_int1_weights                      )> grad_target_int1_weights                     {};
    std::decay_t<decltype (pure_target_weights                      )> grad_pure_target_weights                     {};
    std::decay_t<decltype (posDiffsOuter                            )> grad_posDiffsOuter                           {};
    std::decay_t<decltype (posDiffs                                 )> grad_posDiffs                                {};
    std::decay_t<decltype (target_gaussian_amplitudes               )> grad_target_gaussian_amplitudes              {};
    std::decay_t<decltype (weightedPositions                        )> grad_weightedPositions                       {};

    std::decay_t<decltype (weighted_likelihood_clamped_matrix       )> grad_weighted_likelihood_clamped_matrix      {};
    std::decay_t<decltype (weighted_likelihood_sum                  )> grad_weighted_likelihood_sum                 {};
    std::decay_t<decltype (weighted_likelihood_matrix               )> grad_weighted_likelihood_matrix              {};
    std::decay_t<decltype (weighted_likelihood_matrix_clipped       )> grad_weighted_likelihood_matrix_clipped      {};

    std::decay_t<decltype (likelihood_matrix                        )> grad_likelihood_matrix                       {};


    // walk gradient back

    // const auto fitting_weights = gpe::cwise_fun(fitting_int1_weights, target_clipped_integral, fun::times<scalar_t, scalar_t, scalar_t>);
    gpe::grad::cwise_fun(fitting_int1_weights, target_clipped_integral, grad_fitting_weights, gradfun::times<scalar_t>).addTo(&grad_fitting_int1_weights, &grad_target_clipped_integral);

    // const auto fitting_int1_weights = gpe::cwise_fun(fitting_pure_weights, fitting_normal_amplitudes, fun::times<scalar_t>);
    gpe::grad::cwise_fun(fitting_pure_weights, fitting_normal_amplitudes, grad_fitting_int1_weights, gradfun::times<scalar_t>).addTo(&grad_fitting_pure_weights, &grad_fitting_normal_amplitudes);

    // const auto fitting_normal_amplitudes = gpe::transform(fittingCovariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    gpe::grad::transform(fittingCovariances, grad_fitting_normal_amplitudes, gpe::grad::gaussian_amplitude<scalar_t, N_DIMS>).addTo(&grad_fittingCovariances);
    assert(!has_nan(grad_fittingCovariances));

    // auto fittingCovariances = gpe::reduce_cols(weightedCovs, cov_t(0), fun::plus<cov_t>);
    gpe::grad::sum_cols(weightedCovs, grad_fittingCovariances).addTo(&grad_weightedCovs);

    // const auto weightedCovs = gpe::cwise_fun(responsibilities_3, unweightedCovs, fun::times<scalar_t, cov_t>);
    gpe::grad::cwise_fun(responsibilities_3, unweightedCovs, grad_weightedCovs, gradfun::times<scalar_t, N_DIMS, N_DIMS>).addTo(&grad_responsibilities_3, &grad_unweightedCovs);

    // const auto unweightedCovs = gpe::cwise_fun(posDiffsOuter, target_covariances, fun::plus<cov_t>);
    gpe::grad::cwise_fun(posDiffsOuter, target_covariances, grad_unweightedCovs, gradfun::plus<cov_t>).addTo(&grad_posDiffsOuter, &grad_target_covariances);

    // const auto posDiffsOuter = gpe::cwise_fun(posDiffs, posDiffs, gpe::outerProduct<scalar_t, N_DIMS>);
    gpe::grad::cwise_fun(posDiffs, posDiffs, grad_posDiffsOuter, gpe::grad::outerProduct<N_DIMS, scalar_t>).addTo(&grad_posDiffs, &grad_posDiffs);
    assert(!has_nan(grad_posDiffs));

    // const auto posDiffs = gpe::outer_product(target_positions, fittingPositions, fun::minus<pos_t>);
    gpe::grad::outer_product(target_positions, fittingPositions, grad_posDiffs, gradfun::minus<pos_t>).addTo(&grad_target_positions, &grad_fittingPositions);

    // const auto fittingPositions = gpe::reduce_cols(weightedPositions, pos_t(0), fun::plus<pos_t>);
    gpe::grad::sum_cols(weightedPositions, grad_fittingPositions).addTo(&grad_weightedPositions);

    // const auto weightedPositions = gpe::cwise_fun(responsibilities_3, target_positions, fun::times<scalar_t, pos_t>);
    gpe::grad::cwise_fun(responsibilities_3, target_positions, grad_weightedPositions, gradfun::times<scalar_t, N_DIMS>).addTo(&grad_responsibilities_3, &grad_target_positions);

    // const auto responsibilities_3 = gpe::cwise_fun(clippedFittingWeights, responsibilities_2, fun::divided_BbyA<scalar_t>);
    gpe::grad::cwise_fun(clippedFittingWeights, responsibilities_2, grad_responsibilities_3, gradfun::divided_BbyA<scalar_t>).addTo(&grad_clippedFittingWeights, &grad_responsibilities_2);
    assert(!has_nan(grad_clippedFittingWeights));
    assert(!has_nan(grad_responsibilities_2));

    // const auto clippedFittingWeights = gpe::transform(fitting_pure_weights, gpe::Epsilon<scalar_t>::clip);
    gpe::grad::transform(fitting_pure_weights, grad_clippedFittingWeights, gpe::Epsilon<scalar_t>::grad_clip).addTo(&grad_fitting_pure_weights);

    // const auto fitting_pure_weights = gpe::reduce_cols(responsibilities_2, scalar_t(0), fun::plus<scalar_t>);
    gpe::grad::sum_cols(responsibilities_2, grad_fitting_pure_weights).addTo(&grad_responsibilities_2);

    // const auto responsibilities_2 = gpe::cwise_fun(responsibilities_1, pure_target_weights, fun::times<scalar_t>);
    gpe::grad::cwise_fun(responsibilities_1, pure_target_weights, grad_responsibilities_2, gradfun::times<scalar_t>).addTo(&grad_responsibilities_1, &grad_pure_target_weights);

    // const auto pure_target_weights = gpe::cwise_fun(target_int1_weights, target_gaussian_amplitudes, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    gpe::grad::cwise_fun(target_int1_weights, target_gaussian_amplitudes, grad_pure_target_weights, gradfun::divided_AbyB<scalar_t>).addTo(&grad_target_int1_weights, &grad_target_gaussian_amplitudes);
    assert(!has_nan(grad_target_int1_weights));
    assert(!has_nan(grad_target_gaussian_amplitudes));

    // const auto target_gaussian_amplitudes = gpe::transform(target_covariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    gpe::grad::transform(target_covariances, grad_target_gaussian_amplitudes, gpe::grad::gaussian_amplitude<scalar_t, N_DIMS>).addTo(&grad_target_covariances);
    assert(!has_nan(grad_target_covariances));

    // information: this line potentially causes numerical problems: weighted_likelihood_sum can be very small (epsilon from the clamp), and grad_weighted_likelihood_clamped_matrix += grad_responsibilities_1 / eps
    // const auto responsibilities_1 = gpe::cwise_fun(weighted_likelihood_clamped_matrix, weighted_likelihood_sum, fun::divided_AbyB<scalar_t>);
    gpe::grad::cwise_fun(weighted_likelihood_clamped_matrix, weighted_likelihood_sum, grad_responsibilities_1, gradfun::divided_AbyB<scalar_t>)
                        .addTo(&grad_weighted_likelihood_clamped_matrix, &grad_weighted_likelihood_sum);
    assert(!has_nan(grad_weighted_likelihood_clamped_matrix));
    assert(!has_nan(grad_weighted_likelihood_sum));

    // const auto weighted_likelihood_sum = gpe::reduce_rows(weighted_likelihood_clamped_matrix, scalar_t(0), fun::plus<scalar_t>);
    gpe::grad::sum_rows(weighted_likelihood_clamped_matrix, grad_weighted_likelihood_sum).addTo(&grad_weighted_likelihood_clamped_matrix);

    // const auto weighted_likelihood_clamped_matrix = gpe::cwise_fun(weighted_likelihood_matrix_clipped, clamp_matrix, fun::times<scalar_t, scalar_t, scalar_t>);
    gpe::grad::cwise_fun(weighted_likelihood_matrix_clipped, clamp_matrix, grad_weighted_likelihood_clamped_matrix, gradfun::times<scalar_t>).addTo(&grad_weighted_likelihood_matrix_clipped, false);

    // const auto weighted_likelihood_matrix_clipped = gpe::transform(weighted_likelihood_matrix, gpe::Epsilon<scalar_t>::clip);
    gpe::grad::transform(weighted_likelihood_matrix, grad_weighted_likelihood_matrix_clipped, gpe::Epsilon<scalar_t>::grad_clip).addTo(&grad_weighted_likelihood_matrix);

    // const auto weighted_likelihood_matrix = gpe::cwise_fun(initial_pure_weights, likelihood_matrix, fun::times<scalar_t, scalar_t, scalar_t>);
    gpe::grad::cwise_fun(initial_pure_weights, likelihood_matrix, grad_weighted_likelihood_matrix, gradfun::times<scalar_t>).addTo(&grad_initial_pure_weights, &grad_likelihood_matrix);

    // const auto likelihood_matrix = gpe::outer_product(target_int1_mixture, initial_int1_mixture, gpe::likelihood<scalar_t, N_DIMS>);
    gpe::grad::outer_product(target_int1_mixture, initial_int1_mixture, grad_likelihood_matrix, gpe::grad::likelihood<scalar_t, N_DIMS>).addTo(&grad_target_int1_mixture, &grad_initial_int1_mixture);
    assert(!has_nan(grad_target_int1_mixture));
    assert(!has_nan(grad_initial_int1_mixture));

    // const auto target_int1_mixture = gpe::pack_mixture(target_int1_weights, target_positions, target_covariances);
    gpe::grad::unpackAndAdd(grad_target_int1_mixture, &grad_target_int1_weights, &grad_target_positions, &grad_target_covariances);

    // const auto initial_pure_weights = gpe::cwise_fun(initial_int1_weights, initial_gaussian_amplitudes, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    gpe::grad::cwise_fun(initial_int1_weights, initial_gaussian_amplitudes, grad_initial_pure_weights, gradfun::divided_AbyB<scalar_t>).addTo(&grad_initial_int1_weights, &grad_initial_gaussian_amplitudes);
    assert(!has_nan(grad_initial_int1_weights));
    assert(!has_nan(grad_initial_gaussian_amplitudes));

    // const auto initial_gaussian_amplitudes = gpe::transform(initial_covariances, gpe::gaussian_amplitude<scalar_t, N_DIMS>);
    gpe::grad::transform(initial_covariances, grad_initial_gaussian_amplitudes, gpe::grad::gaussian_amplitude<scalar_t, N_DIMS>).addTo(&grad_initial_covariances);
    assert(!has_nan(grad_initial_covariances));

    // const auto initial_int1_mixture = gpe::pack_mixture(initial_int1_weights, initial_positions, initial_covariances);
    gpe::grad::unpackAndAdd(grad_initial_int1_mixture, &grad_initial_int1_weights, &grad_initial_positions, &grad_initial_covariances);

    // const auto initial_int1_weights = gpe::cwise_fun(initial_weights, initial_clipped_integral, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    gpe::grad::cwise_fun(initial_weights, initial_clipped_integral, grad_initial_int1_weights, gradfun::divided_AbyB<scalar_t>).addTo(&grad_initial_weights, &grad_initial_clipped_integral);
    assert(!has_nan(grad_initial_weights));
    assert(!gpe::isnan(grad_initial_clipped_integral));

    // const auto initial_clipped_integral = gpe::Epsilon<scalar_t>::clip(initial_integral);
    grad_initial_integral = gpe::Epsilon<scalar_t>::grad_clip(initial_integral, grad_initial_clipped_integral);

    // const auto initial_integral = gpe::reduce(initial_component_integrals, scalar_t(0), fun::plus<scalar_t>);
    gpe::grad::sum(initial_component_integrals, grad_initial_integral).addTo(&grad_initial_component_integrals);

    // const auto initial_component_integrals = gpe::transform(initial_mixture, gpe::integrate<scalar_t, N_DIMS>);
    gpe::grad::transform(initial_mixture, grad_initial_component_integrals, gpe::grad::integrate<scalar_t, N_DIMS>).addTo(&grad_initial_mixture);
    assert(!has_nan(grad_initial_mixture));

    // const auto initial_covariances = gpe::select(target_covariances, initial_indices);
    gpe::grad::select(target_covariances, initial_indices, grad_initial_covariances).addTo(&grad_target_covariances);

    // const auto initial_positions = gpe::select(target_positions, initial_indices);
    gpe::grad::select(target_positions, initial_indices, grad_initial_positions).addTo(&grad_target_positions);

    // const auto initial_weights = gpe::select(target_weights, initial_indices);
    gpe::grad::select(target_weights, initial_indices, grad_initial_weights).addTo(&grad_target_weights);

    // const auto initial_mixture = gpe::select(target_mixture, initial_indices);
    gpe::grad::select(target_mixture, initial_indices, grad_initial_mixture).addTo(&grad_target_mixture);

    // const auto target_int1_weights = gpe::cwise_fun(target_weights, target_clipped_integral, fun::divided_AbyB<scalar_t, scalar_t, scalar_t>);
    gpe::grad::cwise_fun(target_weights, target_clipped_integral, grad_target_int1_weights, gradfun::divided_AbyB<scalar_t>).addTo(&grad_target_weights, &grad_target_clipped_integral);
    assert(!has_nan(grad_target_weights));
    assert(!gpe::isnan(grad_target_clipped_integral));

    // const scalar_t target_clipped_integral = gpe::Epsilon<scalar_t>::clip(target_integral);
    grad_target_integral = gpe::Epsilon<scalar_t>::grad_clip(target_integral, grad_target_clipped_integral);

    // const scalar_t target_integral = gpe::reduce(target_component_integrals, scalar_t(0), fun::plus<scalar_t, scalar_t, scalar_t>);
    gpe::grad::sum(target_component_integrals, grad_target_integral).addTo(&grad_target_component_integrals);

    // const auto target_component_integrals = gpe::transform(target_mixture, gpe::integrate<scalar_t, N_DIMS>);
    gpe::grad::transform(target_mixture, grad_target_component_integrals, gpe::grad::integrate<scalar_t, N_DIMS>).addTo(&grad_target_mixture);
    assert(!has_nan(grad_target_mixture));

    // const auto target_covariances = gpe::transform(target_mixture, [](const G& g){ return g.covariance; });
    // const auto target_positions = gpe::transform(target_mixture, [](const G& g){ return g.position; });
    // const auto target_weights = gpe::transform(target_mixture, [](const G& g){ return g.weight; });
    gpe::Vector<G, N_TARGET> target_grad{};
    for (unsigned i = 0; i < target.size(); ++i) {
        target_grad.push_back(G{grad_target_weights[i] + grad_target_mixture[i].weight,
                                grad_target_positions[i] + grad_target_mixture[i].position,
                                grad_target_covariances[i] + grad_target_mixture[i].covariance});
    }
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
                       const Config& config) {
    GPE_UNUSED(gpe_gridDim)
    using G = gpe::Gaussian<N_DIMS, scalar_t>;
    using Bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>;

    assert(gpe_blockDim.y == 1);
    assert(gpe_blockDim.z == 1);
    const auto mixture_id = int(gpe_blockIdx.y);
    assert(mixture_id < n_mixtures);

    Bvh bvh = AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>(mixture_id, nodes, aabbs, mixture, node_attributes, n, n_internal_nodes, n_nodes);
//    #ifndef __CUDA_ARCH__
//        std::vector<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes> node_attributes_debug;
//        std::vector<Node> nodes_debug;
//        std::vector<G> mixture_debug;

//        auto updateDebug = [&]() {
//            node_attributes_debug.clear();
//            nodes_debug.clear();
//            mixture_debug.clear();
//            std::copy(bvh.per_node_attributes, bvh.per_node_attributes + n_nodes, std::back_inserter(node_attributes_debug));
//            std::copy(bvh.nodes, bvh.nodes + n_nodes, std::back_inserter(nodes_debug));
//            std::copy(bvh.gaussians, bvh.gaussians + n.components, std::back_inserter(mixture_debug));
//        };
//        updateDebug();
//    #endif

    gpe::Vector<node_index_t, 32 * 32> stack;
    {
        gpe::Vector<node_index_t, 32> top_stack;
        top_stack.push_back(0);
        while (top_stack.size()) {
            auto node_id = top_stack.pop_back();
            if(node_id >= n_nodes)
                continue;   // ran out of nodes, this is a border case happening when the mixtures contain only zero gaussians.
            if (bvh.per_node_attributes[node_id].grad.size() == 0) {
                top_stack.push_back(bvh.nodes[node_id].left_idx);
                top_stack.push_back(bvh.nodes[node_id].right_idx);
            }
            else {
                stack.push_back(node_id);
            }
        }
    }

    // go top down through all nodes with grad
    while(stack.size())
    {
        node_index_t current_index = stack.pop_back();

        const Node* node = &bvh.nodes[current_index];
        if (current_index >= n_internal_nodes) {
            // leaf node
            if (bvh.per_node_attributes[current_index].grad.size() == 1) {  // grad is empty, if the original gaussian was zero. this check can be removed if the node attributes are initialised with zeroes.
                reinterpret_cast<G&>(target_grad[mixture_id][node->object_idx][0]) = bvh.per_node_attributes[current_index].grad[0];
            }
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
//            updateDebug();
        }
        else {
            bvh.distribute_gradient_on_children(node, bvh.per_node_attributes[current_index].grad, gpe::Epsilon<scalar_t>::large);
//            updateDebug();
        }

        stack.push_back(bvh.nodes[current_index].left_idx);
        stack.push_back(bvh.nodes[current_index].right_idx);
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
at::Tensor backward_impl_t(at::Tensor grad, const ForwardOutput& forward_out, const Config& config) {
    using namespace torch::indexing;
    using LBVH = lbvh::Bvh<N_DIMS, scalar_t>;

    // todo: flatten mixture for kernel, i.g. nbatch/nlayers/ncomponents/7 => nmixture/ncomponents/7

    auto n = gpe::get_ns(forward_out.target);
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.components < 65535, "number of components must be smaller than 65535 for morton code computation")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
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
        dim3 dimBlock = dim3(1, 1, 1);
        dim3 dimGrid = dim3((uint(n_mixtures) + dimBlock.x - 1) / dimBlock.x, 1, 1);

        auto fun = [mixture_a, grad_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
            distribute_grad<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                          mixture_a, grad_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                          n, n_mixtures, n_internal_nodes, n_nodes,
                                                          config);
        };
        gpe::start_serial(gpe::device(mixture_view), dimGrid, dimBlock, fun);
    }

    auto target_gradient = torch::zeros_like(mixture_view);
    auto target_gradient_a = gpe::accessor<scalar_t, 3>(target_gradient);
    {
        dim3 dimBlock = dim3(1, 1, 1);
        dim3 dimGrid = dim3(uint(1),
                            (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
                            (uint(1) + dimBlock.z - 1) / dimBlock.z);

        auto fun = [target_gradient_a, mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
            trickle_down_grad<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                             target_gradient_a,
                                                             mixture_a, nodes_a, aabbs_a, flags_a, node_attributes_a,
                                                             n, n_mixtures, n_internal_nodes, n_nodes,
                                                             config);
        };
        gpe::start_serial(gpe::device(mixture_view), dimGrid, dimBlock, fun);
    }



    return target_gradient.view_as(forward_out.target);
}

} // namespace bvh_mhem_fit

