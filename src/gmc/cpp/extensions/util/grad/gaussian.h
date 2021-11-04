#ifndef GPE_UTIL_GRAD_MATRIX_H
#define GPE_UTIL_GRAD_MATRIX_H

#include <gcem.hpp>

#include "util/cuda.h"
#include "util/gaussian.h"
#include "util/glm.h"
#include "util/grad/algorithms.h"
#include "util/grad/glm.h"
#include "util/grad/scalar.h"
#include "util/scalar.h"

namespace gpe {
namespace grad {



template <typename scalar_t, int DIMS>
EXECUTION_DEVICES void evaluate_inversed(const Gaussian<DIMS, scalar_t>& gaussian, const glm::vec<DIMS, scalar_t>& evalpos,
                                Gaussian<DIMS, scalar_t>* grad_gaussian, glm::vec<DIMS, scalar_t>* grad_evalpos,
                                scalar_t incoming_grad) {
    using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;
#ifdef _MSC_VER
    gradless_scalar_t factor = gcem::pow(2 * glm::pi<gradless_scalar_t>(), -gradless_scalar_t(DIMS) / gradless_scalar_t(2.));
#else
    constexpr gradless_scalar_t factor = gcem::pow(2 * glm::pi<gradless_scalar_t>(), -gradless_scalar_t(DIMS) / gradless_scalar_t(2.));
#endif

    using Vec = glm::vec<DIMS, scalar_t>;
    const auto t = evalpos - gaussian.position;
    const auto covInv_times_t = (gaussian.covariance * t);
    const auto v = scalar_t(-0.5) * glm::dot(t, covInv_times_t);
    const auto det = glm::determinant(gaussian.covariance);
    const auto root = gpe::sqrt(det);
    const auto norm = root * factor;
    const auto exp = gpe::exp(v);
    // return gaussian.weight * norm * exp;


    grad_gaussian->weight = norm * exp * incoming_grad;
    const scalar_t grad_norm = gaussian.weight * exp * incoming_grad;
    const scalar_t grad_exp = gaussian.weight * norm * incoming_grad;

    const scalar_t grad_root = grad_norm * factor;
    const scalar_t grad_det = grad_root / (2 * root);

    const scalar_t grad_v = gpe::grad::exp(v, grad_exp);

//    const auto v = scalar_t(-0.5) * glm::dot(t, covInv_times_t);
    Vec grad_t;
    Vec grad_covInv_times_t;
    gpe::grad::dot(t, covInv_times_t, &grad_t, &grad_covInv_times_t, grad_v * scalar_t(-0.5));

    // according to https://www.youtube.com/watch?v=R_m4kanPy6Q.
//    const auto covInv_times_t = (covInv * t);
    auto grad_covInv = gpe::outerProduct(grad_covInv_times_t, t);
    // ignoring the transpose due to symmetry
    grad_t += gaussian.covariance * grad_covInv_times_t;

    grad_gaussian->covariance = grad_covInv + gpe::grad::determinant(gaussian.covariance, grad_det);

//    const auto t = evalpos - gaussian.position;
    *grad_evalpos = grad_t;
    grad_gaussian->position = -grad_t;
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES void evaluate(const Gaussian<DIMS, scalar_t>& gaussian, const glm::vec<DIMS, scalar_t>& evalpos,
                                Gaussian<DIMS, scalar_t>* grad_gaussian, glm::vec<DIMS, scalar_t>* grad_evalpos,
                                scalar_t incoming_grad) {
    const auto covInv = glm::inverse(gaussian.covariance);
    // return evaluate_inversed(Gaussian(weight, pos, covInv), evalpos)

    gpe::grad::evaluate_inversed(Gaussian<DIMS, scalar_t>(gaussian.weight, gaussian.position, covInv), evalpos, grad_gaussian, grad_evalpos, incoming_grad);
    grad_gaussian->covariance = gpe::grad::inverse_with_cached_covInv(covInv, grad_gaussian->covariance);
}

template <typename scalar_t, int N_DIMS, int N_VIRTUAL_POINTS = 4>
EXECUTION_DEVICES void likelihood(const gpe::Gaussian<N_DIMS, scalar_t>& target, const gpe::Gaussian<N_DIMS, scalar_t>& fitting,
                                      gpe::Gaussian<N_DIMS, scalar_t>* grad_target, gpe::Gaussian<N_DIMS, scalar_t>* grad_fitting,
                                      scalar_t incoming_grad) {
    using Mat = glm::mat<N_DIMS, N_DIMS, scalar_t>;
    const scalar_t a = gpe::evaluate(gpe::Gaussian<N_DIMS, scalar_t>{1, fitting.position, fitting.covariance}, target.position);
    const auto target_cov_inv = glm::inverse(fitting.covariance);
    const auto c = target_cov_inv * target.covariance;
    const scalar_t minus_half_trace = scalar_t(-0.5) * gpe::trace(c);
    const scalar_t b = gpe::exp(minus_half_trace);
    const scalar_t wi_bar = N_VIRTUAL_POINTS * target.weight;
    // pow(0, 0) gives nan in cuda with fast math
    const scalar_t ab = a * b;
    const scalar_t ab_clipped = gpe::Epsilon<scalar_t>::clip(ab);

    //    return gpe::pow(ab_clipped, wi_bar);
    scalar_t grad_ab_clipped, grad_wi_bar;
    gpe::grad::pow(ab_clipped, wi_bar, &grad_ab_clipped, &grad_wi_bar, incoming_grad);

//    scalar_t ab_clipped = gpe::Epsilon<scalar_t>::clip(ab);
    const scalar_t grad_ab = (ab > gpe::Epsilon<scalar_t>::small) ? grad_ab_clipped : scalar_t(0);
    const auto grad_a = grad_ab * b;
    const auto grad_b = grad_ab * a;

    // scalar_t wi_bar = N_VIRTUAL_POINTS * target.weight;
    grad_target->weight = grad_wi_bar * N_VIRTUAL_POINTS;

    // scalar_t b = gpe::exp(minus_half_trace);
    const scalar_t grad_minus_half_trace = gpe::grad::exp(minus_half_trace, grad_b);

    // const scalar_t minus_half_trace = scalar_t(-0.5) * gpe::trace(c);
    const Mat grad_c = Mat(1) * (scalar_t(-0.5) * grad_minus_half_trace);

    // const auto c = target_cov_inv * target.covariance;
    const auto grad_target_cov_inv = grad_c * target.covariance;
    grad_target->covariance += target_cov_inv * grad_c;

    // const auto target_cov_inv = glm::inverse(fitting.covariance);
    grad_fitting->covariance = gpe::grad::inverse(fitting.covariance, grad_target_cov_inv);

    // const scalar_t a = gpe::evaluate(gpe::Gaussian<N_DIMS, scalar_t>{1, fitting.position, fitting.covariance}, target.position);
    gpe::Gaussian<N_DIMS, scalar_t> grad_fitting_gnormal_gaussian;
    gpe::grad::evaluate(gpe::Gaussian<N_DIMS, scalar_t>{1, fitting.position, fitting.covariance}, target.position,
                        &grad_fitting_gnormal_gaussian, &(grad_target->position), grad_a);
    grad_fitting->position = grad_fitting_gnormal_gaussian.position;
    grad_fitting->covariance += grad_fitting_gnormal_gaussian.covariance;
    grad_fitting->weight = 0;
}

} // namespace grad

} // namespace gpe

#endif // GPE_UTIL_GRAD_MATRIX_H
