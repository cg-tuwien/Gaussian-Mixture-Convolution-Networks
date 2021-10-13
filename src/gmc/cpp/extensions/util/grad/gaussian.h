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
    gradless_scalar_t factor = gcem::pow(2 * glm::pi<gradless_scalar_t>(), -gradless_scalar_t(DIMS) / gradless_scalar_t(2.));

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
} // namespace grad

} // namespace gpe

#endif // GPE_UTIL_GRAD_MATRIX_H
