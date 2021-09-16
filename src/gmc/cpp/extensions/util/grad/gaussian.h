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
EXECUTION_DEVICES void evaluate(const Gaussian<DIMS, scalar_t>& gaussian, const glm::vec<DIMS, scalar_t>& evalpos,
                                Gaussian<DIMS, scalar_t>* grad_gaussian, glm::vec<DIMS, scalar_t>* grad_evalpos,
                                scalar_t incoming_grad) {
    using Vec = glm::vec<DIMS, scalar_t>;
    const auto t = evalpos - gaussian.position;
    const auto covInv = glm::inverse(gaussian.covariance);
    const auto covInv_times_t = (covInv * t);
    const auto v = scalar_t(-0.5) * glm::dot(t, covInv_times_t);

//    return gaussian.weight * gpe::exp(v);
    grad_gaussian->weight = incoming_grad * gpe::exp(v);
    scalar_t grad_v = gpe::grad::exp(v, incoming_grad * gaussian.weight);

//    const auto v = scalar_t(-0.5) * glm::dot(t, covInv_times_t);
    Vec grad_t;
    Vec grad_covInv_times_t;
    gpe::grad::dot(t, covInv_times_t, &grad_t, &grad_covInv_times_t, grad_v * scalar_t(-0.5));

    // according to https://www.youtube.com/watch?v=R_m4kanPy6Q.
//    const auto covInv_times_t = (covInv * t);
    auto grad_covInv = gpe::outerProduct(grad_covInv_times_t, t);
    // ignoring the transpose due to symmetry
    grad_t += covInv * grad_covInv_times_t;

//    const auto covInv = glm::inverse(gaussian.covariance);
    grad_gaussian->covariance = gpe::grad::inverse_with_cached_covInv(covInv, grad_covInv);

//    const auto t = evalpos - gaussian.position;
    *grad_evalpos = grad_t;
    grad_gaussian->position = -grad_t;
}

} // namespace grad

} // namespace gpe

#endif // GPE_UTIL_GRAD_MATRIX_H
