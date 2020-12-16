#ifndef GPE_UTIL_GRAD_MATRIX_H
#define GPE_UTIL_GRAD_MATRIX_H

#include <gcem.hpp>

#include "util/cuda.h"
#include "util/glm.h"
#include "util/grad/glm.h"
#include "util/scalar.h"
#include "util/gaussian.h"

namespace gpe {
namespace grad {

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES glm::mat<DIMS, DIMS, scalar_t> gaussian_amplitude(const glm::mat<DIMS, DIMS, scalar_t>& cov, scalar_t grad) {
    constexpr auto a = gcem::pow(scalar_t(2) * glm::pi<scalar_t>(), - DIMS * scalar_t(0.5));
    assert(glm::determinant(cov) > 0);
    const auto d = glm::determinant(cov);
    const auto k = (a * scalar_t(-0.5)) / gpe::sqrt(d * d * d);
    return k * grad::determinant(cov, grad);
}

template <typename scalar_t, int N_DIMS, int N_VIRTUAL_POINTS = 4>
EXECUTION_DEVICES scalar_t likelihood(const gpe::Gaussian<N_DIMS, scalar_t>& target, const gpe::Gaussian<N_DIMS, scalar_t>& fitting,
                                      gpe::Gaussian<N_DIMS, scalar_t>* grad_target, gpe::Gaussian<N_DIMS, scalar_t>* grad_fitting,
                                      scalar_t incoming_grad) {
    scalar_t normal_amplitude = gpe::gaussian_amplitude(fitting.covariance);
    scalar_t a = gpe::evaluate(target.position, normal_amplitude, fitting.position, fitting.covariance);
    auto c = glm::inverse(fitting.covariance) * target.covariance;
    scalar_t b = gpe::exp(scalar_t(-0.5) * gpe::trace(c));
    scalar_t target_normal_amplitudes = gpe::gaussian_amplitude(target.covariance);
    scalar_t wi_bar = N_VIRTUAL_POINTS * target.weight / target_normal_amplitudes;
    // pow(0, 0) gives nan in cuda with fast math
    scalar_t ab = a * b;
    scalar_t ab_clipped = gpe::Epsilon<scalar_t>::clip(ab);
//    return gpe::pow(ab_clipped, wi_bar);

    // pow(0, 0) gives nan in cuda with fast math
//    return gpe::pow(gpe::Epsilon<scalar_t>::clip(a * b), wi_bar);

//    scalar_t wi_bar = N_VIRTUAL_POINTS * target.weight / target_normal_amplitudes;
//    scalar_t target_normal_amplitudes = gpe::gaussian_amplitude(target.covariance);
//    scalar_t b = gpe::exp(scalar_t(-0.5) * gpe::trace(c));
//    auto c = glm::inverse(fitting.covariance) * target.covariance;
//    scalar_t a = gpe::evaluate(target.position, normal_amplitude, fitting.position, fitting.covariance);
//    scalar_t normal_amplitude = gpe::gaussian_amplitude(fitting.covariance);
}

} // namespace grad

} // namespace gpe

#endif // GPE_UTIL_GRAD_MATRIX_H
