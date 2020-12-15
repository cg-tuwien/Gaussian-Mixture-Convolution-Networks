#ifndef GPE_UTIL_GRAD_MATRIX_H
#define GPE_UTIL_GRAD_MATRIX_H

#include <gcem.hpp>

#include "util/cuda.h"
#include "util/glm.h"
#include "util/grad/glm.h"
#include "util/scalar.h"

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

} // namespace grad

} // namespace gpe

#endif // GPE_UTIL_GRAD_MATRIX_H
