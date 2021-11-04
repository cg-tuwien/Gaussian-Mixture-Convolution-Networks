#ifndef GPE_UTIL_GAUSSIAN_H
#define GPE_UTIL_GAUSSIAN_H

#include <gcem.hpp>

#include "util/autodiff.h"
#include "util/cuda.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "util/epsilon.h"

namespace gpe {

template<int N_DIMS, typename scalar_t>
struct Gaussian {
    using pos_t = glm::vec<N_DIMS, scalar_t>;
    using cov_t = glm::mat<N_DIMS, N_DIMS, scalar_t>;
    EXECUTION_DEVICES Gaussian() = default;
    EXECUTION_DEVICES Gaussian(const scalar_t& weight, const pos_t& position, const cov_t& covariance) : weight(weight), position(position), covariance(covariance) {}
    template<typename other_scalar>
    explicit EXECUTION_DEVICES Gaussian(const Gaussian<N_DIMS, other_scalar>& other) {
        weight = scalar_t(other.weight);
        for (unsigned i = 0; i < N_DIMS; ++i) {
            position[i] = scalar_t(other.position[i]);
            for (unsigned j = 0; j < N_DIMS; ++j) {
                covariance[i][j] = scalar_t(other.covariance[i][j]);
            }
        }
    }

    EXECUTION_DEVICES
    Gaussian& operator += (const Gaussian& other) {
        weight += other.weight;
        position += other.position;
        covariance += other.covariance;
        return *this;
    }

    scalar_t weight = 0;
    pos_t position = pos_t(0);
    cov_t covariance = cov_t(0);
};
static_assert (sizeof (Gaussian<2, float>) == 7*4, "Something wrong with Gaussian");
static_assert (sizeof (Gaussian<3, float>) == 13*4, "Something wrong with Gaussian");
static_assert (sizeof (Gaussian<2, double>) == 7*8, "Something wrong with Gaussian");
static_assert (sizeof (Gaussian<3, double>) == 13*8, "Something wrong with Gaussian");

#ifdef GPE_AUTODIFF
template <int N_DIMS, typename scalar_t>
gpe::Gaussian<N_DIMS, scalar_t> removeGrad(const gpe::Gaussian<N_DIMS, autodiff::Variable<scalar_t>>& g) {
    gpe::Gaussian<N_DIMS, scalar_t> r;
    r.weight = removeGrad(g.weight);
    r.position = removeGrad(g.position);
    r.covariance = removeGrad(g.covariance);
    return r;
}
template <int N_DIMS, typename scalar_t>
gpe::Gaussian<N_DIMS, scalar_t> removeGrad(const gpe::Gaussian<N_DIMS, scalar_t>& g) {
    return g;
}
template<int N_DIMS, typename scalar_t>
gpe::Gaussian<N_DIMS, autodiff::Variable<scalar_t>> makeAutodiff(const gpe::Gaussian<N_DIMS, scalar_t>& g) {
    return {makeAutodiff(g.weight), makeAutodiff(g.position), makeAutodiff(g.covariance)};
}
template<int N_DIMS, typename scalar_t>
gpe::Gaussian<N_DIMS, scalar_t> extractGrad(const gpe::Gaussian<N_DIMS, autodiff::Variable<scalar_t>>& g) {
    return {extractGrad(g.weight), extractGrad(g.position), extractGrad(g.covariance)};
}
template<int N_DIMS, typename scalar_t>
void propagateGrad(const gpe::Gaussian<N_DIMS, autodiff::Variable<scalar_t>>& v, const gpe::Gaussian<N_DIMS, scalar_t>& grad) {
    propagateGrad(v.weight, grad.weight);
    propagateGrad(v.position, grad.position);
    propagateGrad(v.covariance, grad.covariance);
}
#endif

template <int DIMS, typename scalar_t>
EXECUTION_DEVICES bool isnan(const Gaussian<DIMS, scalar_t>& g) {
    return gpe::isnan(g.weight) || gpe::isnan(g.position) || gpe::isnan(g.covariance);
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t evaluate_inversed(const Gaussian<DIMS, scalar_t>& gaussian, const glm::vec<DIMS, scalar_t>& evalpos) {
    using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;
#ifdef _MSC_VER
    gradless_scalar_t factor = gcem::pow(2 * glm::pi<gradless_scalar_t>(), -gradless_scalar_t(DIMS) / gradless_scalar_t(2.));
#else
    constexpr gradless_scalar_t factor = gcem::pow(2 * glm::pi<gradless_scalar_t>(), -gradless_scalar_t(DIMS) / gradless_scalar_t(2.));
#endif

    const auto t = evalpos - gaussian.position;
    const auto v = scalar_t(-0.5) * glm::dot(t, (gaussian.covariance * t));
    const auto norm = gpe::sqrt(glm::determinant(gaussian.covariance)) * factor;
    return gaussian.weight * norm * gpe::exp(v);
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t evaluate(const Gaussian<DIMS, scalar_t>& gaussian, const glm::vec<DIMS, scalar_t>& evalpos) {
    return evaluate_inversed(gpe::Gaussian<DIMS, scalar_t>{gaussian.weight, gaussian.position, glm::inverse(gaussian.covariance)}, evalpos);
}

// todo: numerical problems when N_VIRTUAL_POINTS is large: a*b for instance 0.001, wi_bar becomes 5.6 -> bad things
// that depends on cov magnitude => better normalise mixture to have covs in the magnitude of the identity
template <typename scalar_t, int N_DIMS, int N_VIRTUAL_POINTS = 4>
EXECUTION_DEVICES scalar_t likelihood(const gpe::Gaussian<N_DIMS, scalar_t>& target, const gpe::Gaussian<N_DIMS, scalar_t>& fitting) {
    // Continuous projection for fast L 1 reconstruction: Equation 9
    scalar_t a = gpe::evaluate(gpe::Gaussian<N_DIMS, scalar_t>{1, fitting.position, fitting.covariance}, target.position);
    auto c = glm::inverse(fitting.covariance) * target.covariance;
    scalar_t b = gpe::exp(scalar_t(-0.5) * gpe::trace(c));
    scalar_t wi_bar = N_VIRTUAL_POINTS * target.weight;
    // pow(0, 0) gives nan in cuda with fast math
    return gpe::pow(gpe::Epsilon<scalar_t>::clip(a * b), wi_bar);
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

} // namespace gpe

#endif // GPE_UTIL_GAUSSIAN_H
