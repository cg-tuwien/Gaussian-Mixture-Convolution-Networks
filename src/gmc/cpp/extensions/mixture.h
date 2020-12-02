#ifndef MIXTURE_H
#define MIXTURE_H
#include <vector>
#include <iostream>
#include <type_traits>

#include <gcem.hpp>
#include <torch/types.h>

#include "math/gpe_glm.h"
#include "math/matrix.h"
#include "math/scalar.h"
#include "util/autodiff.h"
#include "util/cuda.h"

namespace gpe {


struct MixtureAndXesNs {
    int batch = 0;
    int layers = 0;
    int components = 0;
    int dims = 0;
    int batch_xes = 0;
    int layers_xes = 0;
    int xes = 0;
};
struct MixtureNs {
    int batch = 0;
    int layers = 0;
    int components = 0;
    int dims = 0;
};

inline int n_batch(torch::Tensor mixture) {
    return int(mixture.size(0));
}

inline int n_layers(torch::Tensor mixture) {
    return int(mixture.size(1));
}

inline int n_components(torch::Tensor mixture) {
    return int(mixture.size(2));
}

inline int n_dimensions(torch::Tensor mixture) {
    auto vector_length = mixture.size(-1);
    if (vector_length == 7)
        return 2;
    if (vector_length == 13)
        return 3;

    TORCH_CHECK(false, "mixture must have 7 or 13 elements in the last dimension")
}

inline torch::Tensor weights(torch::Tensor mixture) {
    using namespace torch::indexing;
    return mixture.index({Ellipsis, 0});
}

inline torch::Tensor positions(torch::Tensor mixture) {
    using namespace torch::indexing;
    return mixture.index({Ellipsis, Slice(1, n_dimensions(mixture) + 1)});
}

inline torch::Tensor covariances(torch::Tensor mixture) {
    using namespace torch::indexing;
    auto n_dims = n_dimensions(mixture);
    std::vector<int64_t> new_shape = mixture.sizes().vec();
    new_shape.back() = n_dims;
    new_shape.push_back(n_dims);

    return mixture.index({Ellipsis, Slice(n_dimensions(mixture) + 1, None)}).view(new_shape);
}

inline torch::Tensor pack_mixture(const torch::Tensor weights, const torch::Tensor positions, const torch::Tensor covariances) {
    const auto n_batch = weights.size(0);
    const auto n_layers = weights.size(1);
    const auto n_components = weights.size(2);
    TORCH_CHECK(positions.size(0) == n_batch);
    TORCH_CHECK(covariances.size(0) == n_batch);
    TORCH_CHECK(positions.size(1) == n_layers);
    TORCH_CHECK(covariances.size(1) == n_layers);
    TORCH_CHECK(positions.size(2) == n_components);
    TORCH_CHECK(covariances.size(2) == n_components);

    const auto n_dims = positions.size(3);
    TORCH_CHECK(covariances.size(3) == n_dims);
    TORCH_CHECK(covariances.size(4) == n_dims);

    return torch::cat({weights.view({n_batch, n_layers, n_components, 1}), positions, covariances.view({n_batch, n_layers, n_components, n_dims * n_dims})}, 3);
}

inline torch::Tensor mixture_with_inversed_covariances(torch::Tensor mixture) {
    const auto weights = torch::abs(gpe::weights(mixture));
    const auto positions = gpe::positions(mixture);
    const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
    return gpe::pack_mixture(weights, positions, invCovs.contiguous());
}


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
    scalar_t weight = 0;
    pos_t position = pos_t(0);
    cov_t covariance = cov_t(1);
};
static_assert (sizeof (Gaussian<2, float>) == 7*4, "Something wrong with Gaussian");
static_assert (sizeof (Gaussian<3, float>) == 13*4, "Something wrong with Gaussian");
static_assert (sizeof (Gaussian<2, double>) == 7*8, "Something wrong with Gaussian");
static_assert (sizeof (Gaussian<3, double>) == 13*8, "Something wrong with Gaussian");

#ifndef __CUDACC__
template <int N_DIMS, typename scalar_t>
gpe::Gaussian<N_DIMS, scalar_t> removeGrad(const gpe::Gaussian<N_DIMS, autodiff::Variable<scalar_t>>& g) {
    gpe::Gaussian<N_DIMS, scalar_t> r;
    r.weight = removeGrad(g.weight);
    r.position = removeGrad(g.position);
    r.covariance = removeGrad(g.covariance);
    return r;
}
#endif

template<typename scalar_t>
void printGaussian(const Gaussian<2, scalar_t>& g) {
    printf("g(w=%f, p=%f/%f, c=%f/%f//%f\n", g.weight, g.position.x, g.position.y, g.covariance[0][0], g.covariance[0][1], g.covariance[1][1]);
}
template<typename scalar_t>
void printGaussian(const Gaussian<3, scalar_t>& g) {
    printf("g(w=%f, p=%f/%f/%f, c=%f/%f/%f//%f/%f//%f\n",
           g.weight,
           g.position.x, g.position.y, g.position.z,
           g.covariance[0][0], g.covariance[0][1], g.covariance[0][2],
           g.covariance[1][1], g.covariance[1][2],
           g.covariance[2][2]);
}

template<int N_DIMS, typename scalar_t>
std::ostream& operator <<(std::ostream& stream, const Gaussian<N_DIMS, scalar_t>& g) {
    stream << "Gauss[" << g.weight << "; " << g.position[0];
    for (int i = 1; i < N_DIMS; i++)
        stream << "/" << g.position[i];
    stream << "; ";

    for (int i = 0; i < N_DIMS; i++) {
        for (int j = 0; j < N_DIMS; j++) {
            if (i != 0 || j != 0)
                stream << "/";
            stream << g.covariance[i][j];
        }
    }
    stream << "]";
    return stream;
}

template <typename TensorAccessor>
EXECUTION_DEVICES auto weight(TensorAccessor&& gaussian) -> decltype (gaussian[0]) {
    return gaussian[0];
}

template <int DIMS, typename TensorAccessor>
EXECUTION_DEVICES auto position(TensorAccessor&& gaussian) -> decltype (gpe::vec<DIMS>(gaussian[1])) {
    return gpe::vec<DIMS>(gaussian[1]);
}

template <int DIMS, typename TensorAccessor>
EXECUTION_DEVICES auto covariance(TensorAccessor&& gaussian) -> decltype (gpe::mat<DIMS>(gaussian[1 + DIMS])) {
    return gpe::mat<DIMS>(gaussian[1 + DIMS]);
}
template <int DIMS, typename TensorAccessor>
EXECUTION_DEVICES auto gaussian(TensorAccessor&& gaussian) -> Gaussian<DIMS, gpe::remove_cvref_t<decltype (gaussian[0])>>& {
    return reinterpret_cast<Gaussian<DIMS, gpe::remove_cvref_t<decltype (gaussian[0])>>&>(gaussian[0]);
}
template <int DIMS, typename TensorAccessor>
EXECUTION_DEVICES auto gaussian(const TensorAccessor&& gaussian) -> const Gaussian<DIMS, gpe::remove_cvref_t<decltype (gaussian[0])>>& {
    return reinterpret_cast<const Gaussian<DIMS, gpe::remove_cvref_t<decltype (gaussian[0])>>&>(gaussian[0]);
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t evaluate_inversed(const glm::vec<DIMS, scalar_t>& evalpos,
                                                               const scalar_t& weight,
                                                               const glm::vec<DIMS, scalar_t>& pos,
                                                               const glm::mat<DIMS, DIMS, scalar_t>& inversed_cov) {
    const auto t = evalpos - pos;
    const auto v = scalar_t(-0.5) * glm::dot(t, (inversed_cov * t));
    return weight * gpe::exp(v);
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t evaluate(const glm::vec<DIMS, scalar_t>& evalpos,
                                                      const scalar_t& weight,
                                                      const glm::vec<DIMS, scalar_t>& pos,
                                                      const glm::mat<DIMS, DIMS, scalar_t>& cov) {
    const auto t = evalpos - pos;
    const auto v = scalar_t(-0.5) * glm::dot(t, (glm::inverse(cov) * t));
    return weight * gpe::exp(v);
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t evaluate_inversed(const Gaussian<DIMS, scalar_t>& gaussian,
                                                               const glm::vec<DIMS, scalar_t>& evalpos) {
    const auto t = evalpos - gaussian.position;
    const auto v = scalar_t(-0.5) * glm::dot(t, (gaussian.covariance * t));
    return gaussian.weight * gpe::exp(v);
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t evaluate(const Gaussian<DIMS, scalar_t>& gaussian,
                                                      const glm::vec<DIMS, scalar_t>& evalpos) {
    const auto t = evalpos - gaussian.position;
    const auto v = scalar_t(-0.5) * glm::dot(t, (glm::inverse(gaussian.covariance) * t));
    return gaussian.weight * gpe::exp(v);
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t integrate_inversed(const Gaussian<DIMS, scalar_t>& gaussian) {
    using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;
    constexpr gradless_scalar_t factor = gcem::pow(2 * glm::pi<gradless_scalar_t>(), gradless_scalar_t(DIMS));
    return gaussian.weight * gpe::sqrt(factor / glm::determinant(gaussian.covariance));
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t integrate(const Gaussian<DIMS, scalar_t>& gaussian) {
    using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;
    constexpr gradless_scalar_t factor = gcem::pow(2 * glm::pi<gradless_scalar_t>(), gradless_scalar_t(DIMS));
    return gaussian.weight * gpe::sqrt(factor * glm::determinant(gaussian.covariance));
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t gaussian_amplitude_inversed(const glm::mat<DIMS, DIMS, scalar_t>& inversed_cov) {
    using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;
    constexpr auto a = gcem::pow(scalar_t(2) * glm::pi<scalar_t>(), - DIMS * scalar_t(0.5));
    assert(glm::determinant(inversed_cov) > 0);
    return a * gpe::sqrt(glm::determinant(inversed_cov));
}

template <typename scalar_t, int DIMS>
EXECUTION_DEVICES scalar_t gaussian_amplitude(const glm::mat<DIMS, DIMS, scalar_t>& cov) {
    using gradless_scalar_t = gpe::remove_grad_t<scalar_t>;
    constexpr auto a = gcem::pow(gradless_scalar_t(2) * glm::pi<gradless_scalar_t>(), - DIMS * gradless_scalar_t(0.5));
    assert(glm::determinant(cov) > 0);
    return a / gpe::sqrt(glm::determinant(cov));
}

inline void check_mixture(torch::Tensor mixture) {
    TORCH_CHECK(mixture.is_contiguous(), "mixture must be contiguous")
    TORCH_CHECK(!torch::isnan(mixture).any().item<bool>(), "mixture contains NaNs");
    TORCH_CHECK(!torch::isinf(mixture).any().item<bool>(), "mixture contains infinities");
    TORCH_CHECK(mixture.dim() == 4, "mixture must have 4 dimensions");
    auto n_dims = n_dimensions(mixture);
    TORCH_CHECK(n_dims == 2 || n_dims == 3);
    TORCH_CHECK(torch::all(covariances(mixture).det() > 0).item<bool>(), "mixture contains non positive definite covariances");
}


inline MixtureNs get_ns(torch::Tensor mixture) {
    check_mixture(mixture);

    auto n_batch = gpe::n_batch(mixture);
    auto n_layers = gpe::n_layers(mixture);
    auto n_components = gpe::n_components(mixture);
    auto n_dims = gpe::n_dimensions(mixture);

    return {n_batch, n_layers, n_components, n_dims};
}

inline MixtureAndXesNs check_input_and_get_ns(torch::Tensor mixture, torch::Tensor xes) {
    check_mixture(mixture);

    auto n_batch = gpe::n_batch(mixture);
    auto n_layers = gpe::n_layers(mixture);
    auto n_components = gpe::n_components(mixture);
    auto n_dims = gpe::n_dimensions(mixture);

    TORCH_CHECK(xes.is_contiguous(), "xes must be contiguous")
    TORCH_CHECK(xes.dim() == 4, "xes must have 4 dimensions");
    TORCH_CHECK(xes.dtype() == mixture.dtype(), "mixture and xes must have the same dtype");
    TORCH_CHECK(xes.device() == mixture.device(), "mixture and xes must have the same device");
    auto n_batch_xes = int(xes.size(0));
    auto n_layers_xes = int(xes.size(1));
    auto n_xes = int(xes.size(2));

    TORCH_CHECK(n_batch_xes == 1 || n_batch_xes == n_batch, "xes must have a batch dimension equal to one or the mixture");
    TORCH_CHECK(n_layers_xes == 1 || n_layers_xes == n_layers, "xes must have a layer dimension equal to one or the mixture");
    TORCH_CHECK(xes.size(3) == n_dims, "xes must have the last dimension equal to the number of dimensions of the mixture");
    return {n_batch, n_layers, n_components, n_dims, n_batch_xes, n_layers_xes, n_xes};
}

} // namespace gpe

#endif // MIXTURE_H
