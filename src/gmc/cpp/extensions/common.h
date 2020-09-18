#ifndef COMMON_H
#define COMMON_H

//#include <torch/extension.h>
#include <torch/script.h>
#include <vector>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "math/scalar.h"
#include "math/matrix.h"

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#ifndef __forceinline__
#define __forceinline__ inline
#endif

using uint = unsigned int;

namespace gpe {


struct MixtureAndXesNs {
    uint batch = 0;
    uint layers = 0;
    uint components = 0;
    uint dims = 0;
    uint batch_xes = 0;
    uint layers_xes = 0;
    uint xes = 0;
};
struct MixtureNs {
    uint batch = 0;
    uint layers = 0;
    uint components = 0;
    uint dims = 0;
};

inline uint n_batch(torch::Tensor mixture) {
    return uint(mixture.size(0));
}

inline uint n_layers(torch::Tensor mixture) {
    return uint(mixture.size(1));
}

inline uint n_components(torch::Tensor mixture) {
    return uint(mixture.size(2));
}

inline uint n_dimensions(torch::Tensor mixture) {
    auto vector_length = mixture.size(3);
    if (vector_length == 7)
        return 2;
    if (vector_length == 13)
        return 3;

    TORCH_CHECK(false, "mixture must have 7 or 13 elements in the last dimension")
}

inline torch::Tensor weights(torch::Tensor mixture) {
    using namespace torch::indexing;
    return mixture.index({Slice(), Slice(), Slice(), 0});
}

inline torch::Tensor positions(torch::Tensor mixture) {
    using namespace torch::indexing;
    return mixture.index({Slice(), Slice(), Slice(), Slice(1, n_dimensions(mixture) + 1)});
}

inline torch::Tensor covariances(torch::Tensor mixture) {
    using namespace torch::indexing;
    auto n_dims = n_dimensions(mixture);
    std::vector<int64_t> new_shape = mixture.sizes().vec();
    new_shape.back() = n_dims;
    new_shape.push_back(n_dims);

    return mixture.index({Slice(), Slice(), Slice(), Slice(n_dimensions(mixture) + 1, None)}).view(new_shape);
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

template <typename Gaussian>
__forceinline__ __host__ __device__ auto weight(Gaussian&& gaussian) -> decltype (gaussian[0]) {
    return gaussian[0];
}

template <int DIMS, typename Gaussian>
__forceinline__ __host__ __device__ auto position(Gaussian&& gaussian) -> decltype (vec<DIMS>(gaussian[1])) {
    return vec<DIMS>(gaussian[1]);
}

template <int DIMS, typename Gaussian>
__forceinline__ __host__ __device__ auto covariance(Gaussian&& gaussian) -> decltype (mat<DIMS>(gaussian[1 + DIMS])) {
    return mat<DIMS>(gaussian[1 + DIMS]);
}


template <typename scalar_t, int DIMS>
__forceinline__ __host__ __device__ scalar_t evaluate_gaussian(const glm::vec<DIMS, scalar_t>& evalpos,
                                                               const scalar_t& weight,
                                                               const glm::vec<DIMS, scalar_t>& pos,
                                                               const glm::mat<DIMS, DIMS, scalar_t>& inversed_cov) {
    const auto t = evalpos - pos;
    const auto v = scalar_t(-0.5) * glm::dot(t, (inversed_cov * t));
    return weight * gpe::exp(v);
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

    uint n_batch = gpe::n_batch(mixture);
    uint n_layers = gpe::n_layers(mixture);
    uint n_components = gpe::n_components(mixture);
    uint n_dims = gpe::n_dimensions(mixture);

    return {n_batch, n_layers, n_components, n_dims};
}

inline MixtureAndXesNs check_input_and_get_ns(torch::Tensor mixture, torch::Tensor xes) {
    check_mixture(mixture);

    uint n_batch = gpe::n_batch(mixture);
    uint n_layers = gpe::n_layers(mixture);
    uint n_components = gpe::n_components(mixture);
    uint n_dims = gpe::n_dimensions(mixture);

    TORCH_CHECK(xes.is_contiguous(), "xes must be contiguous")
    TORCH_CHECK(xes.dim() == 4, "xes must have 4 dimensions");
    TORCH_CHECK(xes.dtype() == mixture.dtype(), "mixture and xes must have the same dtype");
    TORCH_CHECK(xes.device() == mixture.device(), "mixture and xes must have the same device");
    uint n_batch_xes = uint(xes.size(0));
    uint n_layers_xes = uint(xes.size(1));
    uint n_xes = uint(xes.size(2));

    TORCH_CHECK(n_batch_xes == 1 || n_batch_xes == n_batch, "xes must have a batch dimension equal to one or the mixture");
    TORCH_CHECK(n_layers_xes == 1 || n_layers_xes == n_layers, "xes must have a layer dimension equal to one or the mixture");
    TORCH_CHECK(xes.size(3) == n_dims, "xes must have the last dimension equal to the number of dimensions of the mixture");
    return {n_batch, n_layers, n_components, n_dims, n_batch_xes, n_layers_xes, n_xes};
}

}

#endif // COMMON_H
