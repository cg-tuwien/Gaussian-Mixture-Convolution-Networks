#ifndef GPE_UTIL_MIXTURE_H
#define GPE_UTIL_MIXTURE_H
#include <torch/types.h>

#include "util/cuda.h"
#include "pieces/pieces.h"

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
    TORCH_CHECK(positions.size(0) == n_batch)
    TORCH_CHECK(covariances.size(0) == n_batch)
    TORCH_CHECK(positions.size(1) == n_layers)
    TORCH_CHECK(covariances.size(1) == n_layers)
    TORCH_CHECK(positions.size(2) == n_components)
    TORCH_CHECK(covariances.size(2) == n_components)

    const auto n_dims = positions.size(3);
    TORCH_CHECK(covariances.size(3) == n_dims)
    TORCH_CHECK(covariances.size(4) == n_dims)

    return torch::cat({weights.view({n_batch, n_layers, n_components, 1}), positions, covariances.view({n_batch, n_layers, n_components, n_dims * n_dims})}, 3);
}

inline torch::Tensor mixture_with_inversed_covariances(torch::Tensor mixture) {
    const auto weights = torch::abs(gpe::weights(mixture));
    const auto positions = gpe::positions(mixture);
//    const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
    const auto invCovs = pieces::matrix_inverse(gpe::covariances(mixture));
    return gpe::pack_mixture(weights, positions, invCovs.contiguous());
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
    //check_mixture(mixture);

    auto n_batch = gpe::n_batch(mixture);
    auto n_layers = gpe::n_layers(mixture);
    auto n_components = gpe::n_components(mixture);
    auto n_dims = gpe::n_dimensions(mixture);

    return {n_batch, n_layers, n_components, n_dims};
}

inline MixtureAndXesNs check_input_and_get_ns(torch::Tensor mixture, torch::Tensor xes) {
    //check_mixture(mixture);

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

inline torch::Tensor aabb_from_positions(const torch::Tensor& positions) {
    using namespace torch::indexing;
    const auto upper = std::get<0>(positions.max(-2));
    const auto lower = std::get<0>(positions.min(-2));
    const auto zeroes = torch::zeros({positions.size(0), positions.size(1), 4 - positions.size(3)}, torch::TensorOptions(positions.device()).dtype(positions.dtype()));
    return torch::cat({upper, zeroes, lower, zeroes}, -1).contiguous();
}

} // namespace gpe

#endif // GPE_UTIL_MIXTURE_H
