#ifndef COMMON_H
#define COMMON_H

#include <torch/extension.h>
#include <vector>

namespace gm {

int n_batch(torch::Tensor mixture) {
    return mixture.size(0);
}

int n_layers(torch::Tensor mixture) {
    return mixture.size(1);
}

int n_components(torch::Tensor mixture) {
    return mixture.size(2);
}

int n_dimensions(torch::Tensor mixture) {
    auto vector_length = mixture.size(3);
    if (vector_length == 7)
        return 2;
    if (vector_length == 13)
        return 3;

    TORCH_CHECK(false, "mixture must have 7 or 13 elements in the last dimension")
    return 0;
}

torch::Tensor weights(torch::Tensor mixture) {
    using namespace torch::indexing;
    return mixture.index({Slice(), Slice(), Slice(), 0});
}

torch::Tensor positions(torch::Tensor mixture) {
    using namespace torch::indexing;
    return mixture.index({Slice(), Slice(), Slice(), Slice(1, n_dimensions(mixture) + 1)});
}

torch::Tensor covariances(torch::Tensor mixture) {
    using namespace torch::indexing;
    auto n_dims = n_dimensions(mixture);
    std::vector<int64_t> new_shape = mixture.sizes().vec();
    new_shape.back() = n_dims;
    new_shape.push_back(n_dims);

    return mixture.index({Slice(), Slice(), Slice(), Slice(n_dimensions(mixture) + 1, None)}).view(new_shape);
}

void check_mixture(torch::Tensor mixture) {
    TORCH_CHECK(!torch::isnan(mixture).any().item<bool>(), "mixture contains NaNs");
    TORCH_CHECK(!torch::isinf(mixture).any().item<bool>(), "mixture contains infinities");
    TORCH_CHECK(mixture.dim() == 4, "mixture must have 4 dimensions");
    auto n_dims = n_dimensions(mixture);
    TORCH_CHECK(n_dims == 2 || n_dims == 3);
    TORCH_CHECK(torch::all(covariances(mixture).det() > 0).item<bool>(), "mixture contains non positive definite covariances");
}

}

#endif // COMMON_H
