#ifndef COMMON_H
#define COMMON_H

#include <torch/extension.h>

namespace gm {

int64_t n_batch(torch::Tensor mixture) {
    AT_ASSERTM(mixture.dim() == 4, "mixture must have 4 dimensions");
    return mixture.size(0);
}

int64_t n_layers(torch::Tensor mixture) {
    AT_ASSERTM(mixture.dim() == 4, "mixture must have 4 dimensions");
    return mixture.size(1);
}

int64_t n_components(torch::Tensor mixture) {
    AT_ASSERTM(mixture.dim() == 4, "mixture must have 4 dimensions");
    return mixture.size(2);
}

int64_t n_dimensions(torch::Tensor mixture) {
    AT_ASSERTM(mixture.dim() == 4, "mixture must have 4 dimensions");

    auto vector_length = mixture.size(3);
    if (vector_length == 7)
        return 2;
    if (vector_length == 13)
        return 3;

    AT_ASSERTM(false, "mixture must have 7 or 13 elements in the last dimension");
    return 0;
}

}

#endif // COMMON_H
