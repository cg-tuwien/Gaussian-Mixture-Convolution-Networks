#ifndef GPE_PIECES_BINDING_H
#define GPE_PIECES_BINDING_H

#include <torch/types.h>

namespace pieces {
torch::Tensor matrix_inverse(const torch::Tensor& matrices);

std::tuple<torch::Tensor, torch::Tensor> symeig(const torch::Tensor& matrices);
torch::Tensor symeig_backward(const torch::Tensor& matrices, const torch::Tensor& cached_values, const torch::Tensor& cached_vectors, const torch::Tensor& grad_values, const torch::Tensor& grad_vectors);
}

#endif // GPE_PIECES_BINDING_H
