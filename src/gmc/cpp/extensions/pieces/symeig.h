#ifndef GPE_PIECES_SYMEIG_H
#define GPE_PIECES_SYMEIG_H

#include <torch/types.h>


namespace pieces {
namespace symeig_impl {

std::tuple<torch::Tensor, torch::Tensor> forward(const torch::Tensor& matrices);
torch::Tensor backward(const torch::Tensor& matrices, const torch::Tensor& cached_values, const torch::Tensor& cached_vectors, const torch::Tensor& grad_values, const torch::Tensor& grad_vectors);

} // namespace symeig
} // namespace pieces

#endif // GPE_PIECES_SYMEIG_H
