#ifndef GPE_PIECES_BINDING_H
#define GPE_PIECES_BINDING_H

#include <torch/types.h>

namespace pieces {
torch::Tensor integrate_inversed(const torch::Tensor& mixture);
torch::Tensor integrate(const torch::Tensor& mixture);

at::Tensor matrix_inverse(const torch::Tensor& matrices);
// std::tuple<torch::Tensor, torch::Tensor> parallel_backward(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes,
//                                                            bool requires_grad_mixture, bool requires_grad_xes);
}

#endif // GPE_PIECES_BINDING_H
