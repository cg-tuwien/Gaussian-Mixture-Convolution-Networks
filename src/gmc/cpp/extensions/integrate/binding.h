#ifndef INTEGRATE_BINDING_H
#define INTEGRATE_BINDING_H

#include <torch/types.h>
#include <tuple>

namespace integrate {
torch::Tensor inversed_forward(const torch::Tensor& mixture);
torch::Tensor forward(const torch::Tensor& mixture);
// std::tuple<torch::Tensor, torch::Tensor> parallel_backward(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes,
//                                                            bool requires_grad_mixture, bool requires_grad_xes);
}

#endif // INTEGRATE_BINDING_H
