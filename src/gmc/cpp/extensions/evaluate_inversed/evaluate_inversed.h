#ifndef GPE_EVALUATE_INVERSED_H
#define GPE_EVALUATE_INVERSED_H

#include <torch/types.h>
#include <tuple>

namespace evaluate_inversed {

std::tuple<torch::Tensor> parallel_forward(const torch::Tensor& mixture, const torch::Tensor& xes);
std::tuple<torch::Tensor, torch::Tensor> parallel_backward(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes,
                                                           const std::tuple<torch::Tensor>& forward_out,
                                                           bool requires_grad_mixture, bool requires_grad_xes);

}

#endif // GPE_EVALUATE_INVERSED_H
