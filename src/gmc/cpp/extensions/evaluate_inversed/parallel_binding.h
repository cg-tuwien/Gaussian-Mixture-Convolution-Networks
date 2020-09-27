#ifndef PARALLEL_BINDING_H
#define PARALLEL_BINDING_H

#include <torch/types.h>
#include <tuple>

torch::Tensor parallel_forward(const torch::Tensor& mixture, const torch::Tensor& xes);
std::tuple<torch::Tensor, torch::Tensor> parallel_backward(const torch::Tensor& grad_output, const torch::Tensor& mixture, const torch::Tensor& xes,
                                                           bool requires_grad_mixture, bool requires_grad_xes);

#endif // PARALLEL_BINDING_H
