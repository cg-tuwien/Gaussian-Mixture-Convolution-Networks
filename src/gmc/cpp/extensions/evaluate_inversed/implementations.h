#ifndef EVALUATE_INVERSED_PARALLEL_IMPLEMENTATION_H
#define EVALUATE_INVERSED_PARALLEL_IMPLEMENTATION_H

#include <torch/script.h>

#include "lbvh/Config.h"

at::Tensor parallel_forward_impl(const torch::Tensor& mixture, const torch::Tensor& xes);

std::tuple<torch::Tensor, torch::Tensor> parallel_backward_impl(const torch::Tensor& grad_output,
                                                                const torch::Tensor& mixture,
                                                                const torch::Tensor& xes,
                                                                bool requires_grad_mixture, bool requires_grad_xes);

at::Tensor parallel_forward_optimised_impl(const torch::Tensor& mixture, const torch::Tensor& xes);

std::tuple<torch::Tensor, torch::Tensor> parallel_backward_optimised_impl(const torch::Tensor& grad_output,
                                                                          const torch::Tensor& mixture,
                                                                          const torch::Tensor& xes,
                                                                          bool requires_grad_mixture, bool requires_grad_xes);


#endif // EVALUATE_INVERSED_PARALLEL_IMPLEMENTATION_H
