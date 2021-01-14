#ifndef INTEGRATE_IMPLEMENTATION_H
#define INTEGRATE_IMPLEMENTATION_H

#include <torch/script.h>

namespace integrate {

template <bool INVERSED>
at::Tensor forward_impl(const torch::Tensor& mixture);

extern template at::Tensor forward_impl<true>(const torch::Tensor& mixture);
extern template at::Tensor forward_impl<false>(const torch::Tensor& mixture);

/*
std::tuple<torch::Tensor, torch::Tensor> parallel_backward_impl(const torch::Tensor& grad_output,
                                                                const torch::Tensor& mixture,
                                                                const torch::Tensor& xes,
                                                                bool requires_grad_mixture, bool requires_grad_xes);
*/
} // namespace integrate

#endif // INTEGRATE_IMPLEMENTATION_H
