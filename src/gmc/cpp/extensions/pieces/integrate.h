#ifndef GPE_PIECES_INTEGRATE_H
#define GPE_PIECES_INTEGRATE_H

#include <torch/types.h>

namespace pieces {

namespace integrate_impl {

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
} // namespace integrate_impl
} // namespace pieces

#endif // GPE_PIECES_INTEGRATE_H
