#include <tuple>
#include <torch/types.h>
#include "bvh_mhem_fit_alpha/Config.h"
#include "bvh_mhem_fit_alpha/implementation.h"

namespace bvh_mhem_fit_alpha {

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
ForwardOutput forward_impl_t(at::Tensor mixture, const Config& configForwardOutput);

extern template ForwardOutput forward_impl_t<2,  float, 2>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<4,  float, 2>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<8,  float, 2>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<16, float, 2>(at::Tensor mixture, const Config& configForwardOutput);

extern template ForwardOutput forward_impl_t<2,  double, 2>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<4,  double, 2>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<8,  double, 2>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<16, double, 2>(at::Tensor mixture, const Config& configForwardOutput);

extern template ForwardOutput forward_impl_t<2,  float, 3>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<4,  float, 3>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<8,  float, 3>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<16, float, 3>(at::Tensor mixture, const Config& configForwardOutput);

extern template ForwardOutput forward_impl_t<2,  double, 3>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<4,  double, 3>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<8,  double, 3>(at::Tensor mixture, const Config& configForwardOutput);
extern template ForwardOutput forward_impl_t<16, double, 3>(at::Tensor mixture, const Config& configForwardOutput);

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
torch::Tensor backward_impl_t(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);

extern template torch::Tensor backward_impl_t<2,  float, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<4,  float, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<8,  float, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<16, float, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);

extern template torch::Tensor backward_impl_t<2,  double, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<4,  double, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<8,  double, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<16, double, 2>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);

extern template torch::Tensor backward_impl_t<2,  float, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<4,  float, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<8,  float, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<16, float, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);

extern template torch::Tensor backward_impl_t<2,  double, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<4,  double, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<8,  double, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);
extern template torch::Tensor backward_impl_t<16, double, 3>(torch::Tensor grad, const ForwardOutput& forward_out, const Config& config);

} // namespace bvh_mhem_fit

