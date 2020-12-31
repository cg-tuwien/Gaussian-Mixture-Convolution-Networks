#ifndef GPE_BVH_MHEM_FIT_ALPHA_IMPLEMENTATION_AUTODIFF_BACKWARD_H
#define GPE_BVH_MHEM_FIT_ALPHA_IMPLEMENTATION_AUTODIFF_BACKWARD_H

#include <torch/types.h>

#include "bvh_mhem_fit_alpha/Config.h"
#include "bvh_mhem_fit_alpha/implementation.h"

namespace bvh_mhem_fit_alpha {

struct ForwardBackWardOutput {
torch::Tensor output;
torch::Tensor mixture_gradient;
};

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
ForwardBackWardOutput implementation_autodiff_backward(at::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);

extern template ForwardBackWardOutput implementation_autodiff_backward<2, float, 2>(at::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);
extern template ForwardBackWardOutput implementation_autodiff_backward<2, double, 2>(at::Tensor mixture, const torch::Tensor& gradient_fitting, const Config& config);
}

#endif // GPE_BVH_MHEM_FIT_ALPHA_IMPLEMENTATION_AUTODIFF_BACKWARD_H
