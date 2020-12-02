#ifndef IMPLEMENTATION_AUTODIFF_BACKWARD_H
#define IMPLEMENTATION_AUTODIFF_BACKWARD_H

#include <torch/types.h>

#include "bvh_mhem_fit/BvhMhemFitConfig.h"
#include "bvh_mhem_fit/implementation.h"

namespace bvh_mhem_fit {

struct ForwardBackWardOutput {
torch::Tensor output;
torch::Tensor mixture_gradient;
};

template<int REDUCTION_N = 4, typename scalar_t, unsigned N_DIMS>
ForwardBackWardOutput implementation_autodiff_backward(at::Tensor mixture, const BvhMhemFitConfig& config);

extern template ForwardBackWardOutput implementation_autodiff_backward<2, float, 2>(at::Tensor mixture, const BvhMhemFitConfig& config);
}

#endif // IMPLEMENTATION_AUTODIFF_BACKWARD_H
