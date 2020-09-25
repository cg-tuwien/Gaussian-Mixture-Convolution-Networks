#include <tuple>
#include <torch/script.h>

#ifndef BVH_MHEM_FIT_IMPLEMENTATION
#define BVH_MHEM_FIT_IMPLEMENTATION

namespace bvh_mhem_fit {

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cuda_bvh_forward_impl(const at::Tensor& mixture, const at::Tensor& xes);

std::tuple<torch::Tensor, torch::Tensor> cuda_bvh_backward_impl(const torch::Tensor& grad_output,
                                                  const torch::Tensor& mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& aabbs,
                                                  const torch::Tensor& xes,
                                                  bool requires_grad_mixture, bool requires_grad_xes);

}
#endif
