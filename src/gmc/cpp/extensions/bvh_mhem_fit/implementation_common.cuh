#ifndef BVH_MHEM_FIT_IMPLEMENTATION_COMMON
#define BVH_MHEM_FIT_IMPLEMENTATION_COMMON

#include <torch/types.h>

namespace bvh_mhem_fit {

namespace  {
torch::Tensor inverse_permutation(const torch::Tensor& p) {
    auto l = torch::arange(p.size(-1), torch::TensorOptions(p.device()).dtype(p.dtype()));
    auto shape = p.sizes().vec();
    assert(shape.size() > 0);
    std::for_each(shape.begin(), shape.end() - 1, [](auto& i) { i = 1; });
    l = l.view(shape).expand_as(p);
    return torch::scatter(torch::empty_like(p), -1, p, l);
}

} // anonymous namespace

} // namespace bvh_mhem_fit

#endif
