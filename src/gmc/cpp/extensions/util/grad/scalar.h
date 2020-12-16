#ifndef GPE_UTIL_GRAD_SCALAR_H
#define GPE_UTIL_GRAD_SCALAR_H

#include "util/cuda.h"
#include "util/scalar.h"
#include "util/epsilon.h"

namespace gpe {
namespace grad {

template <typename scalar_t>
EXECUTION_DEVICES
void pow(scalar_t x, scalar_t y, scalar_t* grad_x, scalar_t* grad_y, scalar_t incoming_grad) {
    x = Epsilon<scalar_t>::clip(x);
    *grad_x = incoming_grad * y * gpe::pow(x, y-1);
    *grad_y = incoming_grad * gpe::pow(x, y) * gpe::log(x);
}

template <typename scalar_t>
EXECUTION_DEVICES
scalar_t exp(scalar_t x, scalar_t incoming_grad) {
    return gpe::exp(x) * incoming_grad;
}

template <typename scalar_t>
EXECUTION_DEVICES
scalar_t log(scalar_t x, scalar_t incoming_grad) {
    assert(x > Epsilon<scalar_t>::large);       // be loud about -inf gradients?
//    return incoming_grad / Epsilon<scalar_t>::clip(x);
    return incoming_grad / x;
}

} // namespace grad

} // namespace gpe

#endif // GPE_UTIL_GRAD_SCALAR_H
