#ifndef GPE_UTIL_EPSILON_H
#define GPE_UTIL_EPSILON_H

#include "util/scalar.h"
#include "util/autodiff.h"
#include "util/cuda.h"

namespace gpe {
template <typename scalar_t = float>
struct Epsilon {
    static constexpr scalar_t small = scalar_t(0.0000001);
    static constexpr scalar_t large = scalar_t(0.0001);
    static EXECUTION_DEVICES scalar_t clip(scalar_t v) { return gpe::max(v, small); }
    static EXECUTION_DEVICES scalar_t grad_clip(scalar_t v, scalar_t incoming_grad) { return (v > small) ? incoming_grad : 0; }
};

#ifdef GPE_AUTODIFF

template<>
struct Epsilon<autodiff::Variable<float>> {
    static constexpr float small = 0.00000000000000000000000000001;
    static constexpr float large = 0.0000000000001;
    static EXECUTION_DEVICES autodiff::Variable<float> clip(autodiff::Variable<float> v) {
        return gpe::max(v, autodiff::Variable<float>(small));
    }
};
template<>
struct Epsilon<autodiff::Variable<double>> {
    static constexpr double small = 0.00000000000000000000000000000000001;
    static constexpr double large = 0.0000000000000001;
    static EXECUTION_DEVICES autodiff::Variable<double> clip(autodiff::Variable<double> v) {
        return gpe::max(v, autodiff::Variable<double>(small));
    }
};

#endif

}

#endif // GPE_UTIL_EPSILON_H
