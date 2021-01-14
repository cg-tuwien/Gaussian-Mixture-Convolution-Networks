#ifndef GPE_UTIL_GAUSSIAN_MIXTURE_H
#define GPE_UTIL_GAUSSIAN_MIXTURE_H

#include "util/containers.h"
#include "util/gaussian.h"

namespace gpe {

template<int N_DIMS, typename scalar_t, unsigned N>
EXECUTION_DEVICES
gpe::Array<gpe::Gaussian<N_DIMS, scalar_t>, N> pack_mixture(const gpe::Array<scalar_t, N>& weights,
                                                            const gpe::Array<glm::vec<N_DIMS, scalar_t>, N>& positions,
                                                            const gpe::Array<glm::mat<N_DIMS, N_DIMS, scalar_t>, N>& covariances) {
    gpe::Array<gpe::Gaussian<N_DIMS, scalar_t>, N> r;
    for (unsigned i = 0; i < N; ++i) {
        r[i].weight = weights[i];
        r[i].position = positions[i];
        r[i].covariance = covariances[i];
    }
    return r;
}

template <typename TensorAccessor>
EXECUTION_DEVICES auto weight(TensorAccessor&& gaussian) -> decltype (gaussian[0]) {
    return gaussian[0];
}

template <int DIMS, typename TensorAccessor>
EXECUTION_DEVICES auto position(TensorAccessor&& gaussian) -> decltype (gpe::vec<DIMS>(gaussian[1])) {
    return gpe::vec<DIMS>(gaussian[1]);
}

template <int DIMS, typename TensorAccessor>
EXECUTION_DEVICES auto covariance(TensorAccessor&& gaussian) -> decltype (gpe::mat<DIMS>(gaussian[1 + DIMS])) {
    return gpe::mat<DIMS>(gaussian[1 + DIMS]);
}
template <int DIMS, typename TensorAccessor>
EXECUTION_DEVICES auto gaussian(TensorAccessor&& gaussian) -> Gaussian<DIMS, gpe::remove_cvref_t<decltype (gaussian[0])>>& {
    return reinterpret_cast<Gaussian<DIMS, gpe::remove_cvref_t<decltype (gaussian[0])>>&>(gaussian[0]);
}
template <int DIMS, typename TensorAccessor>
EXECUTION_DEVICES auto gaussian(const TensorAccessor&& gaussian) -> const Gaussian<DIMS, gpe::remove_cvref_t<decltype (gaussian[0])>>& {
    return reinterpret_cast<const Gaussian<DIMS, gpe::remove_cvref_t<decltype (gaussian[0])>>&>(gaussian[0]);
}

}

#endif // GPE_UTIL_GAUSSIAN_MIXTURE_H
