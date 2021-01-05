#ifndef GPE_UTIL_GRAD_MIXTURE_H
#define GPE_UTIL_GRAD_MIXTURE_H

#include "util/gaussian.h"
#include "util/containers.h"
#include "util/cuda.h"

namespace gpe {

namespace grad {

template <typename scalar_t, int N_DIMS, unsigned N_GAUSSIANS> EXECUTION_DEVICES
void unpackAndAdd(const gpe::Array<gpe::Gaussian<N_DIMS, scalar_t>, N_GAUSSIANS>& grad,
                  gpe::Array<scalar_t, N_GAUSSIANS>* grad_weights,
                  gpe::Array<glm::vec<N_DIMS, scalar_t>, N_GAUSSIANS>* grad_positions,
                  gpe::Array<glm::mat<N_DIMS, N_DIMS, scalar_t>, N_GAUSSIANS>* grad_covariances) {
    for (unsigned i = 0; i < N_GAUSSIANS; ++i) {
        (*grad_weights)[i] += grad[i].weight;
        (*grad_positions)[i] += grad[i].position;
        (*grad_covariances)[i] += grad[i].covariance;
    }
}

} // namespace grad

} // namespace gpe

#endif // GPE_UTIL_GRAD_MIXTURE_H
