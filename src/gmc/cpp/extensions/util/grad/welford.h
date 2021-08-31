#ifndef UTIL_GRAD_WELFORD_H
#define UTIL_GRAD_WELFORD_H


#include "util/cuda.h"
#include "util/glm.h"
#include "util/grad/algorithms.h"
#include "util/grad/glm.h"
#include "util/glm.h"

namespace gpe {
namespace grad {

template<int N_DIMS, typename scalar_t>
struct WeightedMeanAndCov {
    using vec_t = glm::vec<N_DIMS, scalar_t>;
    using mat_t = glm::mat<N_DIMS, N_DIMS, scalar_t>;

    vec_t grad_sum_over_wi_vi = vec_t{};
    mat_t grad_sum_over_wi_vpi_vpiT = mat_t{};
    scalar_t grad_w_sum = 0;

    EXECUTION_DEVICES
    WeightedMeanAndCov(scalar_t w_sum, const vec_t& v_mean, const mat_t& cov, scalar_t incoming_w_sum_grad, const vec_t& incoming_v_mean_grad, const mat_t& incoming_cov_grad) {
        // w_sum = sum_over(w_i)
        // mean = sum_over(w_i * v_i)/w_sum
        // cov = sum_over(w_i * v_i * v_i^T)/w_sum - mean * mean^T
        // the cov formula is 'susceptible to catastrophic cancellation' (wikipedia on variance and covariance), but we use it only for calculating the gradient, which should be fine.
        // it's actually even very good, as the gradient for sums is numerically stable. + we can do it in one iteration over the weights.

        if (w_sum == scalar_t(0))
            return;

        // we have incoming grads for mean, cov, and w_sum.
        // back to front:
        // incoming cov grad onto sum and - mean * mean^T; this is the mean part
        vec_t v_mean_grad = {};
        gpe::grad::outerProduct(v_mean, v_mean, &v_mean_grad, &v_mean_grad, -incoming_cov_grad);

        // grad for cov sum is divided onto grad of sum_over(w_i * vp_i * vp_i^T) and grad of w_sum
        // sum_over(w_i * vp_i * vp_i^T) = w_sum * (cov + mean * mean^T)
        gpe::grad::functors::divided_AbyB(w_sum * (cov + glm::outerProduct(v_mean, v_mean)), w_sum, &grad_sum_over_wi_vpi_vpiT, &grad_w_sum, incoming_cov_grad);

        v_mean_grad += incoming_v_mean_grad;

        // grad for mean is divided onto grad of sum_over(w_i * v_i) and grad of w_sum
        scalar_t grad_w_sum_add = 0;
        gpe::grad::functors::divided_AbyB(w_sum * v_mean, w_sum, &grad_sum_over_wi_vi, &grad_w_sum_add, v_mean_grad); // sum_over(w_i * v_i) = w_sum * v_mean
        grad_w_sum += grad_w_sum_add;

        // now, grad_w_sum + incoming_w_sum_grad can be added on the fly to w_grad;
        grad_w_sum += incoming_w_sum_grad;
        // grad_sum_over_* go into the sums, i.e., they go into w_i and v_i
    }

    EXECUTION_DEVICES
    void addValue(scalar_t w, const vec_t& v, scalar_t* w_grad, vec_t* v_grad) {
        // (w != scalar_t(0)) because we are iffing away 0 weights, because they can produce NaNs
        *w_grad = gpe::sum(gpe::cwise_mul(grad_sum_over_wi_vi, v)) * (w != scalar_t(0))
                  + gpe::sum(gpe::cwise_mul(grad_sum_over_wi_vpi_vpiT, gpe::outerProduct(v, v))) * (w != scalar_t(0))
                  + grad_w_sum;
        vec_t v_grad_over_cov = {};
        gpe::grad::outerProduct(v, v, &v_grad_over_cov, &v_grad_over_cov, grad_sum_over_wi_vpi_vpiT * w);
        *v_grad = grad_sum_over_wi_vi * w + v_grad_over_cov;
    }
};


template<typename scalar_t, typename T>
struct WeightedMean {
    T grad_sum_over_wi_vi = T{};
    scalar_t grad_w_sum = 0;

    EXECUTION_DEVICES
    WeightedMean(scalar_t w_sum, const T& v_mean, scalar_t incoming_w_sum_grad, const T& incoming_v_mean_grad) {
        // w_sum = sum_over(w_i)
        // mean = sum_over(w_i * v_i)/w_sum

        if (w_sum == scalar_t(0))
            return;

        // we have incoming grads for mean and w_sum.
        // back to front:
        // incoming grad for mean is divided onto grad of sum_over(w_i * v_i) and grad of w_sum
        gpe::grad::functors::divided_AbyB(w_sum * v_mean, w_sum, &grad_sum_over_wi_vi, &grad_w_sum, incoming_v_mean_grad); // sum_over(w_i * v_i) = w_sum * v_mean
        // now, grad_w_sum + incoming_w_sum_grad can be added on the fly to w_grad;
        grad_w_sum += incoming_w_sum_grad;
        // grad_sum_over_wi_vi goes into the sum, i.e., it goes into w_i and v_i
    }

    EXECUTION_DEVICES
    void addValue(scalar_t w, const T& v, scalar_t* w_grad, T* v_grad) {
        // (w != scalar_t(0)) because we are iffing away 0 weights, because they can produce NaNs
        *w_grad = gpe::sum(gpe::cwise_mul(grad_sum_over_wi_vi, v)) * (w != scalar_t(0)) + grad_w_sum;
        *v_grad = grad_sum_over_wi_vi * w;
    }
};


} // namespace grad
} // namespace gpe

#endif // UTIL_GRAD_WELFORD_H

