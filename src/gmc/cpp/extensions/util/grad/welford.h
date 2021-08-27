#ifndef UTIL_GRAD_WELFORD_H
#define UTIL_GRAD_WELFORD_H


#include "util/cuda.h"
#include "util/glm.h"
#include "util/grad/algorithms.h"
#include "util/glm.h"

namespace gpe {
namespace grad {

template<int N_DIMS, typename scalar_t>
struct WeightedMeanAndCov {
    using vec_t = glm::vec<N_DIMS, scalar_t>;
    using mat_t = glm::mat<N_DIMS, N_DIMS, scalar_t>;
    scalar_t w_sum = 0;
    vec_t v_mean = vec_t{};
    mat_t C = mat_t{};

    EXECUTION_DEVICES
    WeightedMeanAndCov() = default;

    EXECUTION_DEVICES
    void addValue(scalar_t w, const vec_t& v) {
        w_sum += w;
        const auto v_mean_old = v_mean;
        const auto delta1 = (v - v_mean);
        v_mean += (w / w_sum) * delta1;
        const auto delta2 = (v - v_mean);

        C += w * glm::outerProduct(delta1, delta2);
    }

    EXECUTION_DEVICES
    vec_t mean() const {
        return v_mean;
    }

    EXECUTION_DEVICES
    mat_t cov_matrix() const {
        return C / w_sum;
    }
};


template<typename scalar_t, typename T>
struct WeightedMean {
    scalar_t w_sum = 0;
    T v_mean = T{};
    T grad_sum_over_wi_vi = T{};
    scalar_t grad_w_sum = 0;

    EXECUTION_DEVICES
    WeightedMean(scalar_t w_sum, T v_mean, scalar_t incoming_w_sum_grad, T incoming_v_mean_grad)
        : w_sum(w_sum), v_mean(v_mean) {
        // w_sum = sum_over(w_i)
        // mean = sum_over(w_i * v_i)/w_sum

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

