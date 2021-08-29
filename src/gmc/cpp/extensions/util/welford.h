#ifndef UTIL_WELFORD_H
#define UTIL_WELFORD_H


#include "util/cuda.h"
#include "util/epsilon.h"
#include "util/glm.h"

namespace gpe {

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
        if (w == scalar_t(0.0))
            return;
        const auto v_mean_old = v_mean;
        const auto delta1 = (v - v_mean);
        v_mean += scalar_t(w / w_sum) * delta1;
        const auto delta2 = (v - v_mean);

        C += w * glm::outerProduct(delta1, delta2);
    }

    EXECUTION_DEVICES
    vec_t mean() const {
        return v_mean;
    }

    EXECUTION_DEVICES
    mat_t cov_matrix() const {
        if (w_sum == scalar_t(0.0))
            return mat_t(1);
        return C / w_sum;
    }
};


template<typename scalar_t, typename T>
struct WeightedMean {
    scalar_t w_sum = 0;
    T v_mean = T{};

    EXECUTION_DEVICES
    WeightedMean() = default;

    EXECUTION_DEVICES
    void addValue(scalar_t w, const T& v) {
        w_sum += w;
        if (w == scalar_t(0.0))
            return;
        const auto delta1 = v - v_mean;
        v_mean += scalar_t(w / w_sum) * delta1;  // this scalar_t cast makes the compiler happy, when compiling unit tests with autodiff.
    }

    EXECUTION_DEVICES
    const T& mean() const {
        return v_mean;
    }
};

}

#endif // UTIL_WELFORD_H
