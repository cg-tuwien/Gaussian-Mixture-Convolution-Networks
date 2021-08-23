#ifndef UTIL_WELFORD_H
#define UTIL_WELFORD_H


#include "cuda_qt_creator_definitinos.h"
#include "util/glm.h"

namespace gpe {

template<int N_DIMS, typename scalar_t>
struct WelfordWeightedIncremental {
    using vec_t = glm::vec<N_DIMS, scalar_t>;
    using mat_t = glm::mat<N_DIMS, N_DIMS, scalar_t>;
    scalar_t w_sum = 0;
    vec_t v_mean = vec_t{};
    mat_t C = mat_t{};
    scalar_t c_xx = 0;
    scalar_t c_xy = 0;
    scalar_t c_yy = 0;

    void addValue(scalar_t w, const vec_t& v) {
        w_sum += w;
        const auto v_mean_old = v_mean;
        const auto delta1 = (v - v_mean);
        v_mean += (w / w_sum) * delta1;
        const auto delta2 = (v - v_mean);

        c_xx += w * delta1.x * delta2.x;
        c_xy += w * delta1.x * delta2.y;
        c_yy += w * delta1.y * delta2.y;

        C += w * glm::outerProduct(delta1, delta2);
    }

    vec_t mean() const {
        return v_mean;
    }

    mat_t cov_matrix() const {
        return C / w_sum;
    }

    // todo: 1. test 2. build into preiners formulas 3. test preiners formulas.
};

}

#endif // UTIL_WELFORD_H
