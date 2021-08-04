#ifndef HELPER_H
#define HELPER_H

#include "util/containers.h"
#include "cuda_operations.h"

namespace gpe {

template<typename large_type, unsigned n_dims, typename small_type = large_type>
__host__ __device__
gpe::Array<small_type, n_dims> split_n_dim_index(const gpe::Array<small_type, n_dims>& dimensions, large_type idx ) {
    gpe::Array<small_type, n_dims> tmp;
    tmp.front() = 1;
    for (unsigned i = 1; i < n_dims; ++i) {
        tmp[i] = dimensions[i - 1] * tmp[i - 1];
    }

    for (unsigned i = n_dims - 1; i < n_dims; --i) {
        const auto tmp_idx = idx / tmp[i];
        idx -= tmp_idx * tmp[i];
        tmp[i] = tmp_idx;
    }
    return tmp;
}

template<typename large_type, unsigned n_dims, typename small_type = large_type>
__host__ __device__
large_type join_n_dim_index(const gpe::Array<small_type, n_dims>& dimensions, const gpe::Array<small_type, n_dims>& idx ) {
    large_type joined_idx = 0;
    large_type cum_dims = 1;
    for (unsigned i = 0; i < n_dims; ++i) {
        joined_idx += idx[i] * cum_dims;
        cum_dims *= dimensions[i];
    }
    return joined_idx;
}

} // namespace gpe


#endif // HELPER_H
