#ifndef UTIL_AUTODIFF_H
#define UTIL_AUTODIFF_H

#include <cuda_runtime.h>

#ifndef __CUDACC__
#include <type_traits>

#include <autodiff/reverse.hpp>
#endif

#include "math/gpe_glm.h"

namespace gpe {
template <typename scalar>
struct remove_grad {
    using type = scalar;
};


#ifndef __CUDACC__
template<>
struct remove_grad<autodiff::Variable<float>>{
    using type = float;
};

template<>
struct remove_grad<autodiff::Variable<double>>{
    using type = double;
};
#endif // __CUDACC__

template <typename scalar>
using remove_grad_t = typename remove_grad<scalar>::type;


#ifndef __CUDACC__
template<typename T>
T removeGrad(const T& v) {
    static_assert (std::is_floating_point<T>::value, "should be a basic float or double type");
    return v;
}

template <typename scalar_t>
scalar_t removeGrad(const autodiff::Variable<scalar_t>& v) {
    return autodiff::val(v);
}

template <typename scalar_t>
scalar_t removeGrad(const autodiff::reverse::ExprPtr<scalar_t>& v) {
    return autodiff::val(v);
}

template <int N_DIMS, typename scalar_t>
glm::vec<N_DIMS, scalar_t> removeGrad(const glm::vec<N_DIMS, autodiff::Variable<scalar_t>>& v) {
    glm::vec<N_DIMS, scalar_t> r;
    for (int i = 0; i < N_DIMS; ++i) {
        r[i] = removeGrad(v[i]);
    }
    return r;
}
template <int N_DIMS, typename scalar_t>
glm::vec<N_DIMS, scalar_t> removeGrad(const glm::vec<N_DIMS, autodiff::reverse::ExprPtr<scalar_t>>& v) {
    glm::vec<N_DIMS, scalar_t> r;
    for (int i = 0; i < N_DIMS; ++i) {
        r[i] = removeGrad(v[i]);
    }
    return r;
}
template <int N_DIMS, typename scalar_t>
glm::vec<N_DIMS, scalar_t> removeGrad(const glm::vec<N_DIMS, scalar_t>& v) {
    return v;
}

template <int N_DIMS, typename scalar_t>
glm::mat<N_DIMS, N_DIMS, scalar_t> removeGrad(const glm::mat<N_DIMS, N_DIMS, autodiff::Variable<scalar_t>>& v) {
    glm::mat<N_DIMS, N_DIMS, scalar_t> r;
    for (int i = 0; i < N_DIMS; ++i) {
        r[i] = removeGrad(v[i]);
    }
    return r;
}
template <int N_DIMS, typename scalar_t>
glm::mat<N_DIMS, N_DIMS, scalar_t> removeGrad(const glm::mat<N_DIMS, N_DIMS, autodiff::reverse::ExprPtr<scalar_t>>& v) {
    glm::mat<N_DIMS, N_DIMS, scalar_t> r;
    for (int i = 0; i < N_DIMS; ++i) {
        r[i] = removeGrad(v[i]);
    }
    return r;
}
template <int N_DIMS, typename scalar_t>
glm::mat<N_DIMS, N_DIMS, scalar_t> removeGrad(const glm::mat<N_DIMS, N_DIMS, scalar_t>& v) {
    return v;
}

template<typename scalar_t>
autodiff::Variable<scalar_t> makeAutodiff(scalar_t v) {
    return autodiff::Variable<scalar_t>(v);
}

template<int N_DIMS, typename scalar_t>
glm::vec<N_DIMS, autodiff::Variable<scalar_t>> makeAutodiff(const glm::vec<N_DIMS, scalar_t>& v) {
    glm::vec<N_DIMS, autodiff::Variable<scalar_t>> r;
    for (int i = 0; i < N_DIMS; ++i) {
        r[i] = makeAutodiff(v[i]);
    }
    return r;
}

template<int N_DIMS, typename scalar_t>
glm::mat<N_DIMS, N_DIMS, autodiff::Variable<scalar_t>> makeAutodiff(const glm::mat<N_DIMS, N_DIMS, scalar_t>& v) {
    glm::mat<N_DIMS, N_DIMS, autodiff::Variable<scalar_t>> r;
    for (int i = 0; i < N_DIMS; ++i) {
        r[i] = makeAutodiff(v[i]);
    }
    return r;
}

template<typename scalar_t>
scalar_t extractGrad(const autodiff::Variable<scalar_t>& v) {
    return v.grad();
}

template<int N_DIMS, typename scalar_t>
glm::vec<N_DIMS, scalar_t> extractGrad(const glm::vec<N_DIMS, autodiff::Variable<scalar_t>>& v) {
    glm::vec<N_DIMS, scalar_t> r;
    for (int i = 0; i < N_DIMS; ++i) {
        r[i] = extractGrad(v[i]);
    }
    return r;
}

template<int N_DIMS, typename scalar_t>
glm::mat<N_DIMS, N_DIMS, scalar_t> extractGrad(const glm::mat<N_DIMS, N_DIMS, autodiff::Variable<scalar_t>>& v) {
    glm::mat<N_DIMS, N_DIMS, scalar_t> r;
    for (int i = 0; i < N_DIMS; ++i) {
        r[i] = extractGrad(v[i]);
    }
    return r;
}


#else // __CUDACC__

template <typename T>
__host__ __device__ __forceinline__
T removeGrad(const T& v) {
    return v;
}

#endif // __CUDACC__

}

#endif // UTIL_AUTODIFF_H
