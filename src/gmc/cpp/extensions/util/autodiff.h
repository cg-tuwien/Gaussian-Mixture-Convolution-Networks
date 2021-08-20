#ifndef GPE_UTIL_AUTODIFF_H
#define GPE_UTIL_AUTODIFF_H


namespace gpe {
template <typename scalar>
struct remove_grad {
    using type = scalar;
};

template <typename scalar>
using remove_grad_t = typename remove_grad<scalar>::type;
}


#ifdef GPE_AUTODIFF
#include <type_traits>

#include <autodiff/reverse/var.hpp>
#include <cuda_runtime.h>

#include "util/containers.h"
#include "util/glm.h"

namespace gpe {
template<>
struct remove_grad<autodiff::Variable<float>>{
    using type = float;
};

template<>
struct remove_grad<autodiff::Variable<double>>{
    using type = double;
};

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
scalar_t removeGrad(const autodiff::detail::ExprPtr<scalar_t>& v) {
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
glm::vec<N_DIMS, scalar_t> removeGrad(const glm::vec<N_DIMS, autodiff::detail::ExprPtr<scalar_t>>& v) {
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
glm::mat<N_DIMS, N_DIMS, scalar_t> removeGrad(const glm::mat<N_DIMS, N_DIMS, autodiff::detail::ExprPtr<scalar_t>>& v) {
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

template <uint32_t N, typename T, typename size_type>
auto removeGrad(const gpe::Vector<T, N, size_type>& vec) -> gpe::Vector<decltype (removeGrad(vec.front())), N, size_type> {
    using R = decltype (removeGrad(vec.front()));
    gpe::Vector<R, N, size_type> r;
    for (const auto& val : vec)
        r.push_back(removeGrad(val));
    return r;
}

template <uint32_t N, typename T>
auto removeGrad(const gpe::Array<T, N>& arr) -> gpe::Array<decltype (removeGrad(arr.front())), N> {
    using R = decltype (removeGrad(arr.front()));
    gpe::Array<R, N> r;
    unsigned i = 0;
    for (const auto& val : arr)
        r[i++] = removeGrad(val);
    return r;
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

template<uint32_t N, typename T>
gpe::Array<autodiff::Variable<T>, N> makeAutodiff(const gpe::Array<T, N>& v) {
    gpe::Array<autodiff::Variable<T>, N> r;
    for (unsigned i = 0; i < N; ++i) {
        r[i] = makeAutodiff(v[i]);
    }
    return r;
}

template<uint32_t N1, uint32_t N2, typename T>
gpe::Array2d<autodiff::Variable<T>, N1, N2> makeAutodiff(const gpe::Array2d<T, N1, N2>& m) {
    gpe::Array2d<autodiff::Variable<T>, N1, N2> r;
    for (unsigned i = 0; i < N1; ++i) {
        r[i] = makeAutodiff(m[i]);
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

template <uint32_t N, typename T>
auto extractGrad(const gpe::Array<T, N>& arr) -> gpe::Array<decltype (extractGrad(arr.front())), N> {
    using R = decltype (extractGrad(arr.front()));
    gpe::Array<R, N> r;
    unsigned i = 0;
    for (const auto& val : arr)
        r[i++] = extractGrad(val);
    return r;
}

template <typename scalar_t>
void propagateGrad(const autodiff::Variable<scalar_t>& v, scalar_t grad) {
    v.expr->propagate(grad);
}

template <typename scalar_t>
void propagateGrad(const autodiff::detail::ExprPtr<scalar_t>& v, scalar_t grad) {
    v->propagate(grad);
}

template<int N_DIMS, typename scalar_t>
void propagateGrad(const glm::vec<N_DIMS, autodiff::Variable<scalar_t>>& v, const glm::vec<N_DIMS, scalar_t>& grad) {
    for (int i = 0; i < N_DIMS; ++i) {
        propagateGrad(v[i], grad[i]);
    }
}

template<int N_DIMS, typename scalar_t>
void propagateGrad(const glm::mat<N_DIMS, N_DIMS, autodiff::Variable<scalar_t>>& v, const glm::mat<N_DIMS, N_DIMS, scalar_t>& grad) {
    for (int i = 0; i < N_DIMS; ++i) {
        propagateGrad(v[i], grad[i]);
    }
}

}
#endif // GPE_AUTODIFF


#endif // GPE_UTIL_AUTODIFF_H
