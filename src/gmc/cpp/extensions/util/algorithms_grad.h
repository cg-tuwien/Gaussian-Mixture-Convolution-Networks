#ifndef GPE_UTIL_ALGORITHMS_GRAD_H
#define GPE_UTIL_ALGORITHMS_GRAD_H

#include <cinttypes>

#include <cuda_runtime.h>

#include "containers.h"

#ifdef NDEBUG
#define GPE_ALGORITHMS_INLINE __forceinline__
#else
#define GPE_ALGORITHMS_INLINE
#endif

namespace gpe {
namespace grad {

namespace functors {
template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
void plus(const T1& a, const T2& b, T1* a_grad, T2* b_grad, const decltype (a + b)& incoming_grad) {
    *a_grad = incoming_grad;
    *b_grad = incoming_grad;
}
template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
void times(const T1& a, const T2& b, T1* a_grad, T2* b_grad, const decltype (a * b)& incoming_grad) {
    *a_grad = b * incoming_grad;
    *b_grad = a * incoming_grad;
}

template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
void divided_AbyB(const T1& a, const T2& b, T1* a_grad, T2* b_grad, const decltype (a / b)& incoming_grad) {
    *a_grad = incoming_grad / b;
    *b_grad = -incoming_grad * a / (b * b);
}

template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
void divided_BbyA(const T1& a, const T2& b, T1* a_grad, T2* b_grad, const decltype (b / a)& incoming_grad) {
    *a_grad = -incoming_grad * b / (a * a);
    *b_grad = incoming_grad / a;
}

}

// ////////////////////////////////////////  array algorithms //////////////////////////////////////////

//template<typename T, uint32_t N, typename Function>
//__host__ __device__ GPE_ALGORITHMS_INLINE
//auto transform(const gpe::Array<T, N>& vec, Function fun) -> Array<decltype (fun(vec.front())), N> {
//    using ProductType = decltype (fun(vec.front()));
//    gpe::Array<ProductType, N> retvec;
//    for (unsigned i = 0; i < N; ++i) {
//        retvec[i] = fun(vec[i]);
//    }
//    return retvec;
//}

//template<typename T, uint32_t N1, uint32_t N2, typename Function>
//__host__ __device__ GPE_ALGORITHMS_INLINE
//auto transform(const gpe::Array2d<T, N1, N2>& mat, Function fun) -> Array2d<decltype (fun(mat.front().front())), N1, N2> {
//    using ProductType = decltype (fun(mat.front().front()));
//    gpe::Array2d<ProductType, N1, N2> retmat;
//    for (unsigned i = 0; i < N1; ++i) {
//        for (unsigned j = 0; j < N2; ++j) {
//            retmat[i][j] = fun(mat[i][j]);
//        }
//    }
//    return retmat;
//}

template <typename A1, typename A2>
struct Grads {
    A1 left_grad;
    A2 right_grad;
};

template<typename T1, typename T2, typename T3, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
Grads<gpe::Array<T1, N>, gpe::Array<T2, N>> cwise_fun(const gpe::Array<T1, N>& m1,
               const gpe::Array<T2, N>& m2,
               const gpe::Array<T3, N>& incoming_grad,
               Function fun) {
    Grads<gpe::Array<T1, N>, gpe::Array<T2, N>> grads;
    for (unsigned i = 0; i < N; ++i) {
        fun(m1[i], m2[i], &grads.left_grad[i], &grads.right_grad[i], incoming_grad[i]);
    }
    return grads;
}

template<typename T1, typename T2, typename T3, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
Grads<Array2d<T1, N1, N2>, Array2d<T2, N1, N2>> cwise_fun(const Array2d<T1, N1, N2>& m1,
               const Array2d<T2, N1, N2>& m2,
               const Array2d<T3, N1, N2>& incoming_grad,
               Function fun) {
    Grads<Array2d<T1, N1, N2>, Array2d<T2, N1, N2>> grads;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            fun(m1[i][j], m2[i][j], &grads.left_grad[i][j], &grads.right_grad[i][j], incoming_grad[i][j]);
        }
    }
    return grads;
}

/// multiplies every row in m with the corresponding element in v (column vector)
template<typename T1, typename T2, typename T3, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
Grads<Array2d<T1, N1, N2>, Array<T2, N1>> cwise_fun(
               const Array2d<T1, N1, N2>& m,
               const Array<T2, N1>& v,
               const Array2d<T3, N1, N2>& incoming_grad,
               Function fun) {
    Grads<Array2d<T1, N1, N2>, Array<T2, N1>> grads;
    for (unsigned i = 0; i < N1; ++i) {
        grads.right_grad[i] = {};
        for (unsigned j = 0; j < N2; ++j) {
            const T1& a = m[i][j];
            const T2& b = v[i];
            T2 b_grad;
            fun(a, b, &grads.left_grad[i][j], &b_grad, incoming_grad[i][j]);
            grads.right_grad[i] += b_grad;
        }
    }
    return grads;
}

/// multiplies every column in m with the corresponding element in v (row vector)
template<typename T1, typename T2, typename T3, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
Grads<Array<T1, N2>, Array2d<T2, N1, N2>> cwise_fun(
               const Array<T1, N2>& v,
               const Array2d<T2, N1, N2>& m,
               const Array2d<T3, N1, N2>& incoming_grad,
               Function fun) {
    Grads<Array<T1, N2>, Array2d<T2, N1, N2>> grads;
    for (unsigned j = 0; j < N2; ++j) {
        grads.left_grad[j] = {};
    }
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            const T1& a = v[j];
            const T2& b = m[i][j];
            T2 a_grad;
            fun(a, b, &a_grad, &grads.right_grad[i][j], incoming_grad[i][j]);
            grads.left_grad[j] += a_grad;
        }
    }
    return grads;
}


} // namespace grad

} // namespace gpe

#undef GPE_ALGORITHMS_INLINE

#endif // GPE_UTIL_ALGORITHMS_GRAD_H
