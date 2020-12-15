#ifndef GPE_UTIL_GRAD_ALGORITHMS_H
#define GPE_UTIL_GRAD_ALGORITHMS_H

#include <cinttypes>

#include <cuda_runtime.h>

#include "util/algorithms.h"
#include "util/containers.h"
#include "util/grad/common.h"

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
void minus(const T1& a, const T2& b, T1* a_grad, T2* b_grad, const decltype (a - b)& incoming_grad) {
    *a_grad = incoming_grad;
    *b_grad = -incoming_grad;
}
template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
void times(const T1& a, const T2& b, T1* a_grad, T2* b_grad, const decltype (a * b)& incoming_grad) {
    *a_grad = b * incoming_grad;
    *b_grad = a * incoming_grad;
}
template<typename scalar_t, int N_DIMS1, int N_DIMS2>
__host__ __device__ GPE_ALGORITHMS_INLINE
void times(const scalar_t& a, const glm::mat<N_DIMS1, N_DIMS2, scalar_t>& b, scalar_t* a_grad, glm::mat<N_DIMS1, N_DIMS2, scalar_t>* b_grad, const glm::mat<N_DIMS1, N_DIMS2, scalar_t>& incoming_grad) {
    *a_grad = gpe::sum(gpe::cwise_mul(b, incoming_grad));
    *b_grad = a * incoming_grad;
}
template<int N_DIMS1, int N_DIMS2, typename scalar_t>
__host__ __device__ GPE_ALGORITHMS_INLINE
void times(const glm::mat<N_DIMS1, N_DIMS2, scalar_t>& a, const scalar_t& b, glm::mat<N_DIMS1, N_DIMS2, scalar_t>* a_grad, scalar_t* b_grad, const glm::mat<N_DIMS1, N_DIMS2, scalar_t>& incoming_grad) {
    *a_grad = b * incoming_grad;
    *b_grad = gpe::sum(gpe::cwise_mul(a, incoming_grad));
}
template<typename scalar_t, int N_DIMS>
__host__ __device__ GPE_ALGORITHMS_INLINE
void times(const scalar_t& a, const glm::vec<N_DIMS, scalar_t>& b, scalar_t* a_grad, glm::vec<N_DIMS, scalar_t>* b_grad, const glm::vec<N_DIMS, scalar_t>& incoming_grad) {
    *a_grad = gpe::sum(b * incoming_grad);
    *b_grad = a * incoming_grad;
}
template<int N_DIMS, typename scalar_t>
__host__ __device__ GPE_ALGORITHMS_INLINE
void times(const glm::vec<N_DIMS, scalar_t>& a, const scalar_t& b, glm::vec<N_DIMS, scalar_t>* a_grad, scalar_t* b_grad, const glm::vec<N_DIMS, scalar_t>& incoming_grad, int) {
    *a_grad = b * incoming_grad;
    *b_grad = gpe::sum(a * incoming_grad);
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

template<typename T1, typename T2, typename T3, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
TwoGrads<gpe::Array<T1, N1>, gpe::Array<T2, N2>> outer_product(const gpe::Array<T1, N1>& a,
                    const gpe::Array<T2, N2>& b,
                    const gpe::Array2d<T3, N1, N2>& incoming_grad,
                    Function fun) {
    TwoGrads<gpe::Array<T1, N1>, gpe::Array<T2, N2>> r{};
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            T1 ga;
            T2 gb;
            fun(a[i], b[j], &ga, &gb, incoming_grad[i][j]);
            r.m_left[i] += ga;
            r.m_right[j] += gb;
        }
    }
    return r;
}

template<typename T1, typename T2, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
OneGrad<gpe::Array<T1, N>> transform(const gpe::Array<T1, N>& vec, const gpe::Array<T2, N>& incoming_grad, Function fun) {
    OneGrad<gpe::Array<T1, N>> r;
    for (unsigned i = 0; i < N; ++i) {
        r.m_grad[i] = fun(vec[i], incoming_grad[i]);
    }
    return r;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
OneGrad<gpe::Array2d<T1, N1, N2>> transform(const gpe::Array2d<T1, N1, N2>& mat, const gpe::Array2d<T2, N1, N2>& incoming_grad, Function fun) {
    OneGrad<gpe::Array2d<T1, N1, N2>> r;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            r.m_grad[i][j] = fun(mat[i][j], incoming_grad[i][j]);
        }
    }
    return r;
}

template<typename T1, typename T2, typename T3, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
TwoGrads<gpe::Array<T1, N>, T2> cwise_fun(
                const gpe::Array<T1, N>& a,
                const T2& b,
                const gpe::Array<T3, N>& incoming_grad,
                Function fun) {
    TwoGrads<gpe::Array<T1, N>, T2> grads;
    grads.m_right = {};
    for (unsigned i = 0; i < N; ++i) {
        T2 gr;
        fun(a[i], b, &grads.m_left[i], &gr, incoming_grad[i]);
        grads.m_right += gr;
    }
    return grads;
}

template<typename T1, typename T2, typename T3, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
TwoGrads<gpe::Array<T1, N>, gpe::Array<T2, N>> cwise_fun(
                const gpe::Array<T1, N>& m1,
                const gpe::Array<T2, N>& m2,
                const gpe::Array<T3, N>& incoming_grad,
                Function fun) {
    TwoGrads<gpe::Array<T1, N>, gpe::Array<T2, N>> grads;
    for (unsigned i = 0; i < N; ++i) {
        fun(m1[i], m2[i], &grads.m_left[i], &grads.m_right[i], incoming_grad[i]);
    }
    return grads;
}

template<typename T1, typename T2, typename T3, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
TwoGrads<Array2d<T1, N1, N2>, Array2d<T2, N1, N2>> cwise_fun(const Array2d<T1, N1, N2>& m1,
               const Array2d<T2, N1, N2>& m2,
               const Array2d<T3, N1, N2>& incoming_grad,
               Function fun) {
    TwoGrads<Array2d<T1, N1, N2>, Array2d<T2, N1, N2>> grads;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            fun(m1[i][j], m2[i][j], &grads.m_left[i][j], &grads.m_right[i][j], incoming_grad[i][j]);
        }
    }
    return grads;
}

/// multiplies every row in m with the corresponding element in v (column vector)
template<typename T1, typename T2, typename T3, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
TwoGrads<Array2d<T1, N1, N2>, Array<T2, N1>> cwise_fun(
               const Array2d<T1, N1, N2>& m,
               const Array<T2, N1>& v,
               const Array2d<T3, N1, N2>& incoming_grad,
               Function fun) {
    TwoGrads<Array2d<T1, N1, N2>, Array<T2, N1>> grads;
    for (unsigned i = 0; i < N1; ++i) {
        grads.m_right[i] = {};
        for (unsigned j = 0; j < N2; ++j) {
            const T1& a = m[i][j];
            const T2& b = v[i];
            T2 b_grad;
            fun(a, b, &grads.m_left[i][j], &b_grad, incoming_grad[i][j]);
            grads.m_right[i] += b_grad;
        }
    }
    return grads;
}

/// multiplies every column in m with the corresponding element in v (row vector)
template<typename T1, typename T2, typename T3, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
TwoGrads<Array<T1, N2>, Array2d<T2, N1, N2>> cwise_fun(
               const Array<T1, N2>& v,
               const Array2d<T2, N1, N2>& m,
               const Array2d<T3, N1, N2>& incoming_grad,
               Function fun) {
    TwoGrads<Array<T1, N2>, Array2d<T2, N1, N2>> grads;
    for (unsigned j = 0; j < N2; ++j) {
        grads.m_left[j] = {};
    }
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            const T1& a = v[j];
            const T2& b = m[i][j];
            T2 a_grad;
            fun(a, b, &a_grad, &grads.m_right[i][j], incoming_grad[i][j]);
            grads.m_left[j] += a_grad;
        }
    }
    return grads;
}

template<typename T1, uint32_t N1>
OneGrad<Array<T1, N1>> sum(const Array<T1, N1>&, T1 grad) {
    OneGrad<Array<T1, N1>> r;
    for (unsigned i = 0; i < N1; ++i) {
        r.m_grad[i] = grad;
    }
    return r;
}

template<typename T1, uint32_t N1, uint32_t N2>
OneGrad<Array2d<T1, N1, N2>> sum(const Array2d<T1, N1, N2>&, T1 grad) {
    OneGrad<Array2d<T1, N1, N2>> r;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            r.m_grad[i][j] = grad;
        }
    }
    return r;
}

template<typename T1, uint32_t N1, uint32_t N2>
OneGrad<Array2d<T1, N1, N2>> sum_rows(const Array2d<T1, N1, N2>&, const Array<T1, N1>& grad) {
    OneGrad<Array2d<T1, N1, N2>> r;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            r.m_grad[i][j] = grad[i];
        }
    }
    return r;
}

template<typename T1, uint32_t N1, uint32_t N2>
OneGrad<Array2d<T1, N1, N2>> sum_cols(const Array2d<T1, N1, N2>&, const Array<T1, N2>& grad) {
    OneGrad<Array2d<T1, N1, N2>> r;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            r.m_grad[i][j] = grad[j];
        }
    }
    return r;
}


} // namespace grad

} // namespace gpe

#undef GPE_ALGORITHMS_INLINE

#endif // GPE_UTIL_GRAD_ALGORITHMS_H
