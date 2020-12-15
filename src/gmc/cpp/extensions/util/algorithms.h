#ifndef GPE_UTIL_ALGORITHMS_H
#define GPE_UTIL_ALGORITHMS_H

#include <cinttypes>

#include <cuda_runtime.h>

#include "containers.h"

#ifdef NDEBUG
#define GPE_ALGORITHMS_INLINE __forceinline__
#else
#define GPE_ALGORITHMS_INLINE
#endif

namespace gpe {
namespace functors {
template<typename T1, typename T2 = T1, typename T3 = decltype (T1{} + T2{})>
__host__ __device__ GPE_ALGORITHMS_INLINE
T3 plus(const T1& a, const T2& b) {
    return a + b;
}
template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto minus(const T1& a, const T2& b) -> decltype (a - b) {
    return a - b;
}
template<typename T1, typename T2 = T1, typename T3 = decltype (T1{} * T2{})>
__host__ __device__ GPE_ALGORITHMS_INLINE
T3 times(const T1& a, const T2& b)  {
    return a * b;
}
template<typename T1, typename T2 = T1, typename T3 = decltype (T1{} / T2{})>
__host__ __device__ GPE_ALGORITHMS_INLINE
T3 divided_AbyB(const T1& a, const T2& b) {
    return a / b;
}
template<typename T1, typename T2 = T1, typename T3 = decltype (T2{} / T1{})>
__host__ __device__ GPE_ALGORITHMS_INLINE
T3 divided_BbyA(const T1& a, const T2& b) {
    return b / a;
}
template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto logical_and(const T1& a, const T2& b) -> decltype (a && b) {
    return a && b;
}
template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto logical_or(const T1& a, const T2& b) -> decltype (a || b) {
    return a || b;
}
template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto bit_or(const T1& a, const T2& b) -> decltype (a | b) {
    return a | b;
}
template<typename T1, typename T2 = T1>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto bit_and(const T1& a, const T2& b) -> decltype (a & b) {
    return a & b;
}

}

// ////////////////////////////////////////  array algorithms //////////////////////////////////////////


template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto outer_product(const gpe::Array<T1, N1>& m1,
                   const gpe::Array<T2, N2>& m2,
                   Function fun) -> Array2d<decltype (fun(m1.front(), m2.front())), N1, N2> {
    using ProductType = decltype (fun(m1.front(), m2.front()));
    gpe::Array2d<ProductType, N1, N2> matrix;
    for (unsigned i = 0; i < N1; ++i) {
        const T1& c_i = m1[i];
        for (unsigned j = 0; j < N2; ++j) {
            const T2& c_j = m2[j];
            ProductType v = fun(c_i, c_j);
            matrix[i][j] = v;
        }
    }
    return matrix;
}

template<typename T, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto transform(const gpe::Array<T, N>& vec, Function fun) -> Array<decltype (fun(vec.front())), N> {
    using ProductType = decltype (fun(vec.front()));
    gpe::Array<ProductType, N> retvec;
    for (unsigned i = 0; i < N; ++i) {
        retvec[i] = fun(vec[i]);
    }
    return retvec;
}

template<typename T, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto transform(const gpe::Array2d<T, N1, N2>& mat, Function fun) -> Array2d<decltype (fun(mat.front().front())), N1, N2> {
    using ProductType = decltype (fun(mat.front().front()));
    gpe::Array2d<ProductType, N1, N2> retmat;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            retmat[i][j] = fun(mat[i][j]);
        }
    }
    return retmat;
}

template<typename T1, typename T2, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto cwise_fun(const gpe::Array<T1, N>& m1,
               const gpe::Array<T2, N>& m2,
               Function fun) -> Array<decltype (fun(m1.front(), m2.front())), N> {
    using ProductType = decltype (fun(m1.front(), m2.front()));
    gpe::Array<ProductType, N> vec;
    for (unsigned i = 0; i < N; ++i) {
        const T1& a = m1[i];
        const T2& b = m2[i];
        vec[i] = fun(a, b);
    }
    return vec;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto cwise_fun(const Array2d<T1, N1, N2>& m1,
               const Array2d<T2, N1, N2>& m2,
               Function fun) -> Array2d<decltype (fun(m1.front().front(), m2.front().front())), N1, N2> {
    using ProductType = decltype (fun(m1.front().front(), m2.front().front()));
    gpe::Array2d<ProductType, N1, N2> matrix;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            const T1& a = m1[i][j];
            const T2& b = m2[i][j];
            ProductType v = fun(a, b);
            matrix[i][j] = v;
        }
    }
    return matrix;
}

/// multiplies every row in m with the corresponding element in v (column vector)
template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto cwise_fun(const Array2d<T1, N1, N2>& m,
               const Array<T2, N1>& v,
               Function fun) -> Array2d<decltype (fun(m.front().front(), v.front())), N1, N2> {
    using ProductType = decltype (fun(m.front().front(), v.front()));
    gpe::Array2d<ProductType, N1, N2> matrix;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            const T1& a = m[i][j];
            const T2& b = v[i];
            ProductType v = fun(a, b);
            matrix[i][j] = v;
        }
    }
    return matrix;
}

/// multiplies every column in m with the corresponding element in v (row vector)
template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto cwise_fun(const Array<T1, N2>& v,
               const Array2d<T2, N1, N2>& m,
               Function fun) -> Array2d<decltype (fun(v.front(), m.front().front())), N1, N2> {
    using ProductType = decltype (fun(v.front(), m.front().front()));
    gpe::Array2d<ProductType, N1, N2> matrix;
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            const T1& a = v[j];
            const T2& b = m[i][j];
            ProductType v = fun(a, b);
            matrix[i][j] = v;
        }
    }
    return matrix;
}

template<typename T1, typename T2, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
void cwise_ref_fun(gpe::Array<T1, N>* m1,
                   gpe::Array<T2, N>* m2,
                   Function fun) {
    for (unsigned i = 0; i < N; ++i) {
        fun((*m1)[i], (*m2)[i]);
    }
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
void cwise_ref_fun(Array2d<T1, N1, N2>* m1,
                   Array2d<T2, N1, N2>* m2,
                   Function fun) {
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            fun((*m1)[i][j], (*m2)[i][j]);
        }
    }
}

/// multiplies every row in m with the corresponding element in v (column vector)
template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
void cwise_ref_fun(Array2d<T1, N1, N2>* m,
                   Array<T2, N1>* v,
                   Function fun) {
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            fun((*m)[i][j], (*v)[i]);
        }
    }
}

/// multiplies every column in m with the corresponding element in v (row vector)
template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
void cwise_ref_fun(Array<T1, N2>* v,
                   Array2d<T2, N1, N2>* m,
                   Function fun) {
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            fun((*v)[j], (*m)[i][j]);
        }
    }
}

template<typename T1, typename T2, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
T2 reduce(const gpe::Array<T1, N>& m1, T2 initial, Function fun) {
    for (unsigned i = 0; i < m1.size(); ++i) {
        const T1& a = m1[i];
        initial = fun(initial, a);
    }
    return initial;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
T2 reduce(const gpe::Array2d<T1, N1, N2>& matrix, T2 initial, Function fun) {
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            initial = fun(initial, matrix[i][j]);
        }
    }
    return initial;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
gpe::Array<T2, N1> reduce_rows(const gpe::Array2d<T1, N1, N2>& matrix, T2 initial, Function fun) {
    gpe::Array<T2, N1> retvec;
    for (unsigned i = 0; i < N1; ++i) {
        retvec[i] = initial;
        for (unsigned j = 0; j < N2; ++j) {
            retvec[i] = fun(retvec[i], matrix[i][j]);
        }
    }
    return retvec;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
    gpe::Array<T2, N2> reduce_cols(const gpe::Array2d<T1, N1, N2>& matrix, T2 initial, Function fun) {
    gpe::Array<T2, N2> retvec;
    for (unsigned j = 0; j < N2; ++j) {
        retvec[j] = initial;
    }
    for (unsigned i = 0; i < N1; ++i) {
        for (unsigned j = 0; j < N2; ++j) {
            retvec[j] = fun(retvec[j], matrix[i][j]);
        }
    }
    return retvec;
}

// //////////////////////////////////////// vector algorithms //////////////////////////////////////////

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto outer_product(const gpe::Vector<T1, N1>& m1,
                   const gpe::Vector<T2, N2>& m2,
                   Function fun) -> Vector2d<decltype (fun(m1.front(), m2.front())), N1, N2> {
    using ProductType = decltype (fun(m1.front(), m2.front()));
    gpe::Vector2d<ProductType, N1, N2> matrix;
    matrix.resize(m1.size());
    for (unsigned i = 0; i < m1.size(); ++i) {
        matrix[i].resize(m2.size());
        const T1& c_i = m1[i];
        for (unsigned j = 0; j < m2.size(); ++j) {
            const T2& c_j = m2[j];
            ProductType v = fun(c_i, c_j);
            matrix[i][j] = v;
        }
    }
    return matrix;
}

template<typename T, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto transform(const gpe::Vector<T, N>& vec, Function fun) -> Vector<decltype (fun(vec.front())), N> {
    using ProductType = decltype (fun(vec.front()));
    gpe::Vector<ProductType, N> retvec;
    retvec.resize(vec.size());
    for (unsigned i = 0; i < vec.size(); ++i) {
        retvec[i] = fun(vec[i]);
    }
    return retvec;
}

template<typename T, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto transform(const gpe::Vector2d<T, N1, N2>& mat, Function fun) -> Vector2d<decltype (fun(mat.front().front())), N1, N2> {
    using ProductType = decltype (fun(mat.front().front()));
    gpe::Vector2d<ProductType, N1, N2> retmat;
    retmat.resize(mat.size());
    for (unsigned i = 0; i < mat.size(); ++i) {
        assert(mat[0].size() == mat[i].size());
        assert(mat[i].size() == mat[i].size());
        retmat[i].resize(mat[i].size());
        for (unsigned j = 0; j < mat[i].size(); ++j) {
            retmat[i][j] = fun(mat[i][j]);
        }
    }
    return retmat;
}

template<typename T1, typename T2, uint32_t N, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto cwise_fun(const gpe::Vector<T1, N>& m1,
               const gpe::Vector<T2, N>& m2,
               Function fun) -> Vector<decltype (fun(m1.front(), m2.front())), N> {
    using ProductType = decltype (fun(m1.front(), m2.front()));
    assert(m1.size() == m2.size());
    gpe::Vector<ProductType, N> vec;
    vec.resize(m1.size());
    for (unsigned i = 0; i < m1.size(); ++i) {
        const T1& a = m1[i];
        const T2& b = m2[i];
        vec[i] = fun(a, b);
    }
    return vec;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto cwise_fun(const Vector2d<T1, N1, N2>& m1,
               const Vector2d<T2, N1, N2>& m2,
               Function fun) -> Vector2d<decltype (fun(m1.front().front(), m2.front().front())), N1, N2> {
    using ProductType = decltype (fun(m1.front().front(), m2.front().front()));
    assert(m1.size() == m2.size());
    gpe::Vector2d<ProductType, N1, N2> matrix;
    matrix.resize(m1.size());
    for (unsigned i = 0; i < m1.size(); ++i) {
        assert(m1[0].size() == m1[i].size());
        assert(m1[i].size() == m2[i].size());
        matrix[i].resize(m1[i].size());
        for (unsigned j = 0; j < m1[i].size(); ++j) {
            const T1& a = m1[i][j];
            const T2& b = m2[i][j];
            matrix[i][j] = fun(a, b);
        }
    }
    return matrix;
}

/// multiplies every row in m with the corresponding element in v (column vector)
template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto cwise_fun(const Vector2d<T1, N1, N2>& m,
               const Vector<T2, N1>& v,
               Function fun) -> Vector2d<decltype (fun(m.front().front(), v.front())), N1, N2> {
    using ProductType = decltype (fun(m.front().front(), v.front()));
    gpe::Vector2d<ProductType, N1, N2> matrix;
    assert(m.size() == v.size());
    matrix.resize(m.size());
    for (unsigned i = 0; i < m.size(); ++i) {
        assert(m[0].size() == m[i].size());
        matrix[i].resize(m[i].size());
        for (unsigned j = 0; j < m[i].size(); ++j) {
            const T1& a = m[i][j];
            const T2& b = v[i];
            matrix[i][j] = fun(a, b);
        }
    }
    return matrix;
}

/// multiplies every column in m with the corresponding element in v (row vector)
template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
auto cwise_fun(const Vector<T1, N2>& v,
               const Vector2d<T2, N1, N2>& m,
               Function fun) -> Vector2d<decltype (fun(v.front(), m.front().front())), N1, N2> {
    using ProductType = decltype (fun(v.front(), m.front().front()));
    gpe::Vector2d<ProductType, N1, N2> matrix;
    matrix.resize(m.size());
    for (unsigned i = 0; i < m.size(); ++i) {
        assert(m[0].size() == v.size());
        assert(m[0].size() == m[i].size());
        matrix[i].resize(m[i].size());
        for (unsigned j = 0; j < m[i].size(); ++j) {
            const T2& a = v[j];
            const T1& b = m[i][j];
            matrix[i][j] = fun(a, b);
        }
    }
    return matrix;
}

template<typename T1, typename T2, uint32_t N, typename Function, typename VectorSizeType>
__host__ __device__ GPE_ALGORITHMS_INLINE
T2 reduce(const gpe::Vector<T1, N, VectorSizeType>& m1, T2 initial, Function fun) {
    for (unsigned i = 0; i < m1.size(); ++i) {
        const T1& a = m1[i];
        initial = fun(initial, a);
    }
    return initial;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function, typename VectorSizeType>
__host__ __device__ GPE_ALGORITHMS_INLINE
T2 reduce(const gpe::Vector2d<T1, N1, N2, VectorSizeType>& matrix, T2 initial, Function fun) {
    for (unsigned i = 0; i < matrix.size(); ++i) {
        for (unsigned j = 0; j < matrix[i].size(); ++j) {
            initial = fun(initial, matrix[i][j]);
        }
    }
    return initial;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
gpe::Vector<T2, N1> reduce_rows(const gpe::Vector2d<T1, N1, N2>& matrix, T2 initial, Function fun) {
    gpe::Vector<T2, N1> retvec;
    retvec.resize(matrix.size());
    for (unsigned i = 0; i < matrix.size(); ++i) {
        assert(matrix[0].size() == matrix[i].size());
        retvec[i] = initial;
        for (unsigned j = 0; j < matrix[i].size(); ++j) {
            retvec[i] = fun(retvec[i], matrix[i][j]);
        }
    }
    return retvec;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_ALGORITHMS_INLINE
    gpe::Vector<T2, N2> reduce_cols(const gpe::Vector2d<T1, N1, N2>& matrix, T2 initial, Function fun) {
    gpe::Vector<T2, N2> retvec;
    if (matrix[0].size()) {
        retvec.resize(matrix[0].size());
        for (unsigned j = 0; j < retvec.size(); ++j) {
            retvec[j] = initial;
        }
    }
    for (unsigned i = 0; i < matrix.size(); ++i) {
        assert(matrix[0].size() == matrix[i].size());
        for (unsigned j = 0; j < matrix[i].size(); ++j) {
            retvec[j] = fun(retvec[j], matrix[i][j]);
        }
    }
    return retvec;
}

}

#endif // GPE_UTIL_ALGORITHMS_H
