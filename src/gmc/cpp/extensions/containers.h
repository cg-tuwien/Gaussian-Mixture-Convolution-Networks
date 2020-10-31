#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <bitset>
#include <cassert>
#include <cinttypes>
#include <initializer_list>

#include <cuda_runtime.h>

#ifdef NDEBUG
#define GPE_CONTAINER_INLINE __forceinline__
#else
#define GPE_CONTAINER_INLINE
#endif

namespace gpe {
template<typename T, uint32_t N>
struct Array {
    T data[N];
    __host__ __device__ GPE_CONTAINER_INLINE
    T& operator[](uint32_t i) {
        assert(i < N);
        return data[i];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T& operator[](uint32_t i) const {
        assert(i < N);
        return data[i];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    constexpr uint32_t size() const {
        return N;
    }
};

template<size_t N>
class BitSet {
    static constexpr uint32_t N_INT_BITS = CHAR_BIT * sizeof(uint32_t);
    uint32_t m_data[(N + N_INT_BITS - 1) / N_INT_BITS];

    __host__ __device__ GPE_CONTAINER_INLINE
    uint32_t& wordOf(unsigned p) {
        return m_data[p / N_INT_BITS];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const uint32_t& wordOf(unsigned p) const {
        return m_data[p / N_INT_BITS];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    uint32_t bitOf(unsigned p) const {
        return 1u << (p % N_INT_BITS);
    }

public:
    __host__ __device__ GPE_CONTAINER_INLINE
    BitSet() : m_data() {}

    __host__ __device__ GPE_CONTAINER_INLINE
    void set0(unsigned p) {
        assert(p < N);
        wordOf(p) &= ~bitOf(p);
    }
    __host__ __device__ GPE_CONTAINER_INLINE
        void set1(unsigned p) {
        assert(p < N);
        wordOf(p) |= bitOf(p);
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    void set(unsigned p, bool val) {
        assert(p < N);
        if (val) set1(p);
        else set0(p);
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    bool isSet(unsigned p) const {
        assert(p < N);
        return bool(wordOf(p) & bitOf(p));
    }
};


template<typename T, uint32_t N_ROWS, uint32_t N_COLS = N_ROWS>
using Array2d = Array<Array<T, N_COLS>, N_ROWS>;

// configurable size_t just to fix padding warnings.
template<typename T, uint32_t N, typename size_type = uint32_t>
struct Vector {
    T data[N];
    size_type m_size = 0;
    static_assert (N < (1u << 31), "N is too large; size will be incorrect due to uint32_t cast.");

    __host__ __device__ GPE_CONTAINER_INLINE
    Vector() = default;

    template <typename... TT>
    __host__ __device__ GPE_CONTAINER_INLINE
    Vector(TT... ts) : data{ts...} { // note the use of brace-init-list
        constexpr unsigned size = sizeof...(ts);
        static_assert (size <= N, "init list has too many elements");
        m_size = size;
    }

    template <typename size_type_other>
    __host__ __device__ GPE_CONTAINER_INLINE
    Vector(const Vector<T, N, size_type_other>& other) : data{other.data}, m_size(other.m_size) {}


    template <typename size_type_other>
    __host__ __device__ GPE_CONTAINER_INLINE
    Vector<T, N, size_type>& operator=(const Vector<T, N, size_type_other>& other) {
        m_size = other.m_size;
        for (unsigned i = 0; i < N; ++i)
            data[i] = other.data[i];
        return *this;
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    static Vector<T, N, size_type> filled(const T& value, unsigned n_elements) {
        assert(n_elements < N);
        Vector<T, N, size_type> vec;
        vec.resize(n_elements);
        for (int i = 0; i < n_elements; ++i) {
            vec[i] = value;
        }
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    T& operator[](size_t i) {
        assert(i < m_size);
        return data[i];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T& operator[](size_t i) const {
        assert(i < m_size);
        return data[i];
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    T& front() {
        assert(m_size >= 1);
        return data[0];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T& front() const {
        assert(m_size >= 1);
        return data[0];
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    T& back() {
        assert(m_size >= 1);
        return data[m_size - 1];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T& back() const {
        assert(m_size >= 1);
        return data[m_size - 1];
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    uint32_t size() const {
        return uint32_t(m_size);
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    void resize(uint32_t new_size) {
        assert(new_size <= N);
        m_size = new_size;
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    void push_back(T v) {
        assert(m_size < N);
        data[m_size] = v;
        m_size++;
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    T pop_back() {
        assert(m_size > 0);
        --m_size;
        return data[m_size];
    }

    template<uint32_t N_, typename size_type_ = uint32_t>
    __host__ __device__ GPE_CONTAINER_INLINE
    void push_back(const Vector<T, N_, size_type_>  v) {
        assert(v.size() + size() <= N);
        for (uint32_t i = 0; i < v.size(); ++i)
            push_back(v[i]);
    }

    template<uint32_t N_, typename size_type_ = uint32_t, typename Predicate>
    __host__ __device__ GPE_CONTAINER_INLINE
    void push_back_if(const Vector<T, N_, size_type_>  v, Predicate condition) {
        assert(v.size() + size() <= N);
        for (uint32_t i = 0; i < v.size(); ++i) {
            if (condition(v[i]))
                push_back(v[i]);
        }
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    void clear() {
        m_size = 0;
    }


};

template<typename T, uint32_t N_ROWS, uint32_t N_COLS = N_ROWS, typename size_type = uint32_t>
using Vector2d = Vector<Vector<T, N_COLS, size_type>, N_ROWS, size_type>;

namespace functors {
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
T plus(const T& a, const T& b) {
    return a + b;
}
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
T minus(const T& a, const T& b) {
    return a - b;
}
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
T times(const T& a, const T& b) {
    return a * b;
}
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
T divided_AbyB(const T& a, const T& b) {
    return a / b;
}
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
T divided_BbyA(const T& a, const T& b) {
    return b / a;
}
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
bool logical_and(const T& a, const T& b) {
    return a && b;
}
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
bool logical_or(const T& a, const T& b) {
    return a || b;
}
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
bool bit_or(const T& a, const T& b) {
    return a | b;
}
template<typename T>
__host__ __device__ GPE_CONTAINER_INLINE
bool bit_and(const T& a, const T& b) {
    return a & b;
}

}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_CONTAINER_INLINE
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
__host__ __device__ GPE_CONTAINER_INLINE
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
__host__ __device__ GPE_CONTAINER_INLINE
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
__host__ __device__ GPE_CONTAINER_INLINE
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
__host__ __device__ GPE_CONTAINER_INLINE
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
            ProductType v = fun(a, b);
            matrix[i][j] = v;
        }
    }
    return matrix;
}

/// multiplies every row in m with the corresponding element in v (column vector)
template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_CONTAINER_INLINE
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
            ProductType v = fun(a, b);
            matrix[i][j] = v;
        }
    }
    return matrix;
}

/// multiplies every column in m with the corresponding element in v (row vector)
template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_CONTAINER_INLINE
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
            ProductType v = fun(a, b);
            matrix[i][j] = v;
        }
    }
    return matrix;
}

template<typename T1, typename T2, uint32_t N, typename Function, typename VectorSizeType>
__host__ __device__ GPE_CONTAINER_INLINE
T2 reduce(const gpe::Vector<T1, N, VectorSizeType>& m1, T2 initial, Function fun) {
    for (unsigned i = 0; i < m1.size(); ++i) {
        const T1& a = m1[i];
        initial = fun(initial, a);
    }
    return initial;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function, typename VectorSizeType>
__host__ __device__ GPE_CONTAINER_INLINE
T2 reduce(const gpe::Vector2d<T1, N1, N2, VectorSizeType>& matrix, T2 initial, Function fun) {
    for (unsigned i = 0; i < matrix.size(); ++i) {
        for (unsigned j = 0; j < matrix[i].size(); ++j) {
            initial = fun(initial, matrix[i][j]);
        }
    }
    return initial;
}

template<typename T1, typename T2, uint32_t N1, uint32_t N2, typename Function>
__host__ __device__ GPE_CONTAINER_INLINE
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
__host__ __device__ GPE_CONTAINER_INLINE
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

#endif // CONTAINERS_H
