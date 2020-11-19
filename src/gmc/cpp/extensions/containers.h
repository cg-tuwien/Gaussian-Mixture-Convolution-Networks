#ifndef GPE_CONTAINERS_H
#define GPE_CONTAINERS_H

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
    static_assert (N > 0, "an array of size 0 doesn't appear usefull and would break front and back functions.");

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

    __host__ __device__ GPE_CONTAINER_INLINE
    T& front() {
        return data[0];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T& front() const {
        return data[0];
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    T& back() {
        return data[N - 1];
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T& back() const {
        return data[N - 1];
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    T* begin() {
        return data;
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T* begin() const {
        return data;
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    T* end() {
        return data + N;
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T* end() const {
        return data + N;
    }

};

template<typename T, uint32_t N_ROWS, uint32_t N_COLS = N_ROWS>
using Array2d = Array<Array<T, N_COLS>, N_ROWS>;


// configurable size_t just to fix padding warnings.
template<typename T, uint32_t N, typename size_type = uint32_t>
struct Vector {
    Array<T, N> data;
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
    T* begin() {
        return data.begin();
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T* begin() const {
        return data.begin();
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    T* end() {
        return data.begin() + size();
    }
    __host__ __device__ GPE_CONTAINER_INLINE
    const T* end() const {
        return data.begin() + size();
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

template<typename T, uint32_t N, typename VectorSizeType>
__host__ __device__ GPE_CONTAINER_INLINE
Array<T, N> to_array(const Vector<T, N, VectorSizeType>& vector, const T& default_element) {
    Array<T, N> retval = { vector.data };
    for (uint32_t i = vector.size(); i < N; ++i) {
        retval[i] = default_element;
    }
    return retval;
}

template<typename T, uint32_t N_ROWS, uint32_t N_COLS = N_ROWS, typename size_type = uint32_t>
using Vector2d = Vector<Vector<T, N_COLS, size_type>, N_ROWS, size_type>;

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

}

#endif // GPE_CONTAINERS_H
