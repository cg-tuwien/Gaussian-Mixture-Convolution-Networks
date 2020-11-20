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

// min heap
template<typename T, uint32_t N>
struct ArrayHeap {
    static_assert (N > 1, "N must be greater 1 (it might work with 1, but it's not tested)");
    Array<T, N> m_data;

    __host__ __device__ GPE_CONTAINER_INLINE
    ArrayHeap() = default;
    __host__ __device__ GPE_CONTAINER_INLINE
    ArrayHeap (const Array<T, N>& data) : m_data(data) {
        build();
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    T replaceRoot(const T& value) {
        return replaceElement(value, 0);
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    void build() {
        // build heap using Floyd's algorithm
        for (unsigned i = parentIndex(N-1); i < N; --i) {   // mind the overflow. it'll stop at 0
            const auto copy = m_data[i];
            replaceElement(copy, i);
        }
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    T replaceElement(const T& value, const uint32_t index) {
        assert(&value < m_data.begin() || &value >= m_data.end()); // pass by value if you want to relax this condition.
        const T retval = m_data[index];
        uint32_t current_idx = index;
        while (hasTwoChildren(current_idx)) {
            const auto left_idx = leftChildIndex(current_idx);
            const auto right_idx = rightChildIndex(current_idx);
            const auto left_val = m_data[left_idx];
            const auto right_val = m_data[right_idx];

            if (value <= left_val && value <= right_val) {
                break;
            }
            if (left_val <= right_val) {
                m_data[current_idx] = left_val;
                current_idx = left_idx;
            }
            else {
                // right_val < left_val
                m_data[current_idx] = right_val;
                current_idx = right_idx;
            }
        }
        if (!isLeafIndex(current_idx)) {
            // only left child left or inserting in the middle (in which case we won't enter the if)
            const auto left_idx = leftChildIndex(current_idx);
            const auto left_val = m_data[left_idx];
            if (left_val <= value) {
                m_data[current_idx] = left_val;
                current_idx = left_idx;
            }
        }
        m_data[current_idx] = value;
        return retval;
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    uint32_t leftChildIndex(uint32_t index) const {
        assert(index * 2 + 1 < N);
        return index * 2 + 1;
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    uint32_t rightChildIndex(uint32_t index) const {
        assert(index * 2 + 2 < N);
        return index * 2 + 2;
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    uint32_t parentIndex(uint32_t index) const {
        assert(index > 0);
        assert(index < N);
        return (index - 1) / 2;
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    bool isLeafIndex(uint32_t index) const {
        assert(index < N);
        return index >= N/2;
    }

    __host__ __device__ GPE_CONTAINER_INLINE
    bool hasTwoChildren(uint32_t index) const {
        assert(index < N);
        return index < (N - 1) / 2;
    }
};

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
