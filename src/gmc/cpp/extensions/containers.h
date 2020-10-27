#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <bitset>
#include <cassert>
#include <cinttypes>
#include <initializer_list>

#include <cuda_runtime.h>

namespace gpe {
template<typename T, uint32_t N>
struct Array {
    T data[N];
    __host__ __device__ __forceinline__
    T& operator[](uint32_t i) {
        assert(i < N);
        return data[i];
    }
    __host__ __device__ __forceinline__
    const T& operator[](uint32_t i) const {
        assert(i < N);
        return data[i];
    }
    __host__ __device__ __forceinline__
    constexpr uint32_t size() const {
        return N;
    }
};

template<size_t N>
class BitSet {
    static constexpr uint32_t N_INT_BITS = CHAR_BIT * sizeof(uint32_t);
    uint32_t m_data[(N + N_INT_BITS - 1) / N_INT_BITS];

    __host__ __device__
    uint32_t& wordOf(unsigned p) {
        return m_data[p / N_INT_BITS];
    }
    __host__ __device__
    const uint32_t& wordOf(unsigned p) const {
        return m_data[p / N_INT_BITS];
    }
    __host__ __device__
    uint32_t bitOf(unsigned p) const {
        return 1u << (p % N_INT_BITS);
    }

public:
    __host__ __device__
    BitSet() : m_data() {}

    __host__ __device__
    void set0(unsigned p) {
        assert(p < N);
        wordOf(p) &= ~bitOf(p);
    }
    __host__ __device__
        void set1(unsigned p) {
        assert(p < N);
        wordOf(p) |= bitOf(p);
    }
    __host__ __device__
    void set(unsigned p, bool val) {
        assert(p < N);
        if (val) set1(p);
        else set0(p);
    }
    __host__ __device__
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

    __host__ __device__ __forceinline__
    Vector() = default;

    template <typename... TT>
    __host__ __device__ __forceinline__
    Vector(TT... ts) : data{ts...} { // note the use of brace-init-list
        constexpr unsigned size = sizeof...(ts);
        static_assert (size <= N, "init list has too many elements");
        m_size = size;
    }

    __host__ __device__ __forceinline__
    T& operator[](size_t i) {
        assert(i < m_size);
        return data[i];
    }
    __host__ __device__ __forceinline__
    const T& operator[](size_t i) const {
        assert(i < m_size);
        return data[i];
    }

    __host__ __device__ __forceinline__
    T& front() {
        assert(m_size >= 1);
        return data[0];
    }
    __host__ __device__ __forceinline__
    const T& front() const {
        assert(m_size >= 1);
        return data[0];
    }

    __host__ __device__ __forceinline__
    T& back() {
        assert(m_size >= 1);
        return data[m_size - 1];
    }
    __host__ __device__ __forceinline__
    const T& back() const {
        assert(m_size >= 1);
        return data[m_size - 1];
    }

    __host__ __device__ __forceinline__
    uint32_t size() const {
        return uint32_t(m_size);
    }
    __host__ __device__ __forceinline__
    void resize(uint32_t new_size) {
        assert(new_size <= N);
        m_size = new_size;
    }
    __host__ __device__ __forceinline__
    void push_back(T v) {
        assert(m_size < N);
        data[m_size] = v;
        m_size++;
    }
    __host__ __device__ __forceinline__
    T pop_back() {
        assert(m_size > 0);
        --m_size;
        return data[m_size];
    }

    template<uint32_t N_, typename size_type_ = uint32_t>
    __host__ __device__ __forceinline__
    void push_all_back(const Vector<T, N_, size_type_>  v) {
        assert(v.size() + size() <= N);
        for (uint32_t i = 0; i < v.size(); ++i)
            push_back(v[i]);
    }

    __host__ __device__ __forceinline__
    void clear() {
        m_size = 0;
    }


};

template<typename T, uint32_t N_ROWS, uint32_t N_COLS = N_ROWS, typename size_type = uint32_t>
using Vector2d = Vector<Vector<T, N_COLS, size_type>, N_ROWS, size_type>;

static struct UnitTests {
    UnitTests() {
        testBitSet();
    }

    void testBitSet() {
        BitSet<32> bs;
        for (unsigned i = 0; i < 32; i++)
            assert(bs.isSet(i) == false);

        bs.set1(10);
        bs.set1(15);
        bs.set1(20);

        for (unsigned i = 0; i < 32; i++) {
            if (i != 10 && i != 15 && i != 20)
                assert(bs.isSet(i) == false);
            else
                assert(bs.isSet(i) == true);
        }

        for (unsigned i = 0; i < 32; i++)
            bs.set1(i);

        for (unsigned i = 0; i < 32; i++)
            assert(bs.isSet(i) == true);

        bs.set0(15);
        bs.set0(16);
        bs.set0(17);

        for (unsigned i = 0; i < 32; i++) {
            if (i != 15 && i != 16 && i != 17)
                assert(bs.isSet(i) == true);
            else
                assert(bs.isSet(i) == false);
        }

        for (unsigned i = 0; i < 32; i++)
            bs.set0(i);

        for (unsigned i = 0; i < 32; i++)
            assert(bs.isSet(i) == false);
    }
} unit_tests;

}

#endif // CONTAINERS_H
