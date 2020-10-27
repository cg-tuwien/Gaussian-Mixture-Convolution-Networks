#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <cassert>
#include <cinttypes>

#include <cuda_runtime.h>

namespace gpe {
template<typename T, size_t N>
struct Array {
    T data[N];
    __host__ __device__ __forceinline__
    T& operator[](size_t i) {
        assert(i < N);
        return data[i];
    }
    __host__ __device__ __forceinline__
    const T& operator[](size_t i) const {
        assert(i < N);
        return data[i];
    }
    __host__ __device__ __forceinline__
    size_t size() const {
        return N;
    }
};

// configurable size_t just to fix padding warnings.
template<typename T, uint32_t N, typename size_type = uint32_t>
struct Vector {
    T data[N];
    size_type m_size = 0;
    static_assert (N < (1u << 31), "N is too large; size will be incorrect due to uint32_t cast.");

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
}

#endif // CONTAINERS_H
