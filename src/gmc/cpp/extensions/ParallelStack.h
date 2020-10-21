#ifndef PARALLELSTACK_H
#define PARALLELSTACK_H

#include "cuda_operations.h"
#include "math/scalar.h"

namespace gpe {

template <typename T, size_t SIZE>
struct ParallelStack {
    T stack[SIZE];
    int32_t head = 0;

    /// must be called from every thread
    __host__ __device__ void push(const T& element, bool is_valid, unsigned thread_id) {
        auto vote = gpe::ballot_sync(0xFFFFFFFF, is_valid, thread_id);
        auto write_location = gpe::popc(((1 << thread_id) - 1) & vote);
        assert(head + write_location < SIZE);
        if (is_valid)
            stack[head + write_location] = element;
        gpe::syncthreads();
        if (thread_id == 0)
            head += gpe::popc(vote);
        gpe::syncthreads();
    }

    /// must be called from every thread
    __host__ __device__ bool pop(T* element, int32_t thread_id) {
        auto location = head - 1 - thread_id;
        bool is_valid = location >= 0;
        if (is_valid) {
            *element = stack[location];
        }
        gpe::syncthreads();
        if (thread_id == 0)
            head = gpe::max(int32_t(0), head - 32);
        gpe::syncthreads();
        return is_valid;
    }
};

} // namespace gpe
#endif // PARALLELSTACK_H
