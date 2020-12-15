#ifndef PARALLELSTACK_H
#define PARALLELSTACK_H

#include "cuda_operations.h"
#include "util/scalar.h"

//#define GPE_SINGLE_THREADED_MODE

namespace gpe {

template <typename T, size_t SIZE, unsigned SYNC_ID>
struct ParallelStack {
    T stack[SIZE];
    int32_t head = 0;

    /// must be called from every thread
    __host__ __device__ void push(const T& element, bool is_valid, unsigned thread_id) {
        #ifndef GPE_SINGLE_THREADED_MODE
        auto vote = gpe::ballot_sync(0xFFFFFFFF, is_valid, thread_id, SYNC_ID + 1000 + 0);
        #else
        is_valid = is_valid && thread_id == 0;
        auto vote = 1;
        #endif
        auto write_location = gpe::popc(((1 << thread_id) - 1) & vote);
        assert(head + write_location < SIZE);
        if (is_valid)
            stack[head + write_location] = element;
        #ifndef GPE_SINGLE_THREADED_MODE
        gpe::syncthreads(SYNC_ID + 1000 + 1);
        #endif
        if (thread_id == 0)
            head += gpe::popc(vote);
        #ifndef GPE_SINGLE_THREADED_MODE
        gpe::syncthreads(SYNC_ID + 1000 + 2);
        #endif
    }

    /// must be called from every thread
    __host__ __device__ bool pop(T* element, int32_t thread_id) {
        auto location = head - 1 - thread_id;
        bool is_valid = location >= 0;
        if (is_valid) {
            *element = stack[location];
        }
        #ifndef GPE_SINGLE_THREADED_MODE
        gpe::syncthreads(SYNC_ID + 1000 + 3);
        if (thread_id == 0)
            head = gpe::max(int32_t(0), head - 32);
        gpe::syncthreads(SYNC_ID + 1000 + 4);
        #else
        if (thread_id == 0)
            head--;
        #endif
        return is_valid;
    }

    __host__ __device__ bool contains_elements(int32_t thread_id) {
        #ifndef GPE_SINGLE_THREADED_MODE
        gpe::syncthreads(SYNC_ID + 1000 + 5);
        #endif
        return head > 0;
    }
};

} // namespace gpe
#endif // PARALLELSTACK_H
