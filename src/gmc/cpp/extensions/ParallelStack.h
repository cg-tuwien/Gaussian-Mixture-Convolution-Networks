#ifndef PARALLELSTACK_H
#define PARALLELSTACK_H

#include "cuda_operations.h"
#include "util/cuda.h"
#include "util/scalar.h"

namespace gpe {

template <typename T, size_t SIZE, unsigned SYNC_ID>
struct ParallelStack {
    gpe::Array<T, SIZE>& stack = nullptr;
    // head is not in shared memory. if you change that, you have to check for thread_id == 0 before writing and sync before reading.
    uint32_t head = 0;

    /// must be called from every thread
    EXECUTION_DEVICES
    void push(const T& element, bool is_valid, uint32_t thread_id) {
        auto vote = gpe::ballot_sync(0xFFFFFFFF, is_valid, thread_id, SYNC_ID + 1000 + 0);
        auto write_location = gpe::popc(((1 << thread_id) - 1) & vote);
        assert(head + write_location < SIZE);
        if (is_valid)
            stack[head + write_location] = element;
        gpe::syncwarp(SYNC_ID + 1000 + 1);
        // head is not in shared memory, all threads make the update.  dunno if that is good perf wise..
        head += gpe::popc(vote);
//        gpe::syncwarp(SYNC_ID + 1000 + 2);
    }

    /// must be called from every thread
    EXECUTION_DEVICES
    bool pop(T* element, uint32_t thread_id) {
        auto location = head - 1 - thread_id;
        bool is_valid = location < SIZE;
        if (is_valid) {
            *element = stack[location];
        }
//        gpe::syncwarp(SYNC_ID + 1000 + 3);
//        if (thread_id == 0)
        head = gpe::max(uint32_t(32), head) - 32;
//        gpe::syncwarp(SYNC_ID + 1000 + 4);
        return is_valid;
    }

    EXECUTION_DEVICES
    bool contains_elements(uint32_t thread_id) {
//        gpe::syncwarp(SYNC_ID + 1000 + 5);
        return head > 0;
    }
};

} // namespace gpe
#endif // PARALLELSTACK_H
