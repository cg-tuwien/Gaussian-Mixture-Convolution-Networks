#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>

// avoid unused warning
#define GPE_UNUSED(x) (void)x;

using uint = unsigned int;

namespace gpe {

template<typename Assignable1, typename Assignable2>
__host__ __device__
    inline void swap(Assignable1 &a, Assignable2 &b)
{
    Assignable1 temp = a;
    a = b;
    b = temp;
}

}


#endif // COMMON_H
