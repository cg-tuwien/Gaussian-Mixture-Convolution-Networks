#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>

// avoid unused warning
#define GPE_UNUSED(x) (void)x;

#if defined(GPE_NO_CUDA_ERROR_CHECKING) or defined(NDEBUG)
#define GPE_CUDA_ASSERT(ans)
#else
#define GPE_CUDA_ASSERT(ans) { gpeGpuAssert((ans), __FILE__, __LINE__); }
inline void gpeGpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#endif

using uint = unsigned int;
using uchar = unsigned char;


#endif // COMMON_H
