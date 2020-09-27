#ifndef CUDA_QT_CREATOR_DEFINITINOS_H
#define CUDA_QT_CREATOR_DEFINITINOS_H

#include <cuda_runtime.h>
#include <algorithm>

#ifndef __CUDACC__

int atomicCAS(int* address, int compare, int val);
void __syncthreads();

constexpr dim3 gridDim;
constexpr dim3 blockDim;
constexpr dim3 blockIdx;
constexpr dim3 threadIdx;

using std::min;
using std::max;

namespace torch {
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};

}
#endif

#endif // CUDA_QT_CREATOR_DEFINITINOS_H
