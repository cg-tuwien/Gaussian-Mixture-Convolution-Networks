#ifndef GPE_UTIL_CUDA_H
#define GPE_UTIL_CUDA_H

#include <cuda_runtime.h>

#ifdef __CUDACC__
#define EXECUTION_DEVICES __host__ __device__ /*__forceinline__*/
#else
#define EXECUTION_DEVICES
#endif

namespace gpe {
template< class T > struct remove_cv                   { using type = T; };
template< class T > struct remove_cv<const T>          { using type = T; };
template< class T > struct remove_cv<volatile T>       { using type = T; };
template< class T > struct remove_cv<const volatile T> { using type = T; };
template< class T >
using remove_cv_t = typename remove_cv<T>::type;

template< class T > struct remove_reference      { using type = T;};
template< class T > struct remove_reference<T&>  { using type = T;};
template< class T > struct remove_reference<T&&> { using type = T;};
template< class T >
using remove_reference_t = typename remove_reference<T>::type;

template< class T > struct remove_cvref { using type = remove_cv_t<remove_reference_t<T>>;};
template< class T >
using remove_cvref_t = typename remove_cvref<T>::type;
}

#endif // GPE_UTIL_CUDA_H
