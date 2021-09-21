/*
 * This is a hacked version of pytorch's packed accessor to make it work on cpu as well.
 * The goal is to fix this bug: https://github.com/pytorch/pytorch/issues/45444
 * The file is based on the pytorch header, so the license of pytorch applies:
 *
 * (c) 2020 Adam Celarek
 *
 * From PyTorch:
 * 
 * Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
 * Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
 * Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
 * Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
 * Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
 * Copyright (c) 2011-2013 NYU                      (Clement Farabet)
 * Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
 * Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
 * Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
 * 
 * From Caffe2:
 * 
 * Copyright (c) 2016-present, Facebook Inc. All rights reserved.
 * 
 * All contributions by Facebook:
 * Copyright (c) 2016 Facebook Inc.
 * 
 * All contributions by Google:
 * Copyright (c) 2015 Google Inc.
 * All rights reserved.
 * 
 * All contributions by Yangqing Jia:
 * Copyright (c) 2015 Yangqing Jia
 * All rights reserved.
 * 
 * All contributions from Caffe:
 * Copyright(c) 2013, 2014, 2015, the respective contributors
 * All rights reserved.
 * 
 * All other contributions:
 * Copyright(c) 2015, 2016 the respective contributors
 * All rights reserved.
 * 
 * Caffe2 uses a copyright model similar to Caffe: each contributor holds
 * copyright over their contributions to Caffe2. The project versioning records
 * all such contribution and copyright details. If a contributor wants to further
 * mark their specific copyright on a particular contribution, they should
 * indicate their copyright solely in the commit message of the change when it is
 * committed.
 * 
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories America
 * and IDIAP Research Institute nor the names of its contributors may be
 * used to endorse or promote products derived from this software without
 * specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef HACKED_ACCESSOR_H
#define HACKED_ACCESSOR_H

#include <ATen/core/TensorAccessor.h>
#include <c10/util/ArrayRef.h>
#include <torch/types.h>

namespace gpe {

template<typename T>
struct TorchTypeMapper;

template<>
struct TorchTypeMapper<int16_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Short; }
};

template<>
struct TorchTypeMapper<int32_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Int; }
};

template<>
struct TorchTypeMapper<int64_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Long; }
};

template<>
struct TorchTypeMapper<uint16_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Short; }
};

template<>
struct TorchTypeMapper<uint32_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Int; }
};

template<>
struct TorchTypeMapper<uint64_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Long; }
};

template<>
struct TorchTypeMapper<float> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Float; }
};

template<>
struct TorchTypeMapper<double> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Double; }
};

template <typename T>
struct RestrictPtrTraits {
    typedef T* __restrict__ PtrType;
};

// TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
// For CUDA tensors it is used in device code (only). This means that we restrict ourselves
// to functions and types available there (e.g. IntArrayRef isn't).

// The PtrTraits argument is only relevant to cuda to support `__restrict__` pointers.
template<typename T, size_t N, template <typename U> class PtrTraits = RestrictPtrTraits, typename index_t = int64_t>
class TensorAccessorBase {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

    C10_HOST_DEVICE TensorAccessorBase(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : data_(data_), sizes_(sizes_), strides_(strides_) {}
    C10_HOST c10::IntArrayRef sizes() const {
        return c10::IntArrayRef(sizes_,N);
    }
    C10_HOST c10::IntArrayRef strides() const {
        return c10::IntArrayRef(strides_,N);
    }
    C10_HOST_DEVICE index_t stride(index_t i) const {
        return strides_[i];
    }
    C10_HOST_DEVICE index_t size(index_t i) const {
        return sizes_[i];
    }
    C10_HOST_DEVICE PtrType data() {
        return data_;
    }
    C10_HOST_DEVICE const PtrType data() const {
        return data_;
    }
protected:
    PtrType data_;
    const index_t* sizes_;
    const index_t* strides_;
};

// The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
// `Tensor.accessor<T, N>()`.
// For CUDA `Tensor`s, `GenericPackedTensorAccessor` is used on the host and only
// indexing on the device uses `TensorAccessor`s.
template<typename T, size_t N, template <typename U> class PtrTraits = RestrictPtrTraits, typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T,N,PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

    C10_HOST_DEVICE TensorAccessor(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : TensorAccessorBase<T, N, PtrTraits, index_t>(data_,sizes_,strides_) {}

    C10_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
        return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
    }

    C10_HOST_DEVICE const TensorAccessor<T, N-1, PtrTraits, index_t> operator[](index_t i) const {
        return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i,this->sizes_+1,this->strides_+1);
    }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T,1,PtrTraits,index_t> : public TensorAccessorBase<T,1,PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;

    C10_HOST_DEVICE TensorAccessor(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : TensorAccessorBase<T, 1, PtrTraits, index_t>(data_,sizes_,strides_) {}
    C10_HOST_DEVICE T & operator[](index_t i) {
        assert(i < this->size(0));
        return this->data_[this->strides_[0]*i];
    }
    C10_HOST_DEVICE const T & operator[](index_t i) const {
        assert(i < this->size(0));
        return this->data_[this->strides_[0]*i];
    }
};

// GenericPackedTensorAccessorBase and GenericPackedTensorAccessor are used on for CUDA `Tensor`s on the host
// and as
// In contrast to `TensorAccessor`s, they copy the strides and sizes on instantiation (on the host)
// in order to transfer them on the device when calling kernels.
// On the device, indexing of multidimensional tensors gives to `TensorAccessor`s.
// Use RestrictPtrTraits as PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
// Instantiation from data, sizes, strides is only needed on the host and std::copy isn't available
// on the device, so those functions are host only.
template<typename T, size_t N, template <typename U> class PtrTraits = RestrictPtrTraits, typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;
    C10_HOST GenericPackedTensorAccessorBase() = default;
    C10_HOST GenericPackedTensorAccessorBase(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : data_(data_) {
        std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
        std::copy(strides_, strides_ + N, std::begin(this->strides_));
    }

    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
    C10_HOST GenericPackedTensorAccessorBase(
        PtrType data_,
        const source_index_t* sizes_,
        const source_index_t* strides_)
        : data_(data_) {
        for (int i = 0; i < N; i++) {
            this->sizes_[i] = sizes_[i];
            this->strides_[i] = strides_[i];
        }
    }

    C10_HOST_DEVICE index_t stride(index_t i) const {
        return strides_[i];
    }
    C10_HOST_DEVICE index_t size(index_t i) const {
        return sizes_[i];
    }
    C10_HOST_DEVICE PtrType data() {
        return data_;
    }
    C10_HOST_DEVICE const PtrType data() const {
        return data_;
    }
protected:
    PtrType data_;
    index_t sizes_[N];
    index_t strides_[N];
};

template<typename T, size_t N, template <typename U> class PtrTraits = RestrictPtrTraits, typename index_t = int64_t>
class GenericPackedTensorAccessor : public GenericPackedTensorAccessorBase<T,N,PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;
    C10_HOST GenericPackedTensorAccessor() = default;
    C10_HOST GenericPackedTensorAccessor(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
    C10_HOST GenericPackedTensorAccessor(
        PtrType data_,
        const source_index_t* sizes_,
        const source_index_t* strides_)
        : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(data_, sizes_, strides_) {}

    C10_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) {
        index_t* new_sizes = this->sizes_ + 1;
        index_t* new_strides = this->strides_ + 1;
        assert(i < this->size(0));
        return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
    }

    C10_HOST_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](index_t i) const {
        const index_t* new_sizes = this->sizes_ + 1;
        const index_t* new_strides = this->strides_ + 1;
        assert(i < this->size(0));
        return TensorAccessor<T,N-1,PtrTraits,index_t>(this->data_ + this->strides_[0]*i, new_sizes, new_strides);
    }
};

template<typename T, template <typename U> class PtrTraits, typename index_t>
class GenericPackedTensorAccessor<T,1,PtrTraits,index_t> : public GenericPackedTensorAccessorBase<T,1,PtrTraits,index_t> {
public:
    typedef typename PtrTraits<T>::PtrType PtrType;
    C10_HOST GenericPackedTensorAccessor() = default;
    C10_HOST GenericPackedTensorAccessor(
        PtrType data_,
        const index_t* sizes_,
        const index_t* strides_)
        : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

    // if index_t is not int64_t, we want to have an int64_t constructor
    template <typename source_index_t, class = typename std::enable_if<std::is_same<source_index_t, int64_t>::value>::type>
    C10_HOST GenericPackedTensorAccessor(
        PtrType data_,
        const source_index_t* sizes_,
        const source_index_t* strides_)
        : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(data_, sizes_, strides_) {}

    C10_HOST_DEVICE T & operator[](index_t i) {
        assert(i < this->size(0));
        return this->data_[this->strides_[0] * i];
    }
    C10_HOST_DEVICE const T& operator[](index_t i) const {
        assert(i < this->size(0));
        return this->data_[this->strides_[0]*i];
    }
};

template <typename T, size_t N, template <typename U> class PtrTraits = RestrictPtrTraits>
using PackedTensorAccessor32 = GenericPackedTensorAccessor<T, N, PtrTraits, uint32_t>;

template <typename T, size_t N, template <typename U> class PtrTraits = RestrictPtrTraits>
using PackedTensorAccessor64 = GenericPackedTensorAccessor<T, N, PtrTraits, uint64_t>;


//template<typename scalar_t, size_t N>
//C10_HOST auto accessor(const torch::Tensor& tensor) {
//    auto torch_accessor = tensor.packed_accessor32<scalar_t, N, gpe::RestrictPtrTraits>();
//    return PackedTensorAccessor32<scalar_t, N, gpe::RestrictPtrTraits>(*reinterpret_cast<PackedTensorAccessor32<scalar_t, N, gpe::RestrictPtrTraits>*>(&torch_accessor));
//}

template<typename scalar_t, size_t N, typename tensor_type>
C10_HOST auto accessor(const torch::Tensor& tensor) {
    assert(sizeof (scalar_t) == sizeof (tensor_type));
    auto torch_accessor = tensor.packed_accessor32<tensor_type, N, gpe::RestrictPtrTraits>();
//
//    return PackedTensorAccessor32<scalar_t, N, gpe::RestrictPtrTraits>(reinterpret_cast<scalar_t*>(torch_accessor.data()), sizes.data(), strides.data());
//
    return PackedTensorAccessor32<scalar_t, N, gpe::RestrictPtrTraits>(*reinterpret_cast<PackedTensorAccessor32<scalar_t, N, gpe::RestrictPtrTraits>*>(&torch_accessor));
}

template<typename scalar_t, size_t N>
C10_HOST auto accessor(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
    #define DEFINE_SWITCH(_TYPE, _VALUE) \
        case torch::ScalarType::_VALUE: \
        return accessor<scalar_t, N, _TYPE>(tensor); \

    AT_FORALL_SCALAR_TYPES(DEFINE_SWITCH)
    #undef DEFINE_SWITCH
    default:
        assert(false);
        return accessor<scalar_t, N, uint8_t>(tensor);
    }
}

template<typename struct_t, size_t N, typename tensor_type>
C10_HOST auto struct_accessor(const torch::Tensor& tensor) {
    assert(sizeof(struct_t) % sizeof(tensor_type) == 0);
    assert(tensor.dtype().itemsize() == sizeof(tensor_type));
    assert(tensor.dim() == N + 1);
    assert(tensor.size(-1) * sizeof(tensor_type) == sizeof(struct_t));          // tensor.size(-1) == sizeof(struct_t) / sizeof(tensor_type), but without rounding
    auto torch_accessor = tensor.generic_packed_accessor<tensor_type, N + 1, gpe::RestrictPtrTraits, uint32_t>();
    assert(torch_accessor.stride(N) == 1);
    assert(torch_accessor.size(N) * sizeof(tensor_type) == sizeof(struct_t));   // torch_accessor.size(N) == sizeof(struct_t) / sizeof(tensor_type), but without rounding
    using index_t = decltype(torch_accessor.size(0));

    std::array<index_t, N> strides;
    std::array<index_t, N> sizes;
    for (unsigned i = 0; i < N; ++i) {
        strides[i] = torch_accessor.stride(i) / (sizeof(struct_t) / sizeof(tensor_type));
        sizes[i] = torch_accessor.size(i);
    }

    return PackedTensorAccessor32<struct_t, N, gpe::RestrictPtrTraits>(reinterpret_cast<struct_t*>(torch_accessor.data()), sizes.data(), strides.data());
}

template<typename struct_t, size_t N>
C10_HOST auto struct_accessor(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
    #define DEFINE_SWITCH(_TYPE, _VALUE) \
        case torch::ScalarType::_VALUE: \
        return struct_accessor<struct_t, N, _TYPE>(tensor); \

    AT_FORALL_SCALAR_TYPES(DEFINE_SWITCH)
    #undef DEFINE_SWITCH
    default:
        assert(false);
        return struct_accessor<struct_t, N, uint8_t>(tensor);
    }
}

template<typename data_t, size_t N, typename index_t = uint32_t>
C10_HOST auto accessor(const std::vector<data_t>& vector, const std::array<index_t, N>& sizes) {
    std::array<index_t, N> strides;
    index_t numel = 1;
    for (unsigned i = N-1; i < N; --i) {
        strides[i] = numel;
        numel *= sizes[i];
    }
    assert(numel == vector.size());

    return GenericPackedTensorAccessor<data_t, N, gpe::RestrictPtrTraits, index_t>(vector.data(), sizes.data(), strides.data());
}

template<typename data_t, size_t N, typename index_t = uint32_t>
C10_HOST auto accessor(std::vector<data_t>& vector, const std::array<index_t, N>& sizes) {
    std::array<index_t, N> strides;
    index_t numel = 1;
    for (unsigned i = N-1; i < N; --i) {
        strides[i] = numel;
        numel *= sizes[i];
    }
    assert(numel == vector.size());

    return GenericPackedTensorAccessor<data_t, N, gpe::RestrictPtrTraits, index_t>(vector.data(), sizes.data(), strides.data());
}

template<typename T, size_t N>
using Accessor32 = TensorAccessor<T, N, RestrictPtrTraits, uint32_t>;

//template<typename scalar_t, size_t N>
//auto accessor(torch::Tensor& tensor) {
//    auto torch_accessor = tensor.packed_accessor32<scalar_t, N, gpe::RestrictPtrTraits>();
//    return PackedTensorAccessor32<scalar_t, N, gpe::RestrictPtrTraits>(*reinterpret_cast<PackedTensorAccessor32<scalar_t, N, gpe::RestrictPtrTraits>*>(&torch_accessor));
//}

}
#endif // HACKED_ACCESSOR_H
