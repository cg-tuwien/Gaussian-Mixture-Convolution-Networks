//The MIT License (MIT)

//Copyright (c) 2019 Toru Niina

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.

//(based on https://github.com/ToruNiina/lbvh)

#ifndef LBVH_MORTON_CODE_H
#define LBVH_MORTON_CODE_H
#include <vector_types.h>
#include <cstdint>

#include <omp.h>

#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <torch/types.h>

#include "cuda_operations.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "common.h"

namespace lbvh
{

template<typename morton_cuda_t, typename morton_torch_t>
torch::Tensor sort_morton_codes(const torch::Tensor& morton_codes) {
    const int num_segments = [&](){ int n_segments = 1; for (unsigned i = 0; i < morton_codes.dim() - 1; ++i) n_segments *= morton_codes.sizes()[i]; return n_segments;}();                    // e.g., 4
    const int num_components = morton_codes.size(-1);                          // 2
    auto sorted_morton_codes = morton_codes.clone();
    const morton_cuda_t* d_keys_in = reinterpret_cast<const morton_cuda_t*>(morton_codes.data_ptr<morton_torch_t>());   // e.g., [8, 6, 7, 5, 3, 0, 9, 8]
    morton_cuda_t* d_keys_out = reinterpret_cast<morton_cuda_t*>(sorted_morton_codes.data_ptr<morton_torch_t>());       // e.g., [-, -, -, -, -, -, -, -]

    if (morton_codes.is_cuda()) {
        // Declare, allocate, and initialize device-accessible pointers for sorting data
        int num_items = int(morton_codes.numel());                         // e.g., 8
        const torch::Tensor offsets = torch::arange(0, num_segments + 1, torch::TensorOptions(morton_codes.device()).dtype(torch::ScalarType::Int)) * num_components;
        //        std::cout << "offsets: " << offsets << std::endl;
        int* d_offsets = offsets.data_ptr<int>();                           // e.g., [0, 2, 4, 6, 8]

        // Determine temporary device storage requirements
        void     *d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;

#ifndef GPE_MORTON_OLD_WAY
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            num_items, num_segments,
            d_offsets, d_offsets + 1, 32, 63);
#else
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            num_items, num_segments,
            d_offsets, d_offsets + 1);
#endif

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());
        // Run sorting operation

#ifndef GPE_MORTON_OLD_WAY
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            num_items, num_segments,
            d_offsets, d_offsets + 1, 32, 63);
#else
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            num_items, num_segments,
            d_offsets, d_offsets + 1);
#endif

        // d_keys_out            <-- [6, 8, 5, 7, 0, 3, 8, 9]
        // d_values_out          <-- [1, 0, 3, 2, 5, 4, 7, 6]

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());

        cudaFree(d_temp_storage);
    }
    else {
        // this is most likely not the fastest possible solution, but quick to implement and only cpu code and not perf bottleneck to my knowledge (test if in doubt!)
        #pragma omp parallel for num_threads(omp_get_num_procs())
        for (int i = 0; i < num_segments; ++i) {
            std::sort(d_keys_out + i*num_components, d_keys_out + (i + 1) * num_components);
        }
    }

    return sorted_morton_codes;
}

EXECUTION_DEVICES
constexpr std::uint32_t expand_bits3(std::uint32_t v) noexcept
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

template <typename T>
EXECUTION_DEVICES
constexpr T copy_or(T v, unsigned shift) noexcept {
    return (v << shift) | v;
}

EXECUTION_DEVICES
constexpr std::uint64_t expand_bits2(std::uint32_t v) noexcept
{
    uint64_t t = v;
    t = copy_or(t, 16) & 0x0000'FFFF'0000'FFFFul;
    t = copy_or(t, 8)  & 0x00FF'00FF'00FF'00FFul;
    t = copy_or(t, 4)  & 0x0F0F'0F0F'0F0F'0F0Ful;
    t = copy_or(t, 2)  & 0x3333'3333'3333'3333ul;
    t = copy_or(t, 1)  & 0x5555'5555'5555'5555ul;
    return t;
}
static_assert (expand_bits2(0xffff'fffful) == 0x5555'5555'5555'5555ull, "all bits should be set");
static_assert (expand_bits2(0x0ul) == 0x0ull, "no bits should be set");
static_assert (expand_bits2(0x2c6f'00d9ul) == 0x0450'1455'0000'5141ull, "test failed");

//EXECUTION_DEVICES
//constexpr std::uint32_t expand_bits2(std::uint16_t v) noexcept
//{
//    uint32_t t = v;
//    t = copy_or(t, 8)  & 0x00FF'00FFu;
//    t = copy_or(t, 4)  & 0x0F0F'0F0Fu;
//    t = copy_or(t, 2)  & 0x3333'3333u;
//    t = copy_or(t, 1)  & 0x5555'5555u;
//    return t;
//}

// Calculates a 30-bit Morton code for the
// given 3D point located within the unit cube [0,1].
EXECUTION_DEVICES
std::uint32_t morton_code(float4 xyz, float resolution = 1024.0f) noexcept
{
    xyz.x = gpe::min(gpe::max(xyz.x * resolution, 0.0f), resolution - 1.0f);
    xyz.y = gpe::min(gpe::max(xyz.y * resolution, 0.0f), resolution - 1.0f);
    xyz.z = gpe::min(gpe::max(xyz.z * resolution, 0.0f), resolution - 1.0f);
    const std::uint32_t xx = expand_bits3(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = expand_bits3(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = expand_bits3(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

EXECUTION_DEVICES
std::uint32_t morton_code(double4 xyz, double resolution = 1024.0) noexcept
{
    xyz.x = gpe::min(gpe::max(xyz.x * resolution, 0.0), resolution - 1.0);
    xyz.y = gpe::min(gpe::max(xyz.y * resolution, 0.0), resolution - 1.0);
    xyz.z = gpe::min(gpe::max(xyz.z * resolution, 0.0), resolution - 1.0);
    const std::uint32_t xx = expand_bits3(static_cast<std::uint32_t>(xyz.x));
    const std::uint32_t yy = expand_bits3(static_cast<std::uint32_t>(xyz.y));
    const std::uint32_t zz = expand_bits3(static_cast<std::uint32_t>(xyz.z));
    return xx * 4 + yy * 2 + zz;
}

template<typename scalar_t>
EXECUTION_DEVICES
// 2 left most bits are 0
std::uint32_t morton_code(const glm::vec<3, scalar_t>& vec, uint32_t resolution = 1024) noexcept {
    uint32_t x = gpe::min(gpe::max(uint32_t(vec.x * resolution), 0u), resolution - 1);
    uint32_t y = gpe::min(gpe::max(uint32_t(vec.y * resolution), 0u), resolution - 1);
    uint32_t z = gpe::min(gpe::max(uint32_t(vec.z * resolution), 0u), resolution - 1);
    auto xx = expand_bits3(x);
    auto yy = expand_bits3(y);
    auto zz = expand_bits3(z);
    return (xx << 2) + (yy << 1) + zz;
}

template<typename scalar_t>
EXECUTION_DEVICES
// 2 left most bits are 0
// todo: could be optimised, expand only 2..
std::uint32_t morton_code(const glm::vec<2, scalar_t>& vec, uint32_t resolution = 1024) noexcept {
    uint32_t x = gpe::min(gpe::max(uint32_t(vec.x * resolution), 0u), resolution - 1);
    uint32_t y = gpe::min(gpe::max(uint32_t(vec.y * resolution), 0u), resolution - 1);
    auto xx = expand_bits3(x);
    auto yy = expand_bits3(y);
    return (xx << 2) + (yy << 1);
}

EXECUTION_DEVICES
constexpr std::uint64_t mix_Cov1_12p36pc16i(uint32_t morton_pos, uint32_t morton_cov, uint16_t component_id) noexcept {
    assert((morton_pos & 0x3fff'fffful) == morton_pos);
    assert((morton_pos & 0x3fff'fffful) == morton_pos);
    // layout:
    // we use 30 bits of position morton code, 18 bits for cov_diag morton code and 16 bits for component_id
    // in the result:
    // 12 bits only position, 18 + 18 bits interleaved position with cov_diag, and 16 bits component id
    uint64_t result = 0;
    result |= morton_pos >> 18;
    morton_pos = morton_pos & 0x3'FFFFu;               // 18 bits
    morton_cov = morton_cov >> 12;                  // 18 most significant bits
    assert((morton_cov & 0x0003'FFFFul) == morton_cov);
    uint64_t morton_pos_expanded = expand_bits2(morton_pos);
    uint64_t morton_cov_expanded = expand_bits2(morton_cov);
    uint64_t morton_poscov = (morton_pos_expanded << 1) | morton_cov_expanded;    // 36 bits
    assert((morton_poscov & 0x0000'000F'FFFF'FFFFull) == morton_poscov);
    result = (result << 36) | morton_poscov;
    assert((result & 0x0000'FFFF'FFFF'FFFFull) == result);
    result = (result << 16) | component_id;
    return result;
}

EXECUTION_DEVICES
constexpr std::uint64_t mix_Cov2_54pc10i(uint32_t morton_pos, uint32_t morton_cov, uint16_t component_id) noexcept {
    assert((morton_pos & 0x3fff'fffful) == morton_pos);
    assert((morton_pos & 0x3fff'fffful) == morton_pos);
    // layout:
    // here we use 27 bits of position morton code, 27 bits for cov_diag morton code and 10 bits for component_id
    // in the result:
    // 27 + 27 bits interleaved position with cov_diag, and 10 bits component id
    uint64_t morton_pos_expanded = expand_bits2(morton_pos);
    uint64_t morton_cov_expanded = expand_bits2(morton_cov);
    auto morton_poscov = (morton_pos_expanded << 1) | morton_cov_expanded;    // 60 bits
    uint64_t result = morton_poscov << 4;
    result = result & 0xFFFF'FFFF'FFFF'FC00ull;
    static_assert(0xFFFF'FFFF'FFFF'FC00u == ~0x3FFull);
    assert (component_id < 1024);
    result = result | component_id;
    return result;
}

EXECUTION_DEVICES
constexpr std::uint64_t mix_Cov3_27p27c10i(uint32_t morton_pos, uint32_t morton_cov, uint16_t component_id) noexcept {
    assert((morton_pos & 0x3fff'fffful) == morton_pos);
    assert((morton_pos & 0x3fff'fffful) == morton_pos);
    // layout:
    // here we use 27 bits of position morton code, 27 bits for cov_diag morton code and 10 bits for component_id
    // in the result:
    // 27 bits position, then 27 bits cov_diag, and 10 bits component id
    uint64_t result = uint64_t(morton_pos & (~0x7u)) << (64-30);
    assert((result & 0xFFFF'FFE0'0000'0000ull) == result);
    assert((result >> (64 - 27)) == (morton_pos >> 3));

    result |= uint64_t(morton_cov & (~0x7u)) << (64-27-30);
    assert((result & 0xFFFF'FFFF'FFFF'FC00u) == result);
    static_assert(0xFFFF'FFFF'FFFF'FC00u == ~0x3FFull);
    assert((((result << 27) >> 27) >> (64 - 27 * 2)) == (morton_cov >> 3));
    assert (component_id < 1024);
    result = result | (component_id & 0x3FF);
    return result;
}

EXECUTION_DEVICES
constexpr std::uint64_t mix_Cov4_27c27p10i(uint32_t morton_pos, uint32_t morton_cov, uint16_t component_id) noexcept {
    assert((morton_pos & 0x3fff'fffful) == morton_pos);
    assert((morton_pos & 0x3fff'fffful) == morton_pos);
    // layout:
    // here we use 27 bits of position morton code, 27 bits for cov_diag morton code and 10 bits for component_id
    // in the result:
    // 27 bits position, then 27 bits cov_diag, and 10 bits component id
    uint64_t result = uint64_t(morton_cov & (~0x7u)) << (64-30);
    assert((result & 0xFFFF'FFE0'0000'0000ull) == result);
    assert((result >> (64 - 27)) == (morton_cov >> 3));

    result |= uint64_t(morton_pos & (~0x7u)) << (64-27-30);
    assert((result & 0xFFFF'FFFF'FFFF'FC00u) == result);
    static_assert(0xFFFF'FFFF'FFFF'FC00u == ~0x3FFull);
    assert((((result << 27) >> 27) >> (64 - 27 * 2)) == (morton_pos >> 3));
    assert (component_id < 1024);
    result = result | (component_id & 0x3FF);
    return result;
}

template<int MORTON_CODE_ALGORITHM, typename scalar_t>
EXECUTION_DEVICES
std::uint64_t morton_code(uint16_t component_id, const glm::vec<3, scalar_t>& pos, const glm::vec<3, scalar_t>& cov_diag, scalar_t resolution = 1024.0) noexcept {
    if (MORTON_CODE_ALGORITHM == 0) {
        /// old
        uint32_t morton_pos = morton_code(pos, resolution);         // 30 bits
        uint64_t result = morton_pos;
        result = (result << 16) | component_id;
        return result;
    }

    uint32_t morton_pos = morton_code(pos, resolution);         // 30 bits
    uint32_t morton_cov = morton_code(cov_diag, resolution);    // 30 bits
    if (MORTON_CODE_ALGORITHM == 1) {
        static_assert (mix_Cov1_12p36pc16i(0x3fff'fffful, 0x3fff'fffful, uint16_t(~0)) == ~0ull, "didn't use all bits");
        static_assert (mix_Cov1_12p36pc16i(0x0000'0000ul, 0x0000'0000ul, uint16_t(0)) == 0ull, "some bits unjustly set");
        return mix_Cov1_12p36pc16i(morton_pos, morton_cov, component_id);
    }
    if (MORTON_CODE_ALGORITHM == 2) {
        /// 2. cov 2
        static_assert (mix_Cov2_54pc10i(0x3fff'fffful, 0x3fff'fffful, uint16_t(0x3ff)) == ~0ull, "didn't use all bits");
        static_assert (mix_Cov2_54pc10i(0x0000'0000ul, 0x0000'0000ul, uint16_t(0)) == 0ull, "some bits unjustly set");
        return mix_Cov2_54pc10i(morton_pos, morton_cov, component_id);
    }
    else if (MORTON_CODE_ALGORITHM == 3) {
        static_assert (mix_Cov3_27p27c10i(0x3fff'fffful, 0x3fff'fffful, uint16_t(0x3ff)) == ~0ull, "didn't use all bits");
        static_assert (mix_Cov3_27p27c10i(0x0000'0000ul, 0x0000'0000ul, uint16_t(0)) == 0ull, "some bits unjustly set");
        return mix_Cov3_27p27c10i(morton_pos, morton_cov, component_id);
    }
    else if (MORTON_CODE_ALGORITHM == 4) {
        static_assert (mix_Cov4_27c27p10i(0x3fff'fffful, 0x3fff'fffful, uint16_t(0x3ff)) == ~0ull, "didn't use all bits");
        static_assert (mix_Cov4_27c27p10i(0x0000'0000ul, 0x0000'0000ul, uint16_t(0)) == 0ull, "some bits unjustly set");
        return mix_Cov4_27c27p10i(morton_pos, morton_cov, component_id);
    }
    else {
        assert(false);
        return 0;
    }
}

} // lbvh
#endif// LBVH_MORTON_CODE_H
