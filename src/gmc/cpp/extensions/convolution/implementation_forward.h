#include "convolution/implementation.h"
#include <stdio.h>
#include <type_traits>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
#include "hacked_accessor.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "parallel_start.h"
#include "util/algorithms.h"
#include "util/autodiff.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/helper.h"
#include "util/mixture.h"


namespace convolution {



namespace  {

template <typename scalar_t, int N_DIMS>
__host__ __device__
gpe::Gaussian<N_DIMS, scalar_t> convolve(const gpe::Gaussian<N_DIMS, scalar_t>& g1, const gpe::Gaussian<N_DIMS, scalar_t>& g2) {
    constexpr auto a = gcem::pow(scalar_t(2) * glm::pi<scalar_t>(), N_DIMS * scalar_t(0.5));
    const auto b = gpe::sqrt(glm::determinant(g1.covariance) * glm::determinant(g2.covariance));
    gpe::Gaussian<N_DIMS, scalar_t> ret {g1.weight * g2.weight * a * b, g1.position + g2.position, g1.covariance + g2.covariance};
    ret.weight /= gpe::sqrt(glm::determinant(ret.covariance));
    return ret;
}


} // anonymous namespace


template<typename scalar_t, unsigned N_DIMS>
ForwardOutput forward_impl_t(const torch::Tensor& data, const torch::Tensor& kernels) {
    using namespace torch::indexing;

    const auto n = gpe::get_ns(data);
    const auto kernel_n = gpe::get_ns(kernels);
    const auto n_channels_in = n.layers;
    const auto n_channels_out = kernel_n.batch;
    const auto n_target_components = unsigned(n.components * n_channels_in * kernel_n.components);
    TORCH_CHECK(n.batch * n_channels_out < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(kernel_n.components >= 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n_channels_in == kernel_n.layers, "number of input feature maps must agree with the second kernel dimension")
    TORCH_CHECK(n.dims == kernel_n.dims, "number of dimensions of data and kernel must agree")
    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
    TORCH_CHECK(data.dtype() == kernels.dtype(), "kernel and data dtypes must agree")
    TORCH_CHECK(data.dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")
    TORCH_CHECK(data.device() == kernels.device(), "data and kernel devices must agree")

    const auto data_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(data);
    const auto kernel_a = gpe::struct_accessor<typename gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(kernels);

    auto out_mixture = torch::empty({n.batch, n_channels_out, n_target_components, data.size(-1)}, torch::TensorOptions(data.device()).dtype(data.dtype()));
    auto out_mixture_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 3, scalar_t>(out_mixture);

    std::cout << "n_target_components: " << n_target_components << std::endl;
    std::cout << "n.batch: " << n.batch << std::endl;
    std::cout << "n_channels_out: " << n_channels_out << std::endl;
    std::cout << "n_channels_in: " << n_channels_in << std::endl;
    std::cout << "kernel_n.components: " << kernel_n.components << std::endl;
    std::cout << "n.components: " << n.components << std::endl;


    dim3 dimBlock = dim3(256, 1, 1);
    dim3 dimGrid = dim3((unsigned(n_target_components) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(n.batch) + dimBlock.y - 1) / dimBlock.y,
                        (unsigned(n_channels_out) + dimBlock.z - 1) / dimBlock.z);
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(data), dimGrid, dimBlock, [data_a, kernel_a, out_mixture_a, n_channels_in, n_channels_out, kernel_n, n, n_target_components] __host__ __device__
                                                  (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {

        // index might not fit into 32 bit, i.e. when n.components == 1 << 17, n_feature_maps_in == 1 << 12 and kernel_n.components == 1 << 4
        // however, such large datasets would be infeasable anyways. i.e., if we have (1<<32) output components, then the morton code array alone takes 8 GB. For one output feature map. For one batch dimension.
        // Sorting alone would probably take too long.
        assert(uint64_t(gpe_blockIdx.x) * uint64_t(gpe_blockDim.x) + uint64_t(gpe_threadIdx.x) < (1ull << 32));
        const unsigned component_out_id = gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x;
        if (component_out_id >= n_target_components)
            return;

        const unsigned batch_id = gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y;
        const unsigned channel_out_id = gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z;

        const auto gaussian_indices = gpe::split_n_dim_index<uint32_t, 3, unsigned>({unsigned(n.components), unsigned(n_channels_in), unsigned(kernel_n.components)}, component_out_id);
        const unsigned& component_in_id = gaussian_indices[0];
        const unsigned& channel_in_id = gaussian_indices[1];
        const unsigned& component_kernel_id = gaussian_indices[2];

        assert(batch_id < n.batch);
        assert(channel_in_id < n_channels_in);
        assert(channel_out_id < n_channels_out);
        assert(component_in_id < n.components);
        assert(component_out_id < n_target_components);
        assert(component_kernel_id < kernel_n.components);

        const auto& data_gaussian = data_a[batch_id][channel_in_id][component_in_id];
        const auto& kernel_gaussian = kernel_a[channel_out_id][channel_in_id][component_kernel_id];

        out_mixture_a[int(batch_id)][int(channel_out_id)][int(component_out_id)] = convolve(data_gaussian, kernel_gaussian);

    });

    return ForwardOutput{out_mixture};
}


} // namespace bvh_mhem_fit

