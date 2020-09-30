#include "cuda.h"
#include <algorithm>
#include <chrono>
#include <vector>
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <torch/script.h>

#include <glm/glm.hpp>

#include "common.h"
#include "mixture.h"
#include "cuda_qt_creator_definitinos.h"
#include "lbvh/aabb.h"
#include "lbvh/bvh.h"
#include "lbvh/query.h"
#include "lbvh/predicator.h"
#include "math/symeig_cuda.h"


#include "bvh_mhem_fit/implementation.h"
#include "parallel_start.h"

namespace bvh_mhem_fit {

namespace  {

template<int N_DIMS, typename scalar_t>
std::ostream& operator <<(std::ostream& stream, const Gaussian<N_DIMS, scalar_t>& g) {
    stream << "Gauss[" << g.weight << "; " << g.position[0];
    for (int i = 1; i < N_DIMS; i++)
        stream << "/" << g.position[i];
    stream << "; ";

    for (int i = 0; i < N_DIMS; i++) {
        for (int j = 0; j < N_DIMS; j++) {
            if (i != 0 || j != 0)
                stream << "/";
            stream << g.covariance[i][j];
        }
    }
    stream << "]";
    return stream;
}

template <typename scalar_t, int DIMS, template <typename U> class PtrTraits = gpe::RestrictPtrTraits>
__host__ __device__
void evaluate_bvh_forward(const dim3& gpe_gridDim, const dim3& gpe_blockDim,
                         const dim3& gpe_blockIdx, const dim3& gpe_threadIdx,
                         const torch::PackedTensorAccessor32<scalar_t, 4, PtrTraits> mixture,
                         const torch::PackedTensorAccessor32<lbvh::detail::Node::index_type_torch, 4, PtrTraits> nodes,
                         const torch::PackedTensorAccessor32<scalar_t, 4, PtrTraits> aabbs,
                         const torch::PackedTensorAccessor32<scalar_t, 4, PtrTraits> xes,
                         torch::PackedTensorAccessor32<scalar_t, 3, PtrTraits> sums,
                         const gpe::MixtureAndXesNs n)
{
    GPE_UNUSED(gpe_gridDim)
    using G = Gaussian<DIMS, scalar_t>;
    using Lbvh = lbvh::detail::basic_device_bvh<scalar_t, G, true>;
    const int batch_index = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
    const int layer_index = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
    const int xes_index = int(gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z);

    const auto batch_xes_index = min(batch_index, int(n.batch_xes - 1));
    const auto layer_xes_index = min(layer_index, int(n.layers_xes - 1));

    //    printf("batch_index=%d, layer_index=%d, batch_xes_index=%d, layer_xes_index=%d, xes_index=%d\n", batch_index, layer_index, batch_xes_index, layer_xes_index, xes_index);
    if (batch_index >= int(n.batch) || layer_index >= int(n.layers) || xes_index >= int(n.xes))
        return;
    //    printf("do batch_index=%d, layer_index=%d, batch_xes_index=%d, layer_xes_index=%d, xes_index=%d\n", batch_index, layer_index, batch_xes_index, layer_xes_index, xes_index);


    const unsigned int num_nodes = n.components * 2 + 1;  // (# of internal node) + (# of leaves), 2N+1
    const unsigned int num_objects = n.components;        // (# of leaves), the same as the number of objects
    const auto* bvh_nodes = &reinterpret_cast<const lbvh::detail::Node&>(nodes[batch_index][layer_index][0][0]);
    const auto* bvh_aabbs = &reinterpret_cast<const lbvh::Aabb<scalar_t>&>(aabbs[batch_index][layer_index][0][0]);
    const auto* bvh_gaussians = &reinterpret_cast<const G&>(mixture[batch_index][layer_index][0][0]);
    Lbvh bvh {num_nodes, num_objects, bvh_nodes, bvh_aabbs, bvh_gaussians};

    const auto& x_pos = gpe::vec<DIMS>(xes[batch_xes_index][layer_xes_index][xes_index][0]);
    auto point = lbvh::make_vector_of(x_pos);
    auto& sum = sums[batch_index][layer_index][xes_index];
    auto evaluate = [bvh, &sum, &x_pos] (unsigned index) {
        const auto& g = bvh.objects[index];
        sum += gpe::evaluate_gaussian(x_pos, g.weight, g.position, g.covariance);
    };
    lbvh::query_device_with_fun(bvh, lbvh::inside_aabb(point), evaluate);
}

template <typename scalar_t, int DIMS, template <typename U> class PtrTraits = gpe::RestrictPtrTraits>
__host__ __device__
void kernel_bvh_backward(const dim3& gpe_blockIdx, const dim3& gpe_blockDim,
                        const dim3& gpe_threadIdx, const dim3& gpe_threadDim,
                        const torch::PackedTensorAccessor32<scalar_t, 4, PtrTraits> mixture,
                        const torch::PackedTensorAccessor32<lbvh::detail::Node::index_type_torch, 4, PtrTraits> nodes,
                        const torch::PackedTensorAccessor32<scalar_t, 4, PtrTraits> aabbs,
                        const torch::PackedTensorAccessor32<scalar_t, 4, PtrTraits> xes,
                        torch::PackedTensorAccessor32<scalar_t, 4, PtrTraits> grad_mixture,
                        torch::PackedTensorAccessor32<scalar_t, 4, PtrTraits> grad_xes,
                        const torch::PackedTensorAccessor32<scalar_t, 3, PtrTraits> grad_output,
                        const gpe::MixtureAndXesNs n, bool requires_grad_mixture, bool requires_grad_xes)
{
    GPE_UNUSED(gpe_threadDim)
    using G = Gaussian<DIMS, scalar_t>;
    using Lbvh = lbvh::detail::basic_device_bvh<scalar_t, G, true>;
    const int batch_index = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
    const int layer_index = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
    const int xes_index = int(gpe_blockIdx.z * gpe_blockDim.z + gpe_threadIdx.z);

    const auto batch_xes_index = min(batch_index, int(n.batch_xes - 1));
    const auto layer_xes_index = min(layer_index, int(n.layers_xes - 1));

    if (batch_index >= int(n.batch) || layer_index >= int(n.layers) || xes_index >= int(n.xes))
        return;
    printf("do batch_index=%d, layer_index=%d, batch_xes_index=%d, layer_xes_index=%d, xes_index=%d\n", batch_index, layer_index, batch_xes_index, layer_xes_index, xes_index);

    const unsigned int num_nodes = n.components * 2 + 1;  // (# of internal node) + (# of leaves), 2N+1
    const unsigned int num_objects = n.components;        // (# of leaves), the same as the number of objects
    const auto* bvh_nodes = &reinterpret_cast<const lbvh::detail::Node&>(nodes[batch_index][layer_index][0][0]);
    const auto* bvh_aabbs = &reinterpret_cast<const lbvh::Aabb<scalar_t>&>(aabbs[batch_index][layer_index][0][0]);
    const auto* bvh_gaussians = &reinterpret_cast<const G&>(mixture[batch_index][layer_index][0][0]);
    Lbvh bvh {num_nodes, num_objects, bvh_nodes, bvh_aabbs, bvh_gaussians};

    const auto& x_pos = gpe::vec<DIMS>(xes[batch_xes_index][layer_xes_index][xes_index][0]);
    auto point = lbvh::make_vector_of<scalar_t>(x_pos);

    auto current_grad_mixture = grad_mixture[batch_index][layer_index];
    auto current_grad_xes = grad_xes[batch_xes_index][layer_xes_index][xes_index];
    const auto current_grad_output = grad_output[batch_index][layer_index][xes_index];

    auto evaluate_backward = [&] (int index) {
        const G& g = bvh.objects[index];

        const auto t = x_pos - g.position;
        const auto v = scalar_t(-0.5) * glm::dot(t, (g.covariance * t));
        const auto exp = gpe::exp(v);
        const auto weighted_exp = g.weight * exp;
        const auto local_grad_c_pos = weighted_exp * t * g.covariance;

        if (requires_grad_xes) {
            const auto grad_xes_addition = - current_grad_output * local_grad_c_pos;
            for (int i = 0; i < DIMS; ++i) {
                gpe::atomicAdd(&current_grad_xes[i], grad_xes_addition[i]);
            }
        }
        if (requires_grad_mixture) {
            const auto grad_c_weight_addition = exp * current_grad_output;
            const auto grad_c_pos_addition = local_grad_c_pos * current_grad_output;
            const auto grad_c_cov_addition = - g.weight * scalar_t(0.5) * exp * current_grad_output * glm::outerProduct(t, t);
            gpe::atomicAdd(&current_grad_mixture[index][0], grad_c_weight_addition);
            for (int i = 0; i < DIMS; ++i) {
                gpe::atomicAdd(&current_grad_mixture[index][1 + i], grad_c_pos_addition[i]);
                for (int j = 0; j < DIMS; ++j)
                    gpe::atomicAdd(&current_grad_mixture[index][1 + DIMS + i*DIMS + j], grad_c_cov_addition[i][j]);
            }
        }

    };
    lbvh::query_device_with_fun(bvh, lbvh::inside_aabb(point), evaluate_backward);
}


torch::Tensor inverse_permutation(const torch::Tensor& p) {
    auto l = torch::arange(p.size(-1), torch::TensorOptions(p.device()).dtype(p.dtype()));
    auto shape = p.sizes().vec();
    assert(shape.size() > 0);
    std::for_each(shape.begin(), shape.end() - 1, [](auto& i) { i = 1; });
    l = l.view(shape).expand_as(p);
    return torch::scatter(torch::empty_like(p), -1, p, l);
}

} // anonymous namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cuda_bvh_forward_impl(const at::Tensor& mixture, const at::Tensor& xes) {
    using namespace torch::indexing;
    using LBVH = lbvh::bvh<float, Gaussian<2, float>>;

    auto n = gpe::check_input_and_get_ns(mixture, xes);
    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor")
    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.xes < 65535, "number of xes must be smaller than 65535 for CUDA")
    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
    TORCH_CHECK(n.dims == 2, "atm only 2d gaussians")
    TORCH_CHECK(mixture.dtype() == caffe2::TypeMeta::Make<float>(), "atm only float")

    auto bvh = LBVH(mixture);
    torch::Tensor sum = torch::zeros({n.batch, n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));

    // mixture(batch, layer, component, data)
    // xes(batch, layer, n, data)

    auto xes_copy = xes;
    const auto use_indirect_xes = n.xes == n.components && n.batch == n.batch_xes && n.layers == n.layers_xes;
    if (use_indirect_xes) {
        auto indices = bvh.m_nodes.index({Slice(), Slice(), Slice(bvh.m_n_internal_nodes, None), 3}).to(torch::ScalarType::Long);
        indices = indices.view({n.batch, n.layers, n.components, 1}).expand_as(xes);
        xes_copy = torch::gather(xes, 2, indices);
    }

    dim3 dimBlock = dim3(1, 1, LBVH_N_QUERY_THREADS);
    dim3 dimGrid = dim3((n.batch + dimBlock.x - 1) / dimBlock.x,
                        (n.layers + dimBlock.y - 1) / dimBlock.y,
                        (n.xes + dimBlock.z - 1) / dimBlock.z);
    //    printf("dimBlock=(%d, %d, %d)\n", dimBlock.x, dimBlock.y, dimBlock.z);
    //    printf("dimGrid=(%d, %d, %d)\n", dimGrid.x, dimGrid.y, dimGrid.z);


    //    auto start = std::chrono::high_resolution_clock::now();


    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_bvh_backward_impl", ([&] {
                                   auto sum_a = sum.packed_accessor32<scalar_t, 3, gpe::RestrictPtrTraits>();
                                   auto mixture_a = mixture.packed_accessor32<scalar_t, 4, gpe::RestrictPtrTraits>();
                                   auto nodes_a = bvh.m_nodes.packed_accessor32<lbvh::detail::Node::index_type_torch, 4, gpe::RestrictPtrTraits>();
                                   auto aabbs_a = bvh.m_aabbs.packed_accessor32<scalar_t, 4, gpe::RestrictPtrTraits>();
                                   const auto xes_a = xes_copy.packed_accessor32<scalar_t, 4, gpe::RestrictPtrTraits>();

                                   if (n.dims == 2) {
                                       auto fun = [mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               evaluate_bvh_forward<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               evaluate_bvh_forward<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx, mixture_a, nodes_a, aabbs_a, xes_a, sum_a, n);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                               }));

    //    cudaDeviceSynchronize();
    //    auto end = std::chrono::high_resolution_clock::now();
    //    std::cout << "bvh eval elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";

    if (use_indirect_xes) {
        auto indices = bvh.m_nodes.index({Slice(), Slice(), Slice(bvh.m_n_internal_nodes, None), 3}).to(torch::ScalarType::Long);
        indices = inverse_permutation(indices);
        sum = torch::gather(sum, 2, indices);
    }

    GPE_CUDA_ASSERT(cudaPeekAtLastError())
    GPE_CUDA_ASSERT(cudaDeviceSynchronize())

    return std::make_tuple(sum, bvh.m_nodes, bvh.m_aabbs);
}

std::tuple<torch::Tensor, torch::Tensor> cuda_bvh_backward_impl(const torch::Tensor& grad_output,
                                                                const torch::Tensor& mixture, const torch::Tensor& bvh_nodes, const torch::Tensor& aabbs,
                                                                const torch::Tensor& xes,
                                                                bool requires_grad_mixture, bool requires_grad_xes) {
    using namespace torch::indexing;
    using LBVH = lbvh::bvh<float, Gaussian<2, float>>;
    gpe::check_mixture(mixture);
    auto n = gpe::check_input_and_get_ns(mixture, xes);

    TORCH_CHECK(mixture.device().is_cuda(), "mixture must be a CUDA tensor")
    TORCH_CHECK(grad_output.device().is_cuda(), "grad_output must be a CUDA tensor")
    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions")
    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension")
    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension")
    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension")
    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")

    auto bvh = LBVH(mixture, bvh_nodes, aabbs);
    torch::Tensor grad_mixture = torch::zeros_like(mixture);
    torch::Tensor grad_xes = torch::zeros_like(xes);

    dim3 dimBlock = dim3(1, 1, LBVH_N_QUERY_THREADS);
    dim3 dimGrid = dim3((n.batch + dimBlock.x - 1) / dimBlock.x,
                        (n.layers + dimBlock.y - 1) / dimBlock.y,
                        (n.xes + dimBlock.z - 1) / dimBlock.z);

    auto xes_copy = xes;
    auto grad_output_copy = grad_output;
    const auto use_indirect_xes = n.xes == n.components && n.batch == n.batch_xes && n.layers == n.layers_xes;
    if (use_indirect_xes) {
        auto indices = bvh.m_nodes.index({Slice(), Slice(), Slice(bvh.m_n_internal_nodes, None), 3}).to(torch::ScalarType::Long);
        xes_copy = torch::gather(xes, 2, indices.view({n.batch_xes, n.layers_xes, n.xes, 1}).expand_as(xes));
        grad_output_copy = torch::gather(grad_output, 2, indices);
    }


    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "cuda_bvh_backward_impl", ([&] {
                                   auto mixture_a = mixture.packed_accessor32<scalar_t, 4, gpe::RestrictPtrTraits>();
                                   auto nodes_a = bvh.m_nodes.packed_accessor32<lbvh::detail::Node::index_type_torch, 4, gpe::RestrictPtrTraits>();
                                   auto aabbs_a = bvh.m_aabbs.packed_accessor32<scalar_t, 4, gpe::RestrictPtrTraits>();
                                   auto xes_a = xes_copy.packed_accessor32<scalar_t, 4, gpe::RestrictPtrTraits>();
                                   auto grad_mixture_a = grad_mixture.packed_accessor32<scalar_t, 4, gpe::RestrictPtrTraits>();
                                   auto grad_xes_a = grad_xes.packed_accessor32<scalar_t, 4, gpe::RestrictPtrTraits>();
                                   auto grad_output_a = grad_output_copy.packed_accessor32<scalar_t, 3, gpe::RestrictPtrTraits>();

                                   if (n.dims == 2) {
                                       auto fun = [=] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               kernel_bvh_backward<scalar_t, 2>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                                mixture_a, nodes_a, aabbs_a, xes_a,
                                                                                grad_mixture_a, grad_xes_a, grad_output_a,
                                                                                n, requires_grad_mixture, requires_grad_xes);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                                   else {
                                       auto fun = [=] __host__ __device__
                                           (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
                                               kernel_bvh_backward<scalar_t, 3>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
                                                                                mixture_a, nodes_a, aabbs_a, xes_a,
                                                                                grad_mixture_a, grad_xes_a, grad_output_a,
                                                                                n, requires_grad_mixture, requires_grad_xes);
                                           };
                                       gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture), dimGrid, dimBlock, fun);
                                   }
                               }));

    if (use_indirect_xes) {
        auto indices = bvh.m_nodes.index({Slice(), Slice(), Slice(bvh.m_n_internal_nodes, None), 3}).to(torch::ScalarType::Long);
        indices = inverse_permutation(indices);
        grad_xes = torch::gather(grad_xes, 2, indices.view({n.batch_xes, n.layers_xes, n.xes, 1}).expand_as(xes));
    }
    return {grad_mixture, grad_xes};
}

} // namespace bvh_mhem_fit
