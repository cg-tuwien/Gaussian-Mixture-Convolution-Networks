#include "convolution/implementation.h"
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/types.h>

#include "common.h"
#include "cuda_qt_creator_definitinos.h"
#include "cuda_operations.h"
#include "hacked_accessor.h"
#include "lbvh/bvh.h"
#include "util/glm.h"
#include "util/scalar.h"
#include "util/algorithms.h"
#include "util/containers.h"
#include "util/cuda.h"
#include "util/gaussian.h"
#include "util/gaussian_mixture.h"
#include "util/grad/algorithms.h"
#include "util/grad/glm.h"
#include "util/grad/gaussian.h"
#include "util/grad/mixture.h"
#include "util/mixture.h"
#include "parallel_start.h"
#include "ParallelStack.h"


// todo:
// - in collect_result, run a new fitting with the most important node to fill up the remaining gaussian slots
// - grad of position can explode into the amplitude (e.g. grad on position == 1 => grad on amplitude == 100)
// - grad on covariance can explode into grad on position (10x only, compared to above)

namespace convolution {

template<typename scalar_t, unsigned N_DIMS>
std::pair<torch::Tensor, torch::Tensor> backward_impl_t(const torch::Tensor& grad, const ForwardOutput& forward_out) {
    using namespace torch::indexing;
    using Tree = lbvh::Bvh<N_DIMS, scalar_t>;
    return {};

    // todo: flatten mixture for kernel, i.g. nbatch/nlayers/ncomponents/7 => nmixture/ncomponents/7

//    auto n = gpe::get_ns(forward_out.target);
//    TORCH_CHECK(n.batch * n.layers < 65535, "n_batch x n_layers must be smaller than 65535 for CUDA")
//    TORCH_CHECK(n.components > 1, "number of components must be greater 1 for this implementation")
//    TORCH_CHECK(n.components < 65535, "number of components must be smaller than 65535 for morton code computation")
//    TORCH_CHECK(n.dims == N_DIMS, "something wrong with dispatch")
//    TORCH_CHECK(grad.dtype() == caffe2::TypeMeta::Make<scalar_t>(), "something wrong with dispatch, or maybe this float type is not supported.")

//    const auto n_mixtures = n.batch * n.layers;
//    const auto bvh = Tree(forward_out.target, forward_out.bvh_nodes, {});
//    const auto n_internal_nodes = bvh.m_n_internal_nodes;
//    const auto n_nodes = bvh.m_n_nodes;
//    const auto mixture_view = forward_out.target.view({n_mixtures, n.components, -1}).contiguous();
//    const auto grad_view = grad.view({n_mixtures, config.n_components_fitting, -1}).contiguous();
//    auto flat_bvh_nodes = bvh.m_nodes.view({n_mixtures, n_nodes, -1});
//    auto flag_container = torch::zeros({n_mixtures, n_internal_nodes}, torch::TensorOptions(mixture_view.device()).dtype(torch::ScalarType::Int));

//    auto flags_a = gpe::accessor<int, 2>(flag_container);
//    auto node_attributes = forward_out.bvh_attributes.view({n_mixtures, n_nodes, -1});

//    auto mixture_a = gpe::struct_accessor<gpe::Gaussian<N_DIMS, scalar_t>, 2, scalar_t>(mixture_view);
//    auto grad_a = gpe::accessor<scalar_t, 3>(grad_view);
//    auto nodes_a = gpe::accessor<lbvh::detail::Node::index_type_torch, 3>(flat_bvh_nodes);
//    auto node_attributes_a = gpe::struct_accessor<typename AugmentedBvh<scalar_t, N_DIMS, REDUCTION_N>::NodeAttributes, 2, uint8_t>(node_attributes);

//    {
//        // distribute the fitting gradient using the same algorithm amoung the nodes.
//        dim3 dimBlock = dim3(32, 1, 1);
//        dim3 dimGrid = dim3((uint(n_mixtures) + dimBlock.x - 1) / dimBlock.x, 1, 1);

//        auto fun = [mixture_a, grad_a, nodes_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config]
//                __host__ __device__
//                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
//            distribute_grad<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
//                                                          mixture_a, grad_a, nodes_a, flags_a, node_attributes_a,
//                                                          n, n_mixtures, n_internal_nodes, n_nodes,
//                                                          config);
//        };
//        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture_view), dimGrid, dimBlock, fun);
//    }

//    auto target_gradient = torch::zeros_like(mixture_view);
//    auto target_gradient_a = gpe::accessor<scalar_t, 3>(target_gradient);
//    {
//        dim3 dimBlock = dim3(32, 1, 1);
//        dim3 dimGrid = dim3(uint(1),
//                            (uint(n_mixtures) + dimBlock.y - 1) / dimBlock.y,
//                            (uint(1) + dimBlock.z - 1) / dimBlock.z);

//        auto fun = [target_gradient_a, mixture_a, nodes_a, flags_a, node_attributes_a, n, n_mixtures, n_internal_nodes, n_nodes, config] __host__ __device__
//                (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) {
//            trickle_down_grad<scalar_t, N_DIMS, REDUCTION_N>(gpe_gridDim, gpe_blockDim, gpe_blockIdx, gpe_threadIdx,
//                                                             target_gradient_a,
//                                                             mixture_a, nodes_a, flags_a, node_attributes_a,
//                                                             n, n_mixtures, n_internal_nodes, n_nodes,
//                                                             config);
//        };
//        gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(mixture_view), dimGrid, dimBlock, fun);
//    }



//    return target_gradient.view_as(forward_out.target);
}

} // namespace bvh_mhem_fit

