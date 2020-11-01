#include "lbvh/bvh.h"

#include <iostream>
#include <chrono>

#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/nn/functional.h>

#include "cuda_qt_creator_definitinos.h"
#include "hacked_accessor.h"
#include "lbvh/morton_code.h"
#include "math/symeig_detail.h"
#include "math/symeig_cuda.h"
#include "math/scalar.h"
#include "math/matrix.h"
#include "mixture.h"
#include "parallel_start.h"

namespace lbvh
{
namespace {
inline torch::Tensor compute_aabb_whole(const torch::Tensor& aabbs) {
    using namespace torch::indexing;
    const auto upper = std::get<0>(aabbs.index({Ellipsis, Slice(None, 4)}).max(-2));
    const auto lower = std::get<0>(aabbs.index({Ellipsis, Slice(4, None)}).min(-2));
    return torch::cat({upper, lower}, -1).contiguous();
}
}

namespace kernels
{

template<typename UInt>
__host__ __device__
inline uint2 determine_range(UInt const* node_code, const unsigned int num_leaves, unsigned int idx)
{
    if(idx == 0)
    {
        return make_uint2(0, num_leaves-1);
    }

    // determine direction of the range
    const UInt self_code = node_code[idx];
    const int L_delta = common_upper_bits(self_code, node_code[idx-1]);
    const int R_delta = common_upper_bits(self_code, node_code[idx+1]);
    const int d = (R_delta > L_delta) ? 1 : -1;

    // Compute upper bound for the length of the range
    const int delta_min = gpe::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;
    if(0 <= i_tmp && i_tmp < num_leaves)
    {
        delta = common_upper_bits(self_code, node_code[i_tmp]);
    }
    while(delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
    }

    // Find the other end by binary search
    int l = 0;
    int t = l_max >> 1;
    while(t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if(0 <= i_tmp && i_tmp < num_leaves)
        {
            delta = common_upper_bits(self_code, node_code[i_tmp]);
        }
        if(delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }
    unsigned int jdx = idx + l * d;
    if(d < 0)
    {
        gpe::swap(idx, jdx); // make it sure that idx < jdx
    }
    return make_uint2(idx, jdx);
}

template<typename UInt>
__host__ __device__
inline unsigned int find_split(UInt const* node_code, const unsigned int num_leaves,
               const unsigned int first, const unsigned int last) noexcept
{
    const UInt first_code = node_code[first];
    const UInt last_code  = node_code[last];
    if (first_code == last_code)
    {
        return (first + last) >> 1;
    }
    const int delta_node = common_upper_bits(first_code, last_code);

    // binary search...
    int split  = first;
    int stride = last - first;
    do
    {
        stride = (stride + 1) >> 1;
        const int middle = split + stride;
        if (middle < last)
        {
            const int delta = common_upper_bits(first_code, node_code[middle]);
            if (delta > delta_node)
            {
                split = middle;
            }
        }
    }
    while(stride > 1);

    return split;
}

} // namespace kernels

template<int N_DIMS, typename scalar_t>
void Bvh<N_DIMS, scalar_t>::construct()
{
    using namespace torch::indexing;

//    auto print_nodes = [this]() {
//        std::cout << "nodes: " << m_nodes.sizes() << " " << (m_nodes.is_cuda() ? "(cuda)" : "(cpu)") << std::endl;
//        const torch::Tensor ncpu = m_nodes.cpu().contiguous();
//        for (int i = 0; i < m_n.batch * m_n.layers; ++i) {
//            for (int j = 0; j < this->m_n_nodes; ++j) {
//                for (int k = 0; k < 4; ++k) {
//                    const auto index = i * this->m_n_nodes * 4 + j * 4 + k;
//                    std::cout << ncpu.data_ptr<int16_t>()[index] << ", ";
//                }
//                std::cout << " || ";
//            }
//            std::cout << std::endl;
//        }
//    };

//    auto print_morton_codes = [this](const torch::Tensor& morton_codes) {
//        std::ios_base::fmtflags f(std::cout.flags());
//        std::cout << "morton codes: " << morton_codes.sizes() << " " << (morton_codes.is_cuda() ? "(cuda)" : "(cpu)") << std::endl;
//        const torch::Tensor mccpu = morton_codes.cpu().contiguous();
//        for (int i = 0; i < m_n.batch * m_n.layers; ++i) {
//            std::cout << std::hex;
//            for (int j = 0; j < m_n.components; ++j) {
//                const auto index = i * m_n.components + j;
//                std::cout << "0x" << std::setw(16) << mccpu.data_ptr<morton_torch_t>()[index] << "; ";
//            }
//            std::cout << std::dec << std::endl;
//        }
//        std::cout.flags(f);
//    };

//    m_mixture = m_mixture.cuda();

    //        auto timepoint = std::chrono::high_resolution_clock::now();
    auto watch_stop = [/*&timepoint*/](const std::string& name = "") {
        //            cudaDeviceSynchronize();
        //            if (name.length() > 0)
        //                std::cout << name << ": " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-timepoint).count() << "ms\n";
        //            timepoint = std::chrono::high_resolution_clock::now();
    };
    watch_stop();
    auto object_aabbs = this->compute_aabbs();
    watch_stop("compute_aabbs");

    //        std::cout << "object_aabbs:" << object_aabbs << std::endl;
    const auto aabb_whole = compute_aabb_whole(object_aabbs);
    watch_stop("compute_aabb_whole");
    //        std::cout << "aabb_whole:" << aabb_whole << std::endl;

    auto device = m_mixture.device();

    // --------------------------------------------------------------------
    // we produce unique morton codes by extending the actual morton code with the index.
    torch::Tensor morton_codes = compute_morton_codes(object_aabbs, aabb_whole);
    watch_stop("compute_morton_codes");

    // --------------------------------------------------------------------
    std::tie(morton_codes, object_aabbs) = sort_morton_codes(morton_codes, object_aabbs);
    watch_stop("sort_morton_codes");
    //        print_morton_codes(morton_codes);

    //        std::cout << "sorted object_aabbs:" << object_aabbs << std::endl;

    // assemble aabb array
    const auto internal_aabbs = torch::zeros({m_n.batch, m_n.layers, m_n_internal_nodes, 8}, torch::TensorOptions(device).dtype(object_aabbs.dtype()));
    //        std::cout << "internal: " << internal_aabbs.sizes() << "  object: " << object_aabbs.sizes() << std::endl;
    m_aabbs = torch::cat({internal_aabbs, object_aabbs}, 2).contiguous();
    //        std::cout << "m_aabbs:" << m_aabbs << std::endl;
    watch_stop("assemble aabb array");

    // construct nodes and create leaf nodes
    m_nodes = torch::ones({m_n.batch, m_n.layers, m_n_nodes, 4}, torch::TensorOptions(device).dtype(torch::ScalarType::Short)) * -1;
    //        print_nodes();
    this->create_leaf_nodes(morton_codes);
    watch_stop("construct nodes and create_leaf_nodes");
    //        print_nodes();

    // create internal nodes
    this->create_internal_nodes(morton_codes);
    watch_stop("create_internal_nodes");
    //        print_nodes();

    // create AABB for each node by bottom-up strategy
    this->create_aabbs_for_internal_nodes();
    watch_stop("create_internal_nodes");
    //        std::cout << "m_aabbs:" << m_aabbs << std::endl;
}


template<typename scalar_t> __host__ __device__
glm::mat<2, 2, scalar_t> mul_eigenvecs_with_eigenvals(const glm::mat<2, 2, scalar_t>& eigenvectors, const glm::vec<2, scalar_t>& eigenvalues) {
    return glm::mat<2, 2, scalar_t>(eigenvectors[0] * eigenvalues[0], eigenvectors[1] * eigenvalues[1]);
}
template<typename scalar_t> __host__ __device__
glm::mat<3, 3, scalar_t> mul_eigenvecs_with_eigenvals(const glm::mat<3, 3, scalar_t>& eigenvectors, const glm::vec<3, scalar_t>& eigenvalues) {
    return glm::mat<3, 3, scalar_t>(eigenvectors[0] * eigenvalues[0], eigenvectors[1] * eigenvalues[1], eigenvectors[2] * eigenvalues[2]);
}
template<typename scalar_t> __host__ __device__
const glm::vec<2, scalar_t> colwise_length(const glm::mat<2, 2, scalar_t>& mat) {
    return glm::vec<2, scalar_t>(glm::length(mat[0]), glm::length(mat[1]));
}
template<typename scalar_t> __host__ __device__
const glm::vec<3, scalar_t> colwise_length(const glm::mat<3, 3, scalar_t>& mat) {
    return glm::vec<3, scalar_t>(glm::length(mat[0]), glm::length(mat[1]), glm::length(mat[2]));
}

template<int N_DIMS, typename scalar_t>
at::Tensor Bvh<N_DIMS, scalar_t>::compute_aabbs() {
    constexpr scalar_t threshold = scalar_t(0.001);

    auto aabbs = torch::zeros({m_n.batch, m_n.layers, m_n.components, 8}, torch::TensorOptions(m_mixture.device()).dtype(detail::TorchTypeMapper<scalar_t>::id()));
    auto aabbs_view = aabbs.view({-1, 8});
    auto aabbs_a = gpe::accessor<scalar_t, 2>(aabbs_view);

    torch::Tensor gaussians = m_mixture.contiguous().view({-1, m_mixture.size(-1)});
    auto gaussians_a = gpe::accessor<scalar_t, 2>(gaussians);
    auto n_gaussians = gaussians.size(0);

    dim3 dimBlock = dim3(1024, 1, 1);
    dim3 dimGrid = dim3(unsigned(n_gaussians + dimBlock.x - 1) / dimBlock.x, 1, 1);
    auto fun = [aabbs_a, gaussians_a, n_gaussians, threshold] __host__ __device__
            (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        GPE_UNUSED(gpe_gridDim)
                using Vec = glm::vec<N_DIMS, scalar_t>;
                using Mat = glm::mat<N_DIMS, N_DIMS, scalar_t>;

                const auto gaussian_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
                if (gaussian_id >= n_gaussians)
                    return;

                const gpe::Gaussian<N_DIMS, scalar_t>& gaussian = reinterpret_cast<const gpe::Gaussian<N_DIMS, scalar_t>&>(gaussians_a[gaussian_id][0]);

                scalar_t factor = -2 * gpe::log(threshold / gpe::abs(gaussian.weight));
                // the backward pass doesn't use the weight to compute the gradient for the weight. therefore we need to have a lower
                // bound for the ellipsoid, which is at 1% of a gaussian with weight 1 => log (0.01 / 1)

                factor = gpe::max(factor, scalar_t(-2 * -2.995732273553991));   // -2 * log(0.05), would use constexpr, but unsure about cuda
                //    factor = gpe::max(factor, scalar_t(-2 * -4.605170185988091));   // -2 * log(0.01), would use constexpr, but unsure about cuda
                //    factor = gpe::max(factor, scalar_t(-2 * -6.907755278982137));   // -2 * log(0.001), would use constexpr, but unsure about cuda

                // TODO: when it works, we can probably remove one of the sqrt and sqrt after they are mul together
                factor = gpe::sqrt(factor);

                Vec eigenvalues;
                Mat eigenvectors;
                // torch inverse is slow, do it with glm
                thrust::tie(eigenvalues, eigenvectors) = gpe::detail::compute_symeig(glm::inverse(gaussian.covariance));

                //    printf("g%d: eigenvalues=%f/%f\n", gaussian_id, eigenvalues[0], eigenvalues[1]);
                //    printf("g%d: eigenvectors=\n%f/%f\n%f/%f\n", gaussian_id, eigenvectors[0][0], eigenvectors[0][1], eigenvectors[1][0], eigenvectors[1][1]);

                eigenvalues = glm::sqrt(eigenvalues);
                eigenvectors = mul_eigenvecs_with_eigenvals(eigenvectors, eigenvalues);

                auto ellipsoidM = factor * eigenvectors;
                //    printf("g%d: ellipsoidM=\n%f/%f\n%f/%f\n", gaussian_id, ellipsoidM[0][0], ellipsoidM[0][1], ellipsoidM[1][0], ellipsoidM[1][1]);

                // https://stackoverflow.com/a/24112864/4032670
                // https://members.loria.fr/SHornus/ellipsoid-bbox.html
                // we take the norm over the eigenvectors, that is analogous to simon fraiss' code in gmvis/core/Gaussian.cpp

                ellipsoidM = glm::transpose(ellipsoidM);
                auto delta = colwise_length(ellipsoidM);

                auto upper = gaussian.position + delta;
                auto lower = gaussian.position - delta;

                Aabb<scalar_t>& aabb = reinterpret_cast<Aabb<scalar_t>&>(aabbs_a[gaussian_id][0]);
                aabb.upper = make_vector_of(upper);
                aabb.lower = make_vector_of(lower);
    };
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(m_mixture), dimGrid, dimBlock, fun);
    return aabbs;
}

template<int N_DIMS, typename scalar_t>
at::Tensor Bvh<N_DIMS, scalar_t>::compute_morton_codes(const at::Tensor& aabbs, const at::Tensor& aabb_whole) const {
    const auto aabbs_view = aabbs.view({-1, m_n.components, 8});
    const auto aabb_whole_view = aabb_whole.view({-1, 8});
    const auto aabb_a = gpe::accessor<scalar_t, 3>(aabbs_view);
    const auto aabb_whole_a = gpe::accessor<scalar_t, 2>(aabb_whole_view);

    const auto n_mixtures = aabbs_view.size(0);
    const auto n_components = int(m_n_leaf_nodes);
    assert(n_mixtures == m_n.batch * m_n.layers);

    auto morton_codes = torch::empty({n_mixtures, m_n_leaf_nodes}, torch::TensorOptions(aabbs.device()).dtype(detail::TorchTypeMapper<morton_torch_t>::id()));
    auto morton_codes_a = gpe::accessor<morton_torch_t, 2>(morton_codes);

    dim3 dimBlock = dim3(1, 128, 1);
    dim3 dimGrid = dim3((unsigned(n_mixtures) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(m_n_leaf_nodes) + dimBlock.y - 1) / dimBlock.y);
    auto fun = [morton_codes_a, aabb_a, aabb_whole_a, n_mixtures, n_components] __host__ __device__
            (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
                GPE_UNUSED(gpe_gridDim)

                const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
                const auto component_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
                if (mixture_id >= n_mixtures || component_id >= n_components)
                    return;

                const auto& aabb = reinterpret_cast<const Aabb<float>&>(aabb_a[mixture_id][component_id][0]);
                const auto& whole = reinterpret_cast<const Aabb<float>&>(aabb_whole_a[mixture_id][0]);
                auto& morton_code = reinterpret_cast<morton_cuda_t&>(morton_codes_a[mixture_id][component_id]);

                auto p = centroid(aabb);
                p.x -= whole.lower.x;
                p.y -= whole.lower.y;
                p.z -= whole.lower.z;
                p.x /= (whole.upper.x - whole.lower.x);
                p.y /= (whole.upper.y - whole.lower.y);
                p.z /= (whole.upper.z - whole.lower.z);
                morton_code = lbvh::morton_code(p);

                //    morton_code << sizeof (morton_torch_t) * 2;
                //    morton_code = mixture_id;
                morton_code <<= 32;
                morton_code |= morton_cuda_t(component_id);
                //    morton_code = component_id;
    };
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(m_mixture), dimGrid, dimBlock, fun);
    return morton_codes.view({m_n.batch, m_n.layers, m_n.components});
}

template<int N_DIMS, typename scalar_t>
std::tuple<at::Tensor, at::Tensor> Bvh<N_DIMS, scalar_t>::sort_morton_codes(const at::Tensor& morton_codes, const at::Tensor& object_aabbs) const {
    int num_segments = m_n.batch * m_n.layers;                    // e.g., 4
    int num_components = m_n.components;                          // 2
    auto sorted_morton_codes = morton_codes.clone();
    auto sorted_aabbs = object_aabbs.clone();
    const morton_cuda_t* d_keys_in = reinterpret_cast<const morton_cuda_t*>(morton_codes.data_ptr<morton_torch_t>());   // e.g., [8, 6, 7, 5, 3, 0, 9, 8]
    morton_cuda_t* d_keys_out = reinterpret_cast<morton_cuda_t*>(sorted_morton_codes.data_ptr<morton_torch_t>());       // e.g., [-, -, -, -, -, -, -, -]
    const Aabb<scalar_t>* d_values_in = reinterpret_cast<const Aabb<scalar_t>*>(object_aabbs.data_ptr<scalar_t>());              // e.g., [0, 1, 2, 3, 4, 5, 6, 7]
    Aabb<scalar_t>* d_values_out = reinterpret_cast<Aabb<scalar_t>*>(sorted_aabbs.data_ptr<scalar_t>());                         // e.g., [-, -, -, -, -, -, -, -]

    if (morton_codes.is_cuda()) {
        // Declare, allocate, and initialize device-accessible pointers for sorting data
        int num_items = int(morton_codes.numel());                         // e.g., 8
        const auto offsets = torch::arange(0, num_segments + 1, torch::TensorOptions(morton_codes.device()).dtype(torch::ScalarType::Int)) * num_components;
        //        std::cout << "offsets: " << offsets << std::endl;
        int* d_offsets = offsets.data_ptr<int>();                           // e.g., [0, 2, 4, 6, 8]

        // Determine temporary device storage requirements
        void     *d_temp_storage = nullptr;
        size_t   temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            d_values_in, d_values_out,
            num_items, num_segments,
            d_offsets, d_offsets + 1);

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());
        // Run sorting operation
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp_storage, temp_storage_bytes,
            d_keys_in, d_keys_out,
            d_values_in, d_values_out,
            num_items, num_segments,
            d_offsets, d_offsets + 1);
        // d_keys_out            <-- [6, 8, 5, 7, 0, 3, 8, 9]
        // d_values_out          <-- [1, 0, 3, 2, 5, 4, 7, 6]

        GPE_CUDA_ASSERT(cudaPeekAtLastError());
        GPE_CUDA_ASSERT(cudaDeviceSynchronize());

        cudaFree(d_temp_storage);
    }
    else {
        // this is most likely not the fastest possible solution, but quick to implement and only cpu code and not perf bottleneck to my knowledge (test if in doubt!)
//        #pragma omp parallel for num_threads(omp_get_num_procs())
        for (int i = 0; i < num_segments; ++i) {
            std::map<morton_cuda_t, const Aabb<scalar_t>*> map;
            for (int j = 0; j < num_components; ++j) {
                map.emplace(d_keys_in[i*num_components + j], d_values_in + i*num_components + j);
            }
            int j = 0;
            for(const auto& pair : map) {
                d_keys_out[i*num_components + j] = pair.first;
                d_values_out[i*num_components + j] = *pair.second;
                ++j;
            }
        }
    }

    return std::make_tuple(sorted_morton_codes, sorted_aabbs);
}

template<int N_DIMS, typename scalar_t>
at::Tensor Bvh<N_DIMS, scalar_t>::create_leaf_nodes(const at::Tensor& morton_codes) {
    using namespace torch::indexing;
    auto n_mixtures = m_n.batch * m_n.layers;
    auto n_components = m_n.components;
    // no support for negative slicing indexes at the time of writing v
    auto nodes_view = m_nodes.index({Ellipsis, Slice(m_nodes.size(-2) - m_n.components, None), Slice()})
                          .view({n_mixtures, m_n_leaf_nodes, 4});
    const auto morton_codes_view = morton_codes.view({n_mixtures, m_n_leaf_nodes});
    const auto morton_codes_a = gpe::accessor<morton_torch_t, 2>(morton_codes_view);
    auto nodes_a = gpe::accessor<int16_t, 3>(nodes_view);


    dim3 dimBlock = dim3(1, 128, 1);
    dim3 dimGrid = dim3((unsigned(n_mixtures) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(m_n.components) + dimBlock.y - 1) / dimBlock.y);

    auto fun = [morton_codes_a, nodes_a, n_mixtures, n_components] __host__ __device__
            (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
        GPE_UNUSED(gpe_gridDim)
        using morton_cuda_t = std::make_unsigned_t<morton_torch_t>;

        const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
        const auto component_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
        if (mixture_id >= n_mixtures || component_id >= n_components)
            return;

        const auto& morton_code = reinterpret_cast<const morton_cuda_t&>(morton_codes_a[mixture_id][component_id]);
        auto& node = reinterpret_cast<detail::Node&>(nodes_a[mixture_id][component_id][0]);
        node.object_idx = lbvh::detail::Node::index_type(morton_code); // imo the cast will cut away the morton code. no need for "& 0xfffffff" // uint32_t(morton_code & 0xffffffff);
    };
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(m_mixture), dimGrid, dimBlock, fun);
    return morton_codes;
}

template<int N_DIMS, typename scalar_t>
void Bvh<N_DIMS, scalar_t>::create_internal_nodes(const at::Tensor& morton_codes)
{
    using namespace torch::indexing;
    auto n_mixtures = m_n.batch * m_n.layers;
    auto n_internal_nodes = m_n_internal_nodes;
    auto n_leaf_nodes = m_n_leaf_nodes;
    // no support for negative slicing indexes at the time of writing v
    auto nodes_view = m_nodes.view({n_mixtures, m_n_nodes, 4});
    auto nodes_a = gpe::accessor<int16_t, 3>(nodes_view);
    const auto morton_codes_view = morton_codes.view({n_mixtures, m_n_leaf_nodes});
    const auto morton_codes_a = gpe::accessor<morton_torch_t, 2>(morton_codes_view);

    dim3 dimBlock = dim3(1, 128, 1);
    dim3 dimGrid = dim3((unsigned(n_mixtures) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(m_n_internal_nodes) + dimBlock.y - 1) / dimBlock.y);
    auto fun = [morton_codes_a, nodes_a, n_mixtures, n_internal_nodes, n_leaf_nodes] __host__ __device__
            (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
                GPE_UNUSED(gpe_gridDim)
                using morton_cuda_t = std::make_unsigned_t<morton_torch_t>;

                const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
                const auto node_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y);
                if (mixture_id >= n_mixtures || node_id >= n_internal_nodes)
                    return;

                const morton_cuda_t& morton_code = reinterpret_cast<const morton_cuda_t&>(morton_codes_a[mixture_id][0]);
                auto& node = reinterpret_cast<detail::Node&>(nodes_a[mixture_id][node_id][0]);
//                node.object_idx = lbvh::detail::Node::index_type(0xFFFFFFFF); //  internal nodes // original
                node.object_idx = lbvh::detail::Node::index_type(node_id);

                const uint2 ij  = kernels::determine_range(&morton_code, n_leaf_nodes, node_id);
                const auto gamma = kernels::find_split(&morton_code, n_leaf_nodes, ij.x, ij.y);

                node.left_idx  = lbvh::detail::Node::index_type(gamma);
                node.right_idx = lbvh::detail::Node::index_type(gamma + 1);
                if(gpe::min(ij.x, ij.y) == gamma)
                {
                    node.left_idx += n_leaf_nodes - 1;
                }
                if(gpe::max(ij.x, ij.y) == gamma + 1)
                {
                    node.right_idx += n_leaf_nodes - 1;
                }
                assert(node.left_idx != lbvh::detail::Node::index_type(0xFFFFFFFF));
                assert(node.right_idx != lbvh::detail::Node::index_type(0xFFFFFFFF));
                reinterpret_cast<detail::Node&>(nodes_a[mixture_id][int(node.left_idx)][0]).parent_idx = lbvh::detail::Node::index_type(node_id);
                reinterpret_cast<detail::Node&>(nodes_a[mixture_id][int(node.right_idx)][0]).parent_idx = lbvh::detail::Node::index_type(node_id);
    };
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(m_mixture), dimGrid, dimBlock, fun);
}

template<int N_DIMS, typename scalar_t>
void Bvh<N_DIMS, scalar_t>::create_aabbs_for_internal_nodes() {
    const auto n_mixtures = m_n.batch * m_n.layers;
    const auto n_internal_nodes = m_n_internal_nodes;
    const auto n_nodes = int(m_n_nodes);
    auto flag_container = torch::zeros({n_mixtures, m_n_internal_nodes}, torch::TensorOptions(m_mixture.device()).dtype(torch::ScalarType::Int));
    auto flags_a = gpe::accessor<int, 2>(flag_container);

    auto nodes_view = m_nodes.view({n_mixtures, m_n_nodes, 4});
    auto nodes_a = gpe::accessor<int16_t, 3>(nodes_view);
    auto aabbs_view = m_aabbs.view({-1, m_n_nodes, 8});
    auto aabb_a = gpe::accessor<scalar_t, 3>(aabbs_view);

    // todo: x is fastest spinning. should optimise memory access patterns!
    //       think/test about what's best for this tree walk; many threads idling due to atomic cas flag blah
    dim3 dimBlock = dim3(1, 128, 1);
    dim3 dimGrid = dim3((unsigned(n_mixtures) + dimBlock.x - 1) / dimBlock.x,
                        (unsigned(m_n_leaf_nodes) + dimBlock.y - 1) / dimBlock.y);
    auto fun = [flags_a, nodes_a, aabb_a, n_mixtures, n_internal_nodes, n_nodes] __host__ __device__
            (const dim3& gpe_gridDim, const dim3& gpe_blockDim, const dim3& gpe_blockIdx, const dim3& gpe_threadIdx) mutable {
                GPE_UNUSED(gpe_gridDim)
                const auto mixture_id = int(gpe_blockIdx.x * gpe_blockDim.x + gpe_threadIdx.x);
                const auto node_id = int(gpe_blockIdx.y * gpe_blockDim.y + gpe_threadIdx.y + n_internal_nodes);
                if (mixture_id >= n_mixtures || node_id >= n_nodes)
                    return;

                const auto* node = &reinterpret_cast<const detail::Node&>(nodes_a[int(mixture_id)][int(node_id)][0]);
                while(node->parent_idx != detail::Node::index_type(0xFFFFFFFF)) // means idx == 0
                {
                    auto* flag = &reinterpret_cast<int&>(flags_a[mixture_id][node->parent_idx]);
                    const int old = gpe::atomicCAS(flag, 0, 1);
                    if(old == 0)
                    {
                        // this is the first thread entered here.
                        // wait the other thread from the other child node.
                        return;
                    }
                    assert(old == 1);
                    // here, the flag has already been 1. it means that this
                    // thread is the 2nd thread. merge AABB of both childlen.

                    auto& current_aabb = reinterpret_cast<Aabb<scalar_t>&>(aabb_a[mixture_id][node->parent_idx][0]);
                    node = &reinterpret_cast<const detail::Node&>(nodes_a[mixture_id][node->parent_idx][0]);
                    const auto& left_aabb = reinterpret_cast<Aabb<scalar_t>&>(aabb_a[mixture_id][node->left_idx][0]);
                    const auto& right_aabb = reinterpret_cast<Aabb<scalar_t>&>(aabb_a[mixture_id][node->right_idx][0]);

                    current_aabb = merge(left_aabb, right_aabb);
                }
    };
    gpe::start_parallel<gpe::ComputeDevice::Both>(gpe::device(m_mixture), dimGrid, dimBlock, fun);
}

template class Bvh<2, float>;
template class Bvh<2, double>;
template class Bvh<3, float>;
template class Bvh<3, double>;

} // namespace lbvh

