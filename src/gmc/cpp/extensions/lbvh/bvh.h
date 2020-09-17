#ifndef LBVH_BVH_H
#define LBVH_BVH_H

#include <type_traits>
#include <iostream>
#include <ios>

#include <cuda_runtime.h>

#include <thrust/swap.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include <cub/device/device_segmented_radix_sort.cuh>

#include <torch/script.h>
#include <torch/nn/functional.h>

#include "lbvh/aabb.h"
#include "lbvh/morton_code.h"

#include "common.h"
#include "math/symeig_cuda.h"

#ifndef __CUDACC__
int atomicCAS(int* address, int compare, int val);

constexpr dim3 blockIdx;
constexpr dim3 blockDim;
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

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace lbvh
{
namespace detail
{
struct Node
{
    std::uint32_t parent_idx; // parent node
    std::uint32_t left_idx;   // index of left  child node
    std::uint32_t right_idx;  // index of right child node
    std::uint32_t object_idx; // == 0xFFFFFFFF if internal node.
};

// a set of pointers to use it on device.
template<typename Real, typename Object, bool IsConst>
struct basic_device_bvh;
template<typename Real, typename Object>
struct basic_device_bvh<Real, Object, false>
{
    using real_type  = Real;
    using aabb_type  = Aabb<real_type>;
    using node_type  = detail::Node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;   // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_objects; // (# of leaves), the same as the number of objects

    node_type *  nodes;
    aabb_type *  aabbs;
    object_type* objects;
};
template<typename Real, typename Object>
struct basic_device_bvh<Real, Object, true>
{
    using real_type  = Real;
    using aabb_type  = Aabb<real_type>;
    using node_type  = detail::Node;
    using index_type = std::uint32_t;
    using object_type = Object;

    unsigned int num_nodes;  // (# of internal node) + (# of leaves), 2N+1
    unsigned int num_objects;// (# of leaves), the same as the number of objects

    node_type   const* nodes;
    aabb_type   const* aabbs;
    object_type const* objects;
};

template<typename UInt>
__host__ __device__
inline uint2 determine_range(UInt const* node_code,
        const unsigned int num_leaves, unsigned int idx)
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

    const int delta_min = thrust::min(L_delta, R_delta);
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
        thrust::swap(idx, jdx); // make it sure that idx < jdx
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

torch::Tensor compute_aabbs(const at::Tensor& mixture) {
    namespace F = torch::nn::functional;
    constexpr float threshold = 0.0001f;

    auto n = gpe::get_ns(mixture);

    torch::Tensor factors = -2 * torch::log(threshold / torch::abs(gpe::weights(mixture)));
    factors = factors.where(factors > 0, torch::zeros({1, 1, 1}, factors.device()));
    factors = torch::sqrt(factors);

    torch::Tensor covs = gpe::covariances(mixture).inverse();
    torch::Tensor eigenvalues;
    torch::Tensor eigenvectors;

    std::tie(eigenvalues, eigenvectors) = gpe::symeig_cuda_forward(covs);
    /*
     * eigenvectors is a tensor of [*, *, *, d, d], where d is the dimensionality (2 or 3)
     * the eigenvectors are in the rows of that d * d matrix.
     */
    eigenvalues = torch::sqrt(eigenvalues);
    eigenvectors = eigenvalues.unsqueeze(-1) * eigenvectors;

    auto ellipsoidM = factors.unsqueeze(-1).unsqueeze(-1) * eigenvectors;

    // https://stackoverflow.com/a/24112864/4032670
    // https://members.loria.fr/SHornus/ellipsoid-bbox.html
    // we take the norm over the eigenvectors, that is analogous to simon fraiss' code in gmvis/core/Gaussian.cpp
    auto delta = torch::norm(ellipsoidM, 2, {-2});
    auto centroid = gpe::positions(mixture);
    auto upper = centroid + delta;
    auto lower = centroid - delta;

    // bring that thing into a format that can be read by our lbvh builder
    // https://pytorch.org/docs/master/nn.functional.html#torch.nn.functional.pad
    upper = F::pad(upper, F::PadFuncOptions({0, 4-n.dims}));
    lower = F::pad(lower, F::PadFuncOptions({0, 4-n.dims}));
    return torch::cat({upper, lower}, -1).contiguous();
}

torch::Tensor compute_aabb_whole(const torch::Tensor& aabbs) {
    using namespace torch::indexing;
    const auto upper = std::get<0>(aabbs.index({Ellipsis, Slice(None, 4)}).max(-2));
    const auto lower = std::get<0>(aabbs.index({Ellipsis, Slice(4, None)}).min(-2));
    return torch::cat({upper, lower}, -1).contiguous();
}

template<typename morton_type>
struct morton_torch_tensor_type;

template<>
struct morton_torch_tensor_type<int32_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Int; }
};

template<>
struct morton_torch_tensor_type<int64_t> {
    static inline constexpr torch::ScalarType id() { return torch::ScalarType::Long; }
};

} // detail

namespace kernels {

template<typename scalar_t, typename morton_torch_t>
__global__ void compute_morton_codes(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aabb_a,
                       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aabb_whole_a,
                       torch::PackedTensorAccessor32<morton_torch_t, 2, torch::RestrictPtrTraits> morton_codes_a,
                       const unsigned n_mixtures, const unsigned n_components) {
    using morton_cuda_t = std::make_unsigned_t<morton_torch_t>;

    const auto mixture_id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto component_id = blockIdx.y * blockDim.y + threadIdx.y;
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
    morton_code |= component_id;
//    morton_code = component_id;
}

template<typename scalar_t, typename morton_torch_t>
__global__ void createLeafNodes(const torch::PackedTensorAccessor32<morton_torch_t, 2, torch::RestrictPtrTraits> morton_codes_a,
                                torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> nodes_a,
                                const unsigned n_mixtures, const unsigned n_components) {
    using morton_cuda_t = std::make_unsigned_t<morton_torch_t>;

    const auto mixture_id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto component_id = blockIdx.y * blockDim.y + threadIdx.y;
//    printf("mixture_id: %d, component: %d\n", mixture_id, component_id);
    if (mixture_id >= n_mixtures || component_id >= n_components)
        return;
    printf("mixture_id: %d, component: %d go\n", mixture_id, component_id);

    const auto& morton_code = reinterpret_cast<const morton_cuda_t&>(morton_codes_a[mixture_id][component_id]);
    auto& node = reinterpret_cast<detail::Node&>(nodes_a[mixture_id][component_id][0]);
    node.object_idx = uint32_t(morton_code & 0xffffffff);
}

template <typename scalar_t, typename morton_torch_t>
__global__ void create_internal_nodes(const torch::PackedTensorAccessor32<morton_torch_t, 2, torch::RestrictPtrTraits> morton_codes_a,
                                      torch::PackedTensorAccessor32<int32_t, 3, torch::RestrictPtrTraits> nodes_a,
                                      const unsigned n_mixtures, const unsigned n_internal_nodes, const unsigned n_leafs) {
    using morton_cuda_t = std::make_unsigned_t<morton_torch_t>;

    const auto mixture_id = blockIdx.x * blockDim.x + threadIdx.x;
    const auto node_id = blockIdx.y * blockDim.y + threadIdx.y;
    if (mixture_id >= n_mixtures || node_id >= n_internal_nodes)
        return;

    const morton_cuda_t& morton_code = reinterpret_cast<const morton_cuda_t&>(morton_codes_a[mixture_id][0]);
    auto& node = reinterpret_cast<detail::Node&>(nodes_a[mixture_id][node_id][0]);
    node.object_idx = 0xFFFFFFFF; //  internal nodes

    const uint2 ij  = detail::determine_range(&morton_code, n_leafs, node_id);
    const auto gamma = detail::find_split(&morton_code, n_leafs, ij.x, ij.y);

    node.left_idx  = gamma;
    node.right_idx = gamma + 1;
    if(min(ij.x, ij.y) == gamma)
    {
        node.left_idx += n_leafs - 1;
    }
    if(max(ij.x, ij.y) == gamma + 1)
    {
        node.right_idx += n_leafs - 1;
    }
    assert(node.left_idx >= 0);
    assert(node.right_idx >= 0);
    reinterpret_cast<detail::Node&>(nodes_a[mixture_id][int(node.left_idx)][0]).parent_idx = node_id;
    reinterpret_cast<detail::Node&>(nodes_a[mixture_id][int(node.right_idx)][0]).parent_idx = node_id;
}

} // kernels


template<typename Real, typename Object>
using  bvh_device = detail::basic_device_bvh<Real, Object, false>;
template<typename Real, typename Object>
using cbvh_device = detail::basic_device_bvh<Real, Object, true>;

template<typename Real, typename Object>
class bvh
{
  public:
    using real_type   = Real;
    using index_type = std::uint32_t;
    using object_type = Object;
    using aabb_type   = Aabb<real_type>;
    using node_type   = detail::Node;

    using scalar_t = real_type;
    using morton_torch_t = int64_t;
    using morton_cuda_t = std::make_unsigned<morton_torch_t>::type;


  public:

    bvh(const torch::Tensor& mixture)
        : m_mixture(mixture), m_n(gpe::get_ns(mixture))
    {
        this->construct();
    }

    bvh_device<real_type, object_type> get_device_repr()       noexcept
    {
        return bvh_device<real_type, object_type>{
            static_cast<unsigned int>(nodes_.size()),
            static_cast<unsigned int>(objects_d_.size()),
            nodes_.data().get(), aabbs_.data().get(), objects_d_.data().get()
        };
    }
    cbvh_device<real_type, object_type> get_device_repr() const noexcept
    {
        return cbvh_device<real_type, object_type>{
            static_cast<unsigned int>(nodes_.size()),
            static_cast<unsigned int>(objects_d_.size()),
            nodes_.data().get(), aabbs_.data().get(), objects_d_.data().get()
        };
    }

    void construct()
    {
        using namespace torch::indexing;

        const unsigned int num_objects        = m_n.components;
        const unsigned int num_internal_nodes = num_objects - 1;
        const unsigned int num_nodes          = num_objects * 2 - 1;

        auto print_nodes = [this, num_nodes]() {
            std::cout << "nodes: " << m_nodes.sizes() << " " << (m_nodes.is_cuda() ? "(cuda)" : "(cpu)") << std::endl;
            const torch::Tensor ncpu = m_nodes.cpu().contiguous();
            for (int i = 0; i < m_n.batch * m_n.layers; ++i) {
                for (int j = 0; j < num_nodes; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        const auto index = i * num_nodes * 4 + j * 4 + k;
                        std::cout << ncpu.data_ptr<int>()[index] << ", ";
                    }
                    std::cout << " || ";
                }
                std::cout << std::endl;
            }
        };

        auto object_aabbs = detail::compute_aabbs(m_mixture);
        std::cout << "object_aabbs:" << object_aabbs << std::endl;
        const auto aabb_whole = detail::compute_aabb_whole(object_aabbs);
        std::cout << "aabb_whole:" << aabb_whole << std::endl;

        auto device = m_mixture.device();

        // --------------------------------------------------------------------
        // calculate morton code of each AABB
        // we produce unique morton codes by extending the actual morton code with the index.
        // TODO: easy, either some scaling and translation using torch + thrust transform, or custom kernel.
        torch::Tensor morton_codes = compute_morton_codes(object_aabbs, aabb_whole);
        {
            std::ios_base::fmtflags f(std::cout.flags());
            std::cout << "morton codes: " << morton_codes.sizes() << " " << (morton_codes.is_cuda() ? "(cuda)" : "(cpu)") << std::endl;
            const torch::Tensor mccpu = morton_codes.cpu().contiguous();
            for (int i = 0; i < m_n.batch * m_n.layers; ++i) {
                std::cout << std::hex;
                for (int j = 0; j < m_n.components; ++j) {
                    const auto index = i * m_n.components + j;
                    std::cout << "0x" << std::setw(16) << mccpu.data_ptr<morton_torch_t>()[index] << "; ";
                }
                std::cout << std::dec << std::endl;
            }
            std::cout.flags(f);
        }


        // --------------------------------------------------------------------
        // sort object-indices by morton code
        // TODO: http://www.orangeowlsolutions.com/archives/1297 (2 sorts), or, we can prepend the gm index to the morton code?
        //       or, this would be the fastest (probably): https://nvlabs.github.io/cub/structcub_1_1_device_segmented_radix_sort.html (implemented; test whether prepending would be faster)

        std::tie(morton_codes, object_aabbs) = sort_morton_codes(morton_codes, object_aabbs);
        {
            std::ios_base::fmtflags f(std::cout.flags());
            std::cout << "sorted morton codes: " << morton_codes.sizes() << " " << (morton_codes.is_cuda() ? "(cuda)" : "(cpu)") << std::endl;
            const torch::Tensor mccpu = morton_codes.cpu().contiguous();
            for (int i = 0; i < m_n.batch * m_n.layers; ++i) {
                std::cout << std::hex;
                for (int j = 0; j < m_n.components; ++j) {
                    const auto index = i * m_n.components + j;
                    std::cout << "0x" << std::setw(16) << mccpu.data_ptr<morton_torch_t>()[index] << "; ";
                }
                std::cout << std::dec << std::endl;
            }
            std::cout.flags(f);
        }
        std::cout << "sorted object_aabbs:" << object_aabbs << std::endl;

        // assemble aabb array
        const auto internal_aabbs = torch::zeros({m_n.batch, m_n.layers, num_internal_nodes, 8}, torch::TensorOptions(device).dtype(object_aabbs.dtype()));
        std::cout << "internal: " << internal_aabbs.sizes() << "  object: " << object_aabbs.sizes() << std::endl;
        m_aabbs = torch::cat({internal_aabbs, object_aabbs}, 2).contiguous();
        std::cout << "m_aabbs:" << m_aabbs << std::endl;

        // construct nodes and create leaf nodes
        m_nodes = torch::ones({m_n.batch, m_n.layers, num_nodes, 4}, torch::TensorOptions(device).dtype(torch::ScalarType::Int)) * -1;
        print_nodes();
        this->create_leaf_nodes(morton_codes);
        print_nodes();

        // create internal nodes
        this->create_internal_nodes(morton_codes);
        print_nodes();

//        // --------------------------------------------------------------------
//        // create AABB for each node by bottom-up strategy
//        // TODO: would also work.
//        thrust::device_vector<int> flag_container(num_internal_nodes, 0);
//        const auto flags = flag_container.data().get();

//        thrust::for_each(thrust::device,
//            thrust::make_counting_iterator<index_type>(num_internal_nodes),
//            thrust::make_counting_iterator<index_type>(num_nodes),
//            [self, flags] __device__ (index_type idx)
//            {
//                unsigned int parent = self.nodes[idx].parent_idx;
//                while(parent != 0xFFFFFFFF) // means idx == 0
//                {
//                    const int old = atomicCAS(flags + parent, 0, 1);
//                    if(old == 0)
//                    {
//                        // this is the first thread entered here.
//                        // wait the other thread from the other child node.
//                        return;
//                    }
//                    assert(old == 1);
//                    // here, the flag has already been 1. it means that this
//                    // thread is the 2nd thread. merge AABB of both childlen.

//                    const auto lidx = self.nodes[parent].left_idx;
//                    const auto ridx = self.nodes[parent].right_idx;
//                    const auto lbox = self.aabbs[lidx];
//                    const auto rbox = self.aabbs[ridx];
//                    self.aabbs[parent] = merge(lbox, rbox);

//                    // look the next parent...
//                    parent = self.nodes[parent].parent_idx;
//                }
//                return;
//            });
        return;
    }

    thrust::device_vector<object_type> getObjects() const
    {
        return objects_d_;
    }

    thrust::device_vector<aabb_type> getAabbs() const
    {
        return aabbs_;
    }

    thrust::device_vector<node_type> getNodes() const
    {
        return nodes_;
    }
protected:
    torch::Tensor compute_morton_codes(const torch::Tensor& aabbs, const torch::Tensor& aabb_whole) const {
        const auto aabbs_view = aabbs.view({-1, m_n.components, 8});
        const auto aabb_whole_view = aabb_whole.view({-1, 8});
        const auto aabb_a = aabbs_view.packed_accessor32<float, 3, torch::RestrictPtrTraits>();
        const auto aabb_whole_a = aabb_whole_view.packed_accessor32<float, 2, torch::RestrictPtrTraits>();

        const auto n_mixtures = aabbs_view.size(0);
        assert(n_mixtures == m_n.batch * m_n.layers);

        auto morton_codes = torch::empty({n_mixtures, m_n.components}, torch::TensorOptions(aabbs.device()).dtype(detail::morton_torch_tensor_type<morton_torch_t>::id()));
        auto morton_codes_a = morton_codes.packed_accessor32<morton_torch_t, 2, torch::RestrictPtrTraits>();

        dim3 dimBlock = dim3(1, 128, 1);
        dim3 dimGrid = dim3((n_mixtures + dimBlock.x - 1) / dimBlock.x,
                                  (m_n.components + dimBlock.y - 1) / dimBlock.y);
        kernels::compute_morton_codes<scalar_t, morton_torch_t><<<dimGrid, dimBlock>>>(aabb_a, aabb_whole_a, morton_codes_a, n_mixtures, m_n.components);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        return morton_codes.view({m_n.batch, m_n.layers, m_n.components});
    }

    std::tuple<torch::Tensor, torch::Tensor> sort_morton_codes(const torch::Tensor& morton_codes, const torch::Tensor& object_aabbs) const {
        auto sorted_morton_codes = morton_codes.clone();
        auto sorted_aabbs = object_aabbs.clone();

        // Declare, allocate, and initialize device-accessible pointers for sorting data
        int num_items = morton_codes.numel();                         // e.g., 8
        int num_segments = m_n.batch * m_n.layers;                    // e.g., 4
        int num_components = m_n.components;                          // 2
        const auto offsets = torch::arange(0, num_segments + 1, torch::TensorOptions(morton_codes.device()).dtype(torch::ScalarType::Int)) * num_components;
        std::cout << "offsets: " << offsets << std::endl;
        int* d_offsets = offsets.data_ptr<int>();                           // e.g., [0, 2, 4, 6, 8]
        const morton_cuda_t* d_keys_in = reinterpret_cast<const morton_cuda_t*>(morton_codes.data_ptr<morton_torch_t>());   // e.g., [8, 6, 7, 5, 3, 0, 9, 8]
        morton_cuda_t* d_keys_out = reinterpret_cast<morton_cuda_t*>(sorted_morton_codes.data_ptr<morton_torch_t>());       // e.g., [-, -, -, -, -, -, -, -]
        const Aabb<float>* d_values_in = reinterpret_cast<const Aabb<float>*>(object_aabbs.data_ptr<float>());              // e.g., [0, 1, 2, 3, 4, 5, 6, 7]
        Aabb<float>* d_values_out = reinterpret_cast<Aabb<float>*>(sorted_aabbs.data_ptr<float>());                         // e.g., [-, -, -, -, -, -, -, -]

        // Determine temporary device storage requirements
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(
//        cub::DeviceSegmentedRadixSort::SortKeys(
                    d_temp_storage, temp_storage_bytes,
                    d_keys_in, d_keys_out,
                    d_values_in, d_values_out,
                    num_items, num_segments,
                    d_offsets, d_offsets + 1);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        // Run sorting operation
        cub::DeviceSegmentedRadixSort::SortPairs(
//        cub::DeviceSegmentedRadixSort::SortKeys(
                    d_temp_storage, temp_storage_bytes,
                    d_keys_in, d_keys_out,
                    d_values_in, d_values_out,
                    num_items, num_segments,
                    d_offsets, d_offsets + 1,
                    0, sizeof (morton_cuda_t) * 8, 0, true);
        // d_keys_out            <-- [6, 8, 5, 7, 0, 3, 8, 9]
        // d_values_out          <-- [1, 0, 3, 2, 5, 4, 7, 6]

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cudaFree(d_temp_storage);

        return std::make_tuple(sorted_morton_codes, sorted_aabbs);
    }

    torch::Tensor create_leaf_nodes(const torch::Tensor& morton_codes) {
        using namespace torch::indexing;
        auto n_mixtures = m_n.batch * m_n.layers;
        // no support for negative slicing indexes at the time of writing v
        auto nodes_view = m_nodes.index({Ellipsis, Slice(m_nodes.size(-2) - m_n.components, None), Slice()})
                                  .view({n_mixtures, m_n.components, sizeof(detail::Node)/sizeof (int32_t)});
        const auto morton_codes_view = morton_codes.view({n_mixtures, m_n.components});
        const auto morton_codes_a = morton_codes_view.packed_accessor32<morton_torch_t, 2, torch::RestrictPtrTraits>();
        auto nodes_a = nodes_view.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>();


        dim3 dimBlock = dim3(1, 128, 1);
        dim3 dimGrid = dim3((n_mixtures + dimBlock.x - 1) / dimBlock.x,
                                  (m_n.components + dimBlock.y - 1) / dimBlock.y);
        kernels::createLeafNodes<scalar_t, morton_torch_t><<<dimGrid, dimBlock>>>(morton_codes_a, nodes_a, n_mixtures, m_n.components);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        return morton_codes;
    }

    void create_internal_nodes(const torch::Tensor& morton_codes)
    {
        using namespace torch::indexing;
        auto n_mixtures = m_n.batch * m_n.layers;
        // no support for negative slicing indexes at the time of writing v
        auto nodes_view = m_nodes.view({n_mixtures, -1, sizeof(detail::Node)/sizeof (int32_t)});
        auto nodes_a = nodes_view.packed_accessor32<int32_t, 3, torch::RestrictPtrTraits>();
        const auto morton_codes_view = morton_codes.view({n_mixtures, m_n.components});
        const auto morton_codes_a = morton_codes_view.packed_accessor32<morton_torch_t, 2, torch::RestrictPtrTraits>();

        auto n_internal_nodes = m_nodes.size(-2) - m_n.components;
        dim3 dimBlock = dim3(1, 128, 1);
        dim3 dimGrid = dim3((n_mixtures + dimBlock.x - 1) / dimBlock.x,
                            (n_internal_nodes + dimBlock.y - 1) / dimBlock.y);
        kernels::create_internal_nodes<scalar_t, morton_torch_t><<<dimGrid, dimBlock>>>(morton_codes_a, nodes_a,
                                                                                        n_mixtures, n_internal_nodes, m_n.components);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
    }

private:
    gpe::MixtureNs m_n;
    const torch::Tensor m_mixture;
    torch::Tensor m_aabbs;
    torch::Tensor m_nodes;

    thrust::device_vector<object_type>   objects_d_;
    thrust::device_vector<aabb_type>     aabbs_;
    thrust::device_vector<node_type>     nodes_;
};



} // lbvh
#endif// LBVH_BVH_H
