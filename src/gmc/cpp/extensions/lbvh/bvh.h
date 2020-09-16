#ifndef LBVH_BVH_H
#define LBVH_BVH_H
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

namespace lbvh
{
namespace detail
{
struct node
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
    using node_type  = detail::node;
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
    using node_type  = detail::node;
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
template<typename Real, typename Object, bool IsConst, typename UInt>
void construct_internal_nodes(const basic_device_bvh<Real, Object, IsConst>& self,
        UInt const* node_code, const unsigned int num_objects)
{
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<unsigned int>(0),
        thrust::make_counting_iterator<unsigned int>(num_objects - 1),
        [self, node_code, num_objects] __host__ __device__ (const unsigned int idx)
        {
            self.nodes[idx].object_idx = 0xFFFFFFFF; //  internal nodes

            const uint2 ij  = determine_range(node_code, num_objects, idx);
            const int gamma = find_split(node_code, num_objects, ij.x, ij.y);

            self.nodes[idx].left_idx  = gamma;
            self.nodes[idx].right_idx = gamma + 1;
            if(thrust::min(ij.x, ij.y) == gamma)
            {
                self.nodes[idx].left_idx += num_objects - 1;
            }
            if(thrust::max(ij.x, ij.y) == gamma + 1)
            {
                self.nodes[idx].right_idx += num_objects - 1;
            }
            self.nodes[self.nodes[idx].left_idx].parent_idx  = idx;
            self.nodes[self.nodes[idx].right_idx].parent_idx = idx;
            return;
        });
    return;
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

template<typename scalar_t>
__global__ void kernel(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> aabb_a,
                       const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> aabb_whole_a,
                       torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> morton_codes_a,
                       const unsigned n_mixtures, const unsigned n_components) {
        const auto mixture_id = blockIdx.x * blockDim.x + threadIdx.x;
        const auto component_id = blockIdx.y * blockDim.y + threadIdx.y;
        if (mixture_id >= n_mixtures || component_id >= n_components)
            return;
        const auto& aabb = reinterpret_cast<const Aabb<float>&>(aabb_a[mixture_id][component_id][0]);
        const auto& whole = reinterpret_cast<const Aabb<float>&>(aabb_whole_a[mixture_id][0]);
        auto& morton_code = reinterpret_cast<uint64_t&>(morton_codes_a[mixture_id][component_id]);

        auto p = centroid(aabb);
        p.x -= whole.lower.x;
        p.y -= whole.lower.y;
        p.z -= whole.lower.z;
        p.x /= (whole.upper.x - whole.lower.x);
        p.y /= (whole.upper.y - whole.lower.y);
        p.z /= (whole.upper.z - whole.lower.z);
        morton_code = lbvh::morton_code(p);

        morton_code <<= 32;
        morton_code |= component_id;
};

torch::Tensor compute_morton_codes(const torch::Tensor& aabbs, const torch::Tensor& aabb_whole) {
    const auto n_components = aabbs.size(-2);
    const auto aabbs_view = aabbs.view({-1, n_components, 8});
    const auto aabb_whole_view = aabb_whole.view({-1, 8});
    const auto aabb_a = aabbs_view.packed_accessor32<float, 3, torch::RestrictPtrTraits>();
    const auto aabb_whole_a = aabb_whole_view.packed_accessor32<float, 2, torch::RestrictPtrTraits>();

    const auto n_mixtures = aabbs_view.size(0);

    auto morton_codes = torch::tensor({n_mixtures, n_components}, torch::TensorOptions(aabbs.device()).dtype(torch::ScalarType::Long));
    auto morton_codes_a = morton_codes.packed_accessor32<float, 2, torch::RestrictPtrTraits>();

    const dim3 dimBlock = dim3(128, 64, 1);
    const dim3 dimGrid = dim3((n_mixtures + dimBlock.x - 1) / dimBlock.x,
                              (n_components + dimBlock.y - 1) / dimBlock.y);
    kernel<<<dimGrid, dimBlock>>>(aabb_a, aabb_whole_a, morton_codes_a, n_mixtures, n_components);
}

std::tuple<torch::Tensor, torch::Tensor> sort_morton_codes(const torch::Tensor& morton_codes, const torch::Tensor& object_aabbs) {
    auto sorted_morton_codes = morton_codes.clone();
    auto sorted_aabbs = object_aabbs.clone();

    // Declare, allocate, and initialize device-accessible pointers for sorting data
    int num_items = morton_codes.numel();                               // e.g., 8
    int num_segments = morton_codes.size(0);                            // e.g., 4
    int num_components = morton_codes.size(1);                          // 2
    const auto offsets = torch::arange(0, num_segments + 1, torch::TensorOptions(morton_codes.device()).dtype(torch::ScalarType::Int)) * num_components;
    int* d_offsets = offsets.data_ptr<int>();                           // e.g., [0, 2, 4, 6, 8]
    const uint64_t* d_keys_in = reinterpret_cast<const uint64_t*>(morton_codes.data_ptr<int64_t>());        // e.g., [8, 6, 7, 5, 3, 0, 9, 8]
    uint64_t* d_keys_out = reinterpret_cast<uint64_t*>(sorted_morton_codes.data_ptr<int64_t>());            // e.g., [-, -, -, -, -, -, -, -]
    const Aabb<float>* d_values_in = reinterpret_cast<const Aabb<float>*>(object_aabbs.data_ptr<float>());  // e.g., [0, 1, 2, 3, 4, 5, 6, 7]
    Aabb<float>* d_values_out = reinterpret_cast<Aabb<float>*>(sorted_aabbs.data_ptr<float>());             // e.g., [-, -, -, -, -, -, -, -]

    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        num_items, num_segments, d_offsets, d_offsets + 1);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out,
        num_items, num_segments, d_offsets, d_offsets + 1);
    // d_keys_out            <-- [6, 8, 5, 7, 0, 3, 8, 9]
    // d_values_out          <-- [1, 0, 3, 2, 5, 4, 7, 6]
    cudaFree(d_temp_storage);

    return std::make_tuple(sorted_morton_codes, sorted_aabbs);
}

} // detail

template<typename Real, typename Object>
struct default_morton_code_calculator
{
    default_morton_code_calculator(Aabb<Real> w): whole(w) {}
    default_morton_code_calculator()  = default;
    ~default_morton_code_calculator() = default;
    default_morton_code_calculator(default_morton_code_calculator const&) = default;
    default_morton_code_calculator(default_morton_code_calculator&&)      = default;
    default_morton_code_calculator& operator=(default_morton_code_calculator const&) = default;
    default_morton_code_calculator& operator=(default_morton_code_calculator&&)      = default;

    __device__ __host__
    inline unsigned int operator()(const Object&, const Aabb<Real>& box) noexcept
    {
        auto p = centroid(box);
        p.x -= whole.lower.x;
        p.y -= whole.lower.y;
        p.z -= whole.lower.z;
        p.x /= (whole.upper.x - whole.lower.x);
        p.y /= (whole.upper.y - whole.lower.y);
        p.z /= (whole.upper.z - whole.lower.z);
        return morton_code(p);
    }
    Aabb<Real> whole;
};

template<typename Real, typename Object>
using  bvh_device = detail::basic_device_bvh<Real, Object, false>;
template<typename Real, typename Object>
using cbvh_device = detail::basic_device_bvh<Real, Object, true>;

template<typename Real, typename Object,
         typename MortonCodeCalculator = default_morton_code_calculator<Real, Object>>
class bvh
{
  public:
    using real_type   = Real;
    using index_type = std::uint32_t;
    using object_type = Object;
    using aabb_type   = Aabb<real_type>;
    using node_type   = detail::node;
    using morton_code_calculator_type = MortonCodeCalculator;

  public:

    bvh(const torch::Tensor& mixture)
        : m_mixture(mixture)
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
        auto n = gpe::get_ns(m_mixture);

        const unsigned int num_objects        = n.components;
        const unsigned int num_internal_nodes = num_objects - 1;
        const unsigned int num_nodes          = num_objects * 2 - 1;

        auto object_aabbs = detail::compute_aabbs(m_mixture);
        const auto aabb_whole = detail::compute_aabb_whole(object_aabbs);

        // --------------------------------------------------------------------
        // calculate morton code of each AABB
        // we produce unique morton codes by extending the actual morton code with the index.
        // TODO: easy, either some scaling and translation using torch + thrust transform, or custom kernel.
        auto morton_codes = detail::compute_morton_codes(object_aabbs, aabb_whole);


        // --------------------------------------------------------------------
        // sort object-indices by morton code
        // TODO: either torch, or http://www.orangeowlsolutions.com/archives/1297 (2 sorts), or, we can prepend the gm index to the morton code?
        //       or, this would be the fastest (probably): https://nvlabs.github.io/cub/structcub_1_1_device_segmented_radix_sort.html

        std::tie(morton_codes, object_aabbs) = detail::sort_morton_codes(morton_codes, object_aabbs);

        // assemble aabb array (internal nodes will be filled later)
        const auto internal_aabbs = torch::zeros({n.batch, n.layers, num_internal_nodes, 8});
        m_aabbs = torch::cat({internal_aabbs, object_aabbs}, 3).contiguous();

//        thrust::device_vector<unsigned int> indices(num_objects);
//        thrust::copy(thrust::make_counting_iterator<index_type>(0),
//                     thrust::make_counting_iterator<index_type>(num_objects),
//                     indices.begin());
//        // keep indices ascending order
//        thrust::stable_sort_by_key(morton.begin(), morton.end(),
//            thrust::make_zip_iterator(
//                thrust::make_tuple(aabbs_.begin() + num_internal_nodes,
//                                   indices.begin())));

//        // --------------------------------------------------------------------
//        // construct leaf nodes and aabbs
//        // TODO: we can build our own kernel here.
//        node_type default_node;
//        default_node.parent_idx = 0xFFFFFFFF;
//        default_node.left_idx   = 0xFFFFFFFF;
//        default_node.right_idx  = 0xFFFFFFFF;
//        default_node.object_idx = 0xFFFFFFFF;
//        this->nodes_.resize(num_nodes, default_node);

//        thrust::transform(indices.begin(), indices.end(),
//            this->nodes_.begin() + num_internal_nodes,
//            [] __device__ (const index_type idx)
//            {
//                node_type n;
//                n.parent_idx = 0xFFFFFFFF;
//                n.left_idx   = 0xFFFFFFFF;
//                n.right_idx  = 0xFFFFFFFF;
//                n.object_idx = idx;
//                return n;
//            });

//        // --------------------------------------------------------------------
//        // construct internal nodes
//        // TODO: can also be done for all at once
//        const auto self = this->get_device_repr();
//        {
//            const unsigned long long int* node_code = morton64.data().get();
//            detail::construct_internal_nodes(self, node_code, num_objects);
//        }

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
private:
    const torch::Tensor m_mixture;
    torch::Tensor m_aabbs;
    torch::Tensor m_nodes;

    thrust::device_vector<object_type>   objects_d_;
    thrust::device_vector<aabb_type>     aabbs_;
    thrust::device_vector<node_type>     nodes_;
};



} // lbvh
#endif// LBVH_BVH_H
