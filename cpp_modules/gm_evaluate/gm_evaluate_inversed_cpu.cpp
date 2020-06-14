#include <torch/extension.h>

#include <vector>
#include <algorithm>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

#include "common.h"


//torch::Tensor evaluate_inversed_forward(
//    torch::Tensor mixture,
//    torch::Tensor xes) {
//    using namespace torch::indexing;

//    gm::check_mixture(mixture);

//    auto n_batch = gm::n_batch(mixture);
//    auto n_layers = gm::n_layers(mixture);
//    auto n_components = gm::n_components(mixture);
//    auto n_dims = gm::n_dimensions(mixture);

//    TORCH_CHECK(xes.dim() == 4, "xes must have 4 dimensions");
//    TORCH_CHECK(xes.size(0) == 1 || xes.size(0) == n_batch, "xes must have a batch dimension of size 1 or of size equal to the mixture");
//    TORCH_CHECK(xes.size(1) == 1 || xes.size(1) == n_layers, "xes must have a layer dimension of size 1 or of size equal to the mixture");

//    auto n_xes = xes.size(2);
//    TORCH_CHECK(xes.size(3) == n_dims, "xes must have the last dimension equal to the number of dimensions of the mixture");

//    xes = xes.view({xes.size(0), xes.size(1), 1, n_xes, n_dims});
//    torch::Tensor values_sum = torch::zeros({n_batch, n_layers, n_xes}, torch::dtype(torch::kFloat32).device(mixture.device()));

//    int64_t total_memory_space = n_batch * n_layers * n_components * n_xes * n_dims;  //# did i forget something?
//    int64_t n_memory_slices = std::max(total_memory_space / (1024 * 1024 * 200), int64_t(1));
//    int64_t comp_slice_size = std::max(n_components / n_memory_slices, int64_t(1));
//    n_memory_slices = n_components / comp_slice_size + int(n_components % comp_slice_size != 0);

//    for (int64_t i = 0; i < n_memory_slices; ++i) {
//        int64_t comps_begin = i * comp_slice_size;
//        int64_t comps_end = std::min(comps_begin + comp_slice_size, n_components);
//        int64_t n_comps_slice = comps_end - comps_begin;

//        torch::Tensor mixture_slice = mixture.index({Slice(), Slice(), Slice(comps_begin, comps_end), Slice()});
//        torch::Tensor values = xes - gm::positions(mixture_slice).view({n_batch, n_layers, n_comps_slice, 1, n_dims});

//        // x^t A x -> quadratic form
//        torch::Tensor x_t = values.view({n_batch, n_layers, n_comps_slice, -1, 1, n_dims});
//        torch::Tensor x = values.view({n_batch, n_layers, n_comps_slice, -1, n_dims, 1});
//        torch::Tensor A = gm::covariances(mixture_slice).view({n_batch, n_layers, n_comps_slice, 1, n_dims, n_dims});
//        values = -0.5 * x_t.matmul(A).matmul(x);
//        values = values.view({n_batch, n_layers, n_comps_slice, -1});

//        values = gm::weights(mixture_slice).view({n_batch, n_layers, n_comps_slice, 1}) * torch::exp(values);
//        values_sum += values.sum(2);
//    }

//    return values_sum;
//}


// http://www.machinedlearnings.com/2011/06/fast-approximate-logarithm-exponential.html
// https://github.com/xodobox/fastapprox/blob/master/fastapprox/src/fastexp.h
// 2x faster, error in the range of e^-4 (dunno about relativ error)
static inline float fasterpow2 (float p)
{
  float clipp = (p < -126) ? -126.0f : p;
  union { uint32_t i; float f; } v = { uint32_t ( (1 << 23) * (clipp + 126.94269504f) ) };
  return v.f;
}

static inline float fasterexp (float p)
{
  return fasterpow2 (1.442695040f * p);
}

// slightly faster than std::exp, slightly less precise (error in the range of e-10)
static inline float
fastpow2 (float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = int(clipp);
  float z = clipp - w + offset;
  union { uint32_t i; float f; } v = { uint32_t ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

static inline float
fastexp (float p)
{
  return fastpow2 (1.442695040f * p);
}


struct Ns {
    int batch = 0;
    int layers = 0;
    int components = 0;
    int dims = 0;
    int batch_xes = 0;
    int layers_xes = 0;
    int xes = 0;
};

template <typename scalar_t, int DIMS>
void execute_parallel_forward(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>& mixture_a,
                      const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>& xes_a,
                      torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>& sum_a,
                      const Ns& n) {

    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < n.batch * n.layers * n.xes; ++i) {
        auto batch_layer_index = i / n.xes;
        auto xes_index = i % n.xes;

        const glm::vec<2, scalar_t>& x_pos = reinterpret_cast<const glm::vec<2, scalar_t>&>(xes_a[batch_layer_index][xes_index][0]);
    //            scalar_t& qx = xes_a[batch_layer_index][xes_index][0]; // todo: 3d
    //            scalar_t& qy = xes_a[batch_layer_index][xes_index][1]; // todo: 3d

        scalar_t& sum = sum_a[batch_layer_index][xes_index];
        for (int c = 0; c < n.components; ++c) {
            const scalar_t& c_weight = mixture_a[batch_layer_index][c][0];
            const glm::vec<2, scalar_t>& c_pos = reinterpret_cast<const glm::vec<2, scalar_t>&>(mixture_a[batch_layer_index][c][1]);
            const glm::mat<2, 2, scalar_t>& c_cov = reinterpret_cast<const glm::mat<2, 2, scalar_t>&>(mixture_a[batch_layer_index][c][3]);
            auto t = x_pos - c_pos;
            auto v = scalar_t(-0.5) * glm::dot(t, (c_cov * t));
            sum += c_weight * std::exp(v);
            /// TODO: templetise for 2d/3d
        }

    }
}

Ns check_input_and_get_ns(torch::Tensor mixture, torch::Tensor xes) {

    gm::check_mixture(mixture);

    int n_batch = gm::n_batch(mixture);
    int n_layers = gm::n_layers(mixture);
    int n_components = gm::n_components(mixture);
    int n_dims = gm::n_dimensions(mixture);

    TORCH_CHECK(xes.dim() == 4, "xes must have 4 dimensions");
    TORCH_CHECK(xes.dtype() == mixture.dtype(), "mixture and xes must have the same dtype");
    TORCH_CHECK(xes.device() == mixture.device(), "mixture and xes must have the same device");
    int n_batch_xes = int(xes.size(0));
    int n_layers_xes = int(xes.size(1));
    int n_xes = int(xes.size(2));

    TORCH_CHECK(/*n_batch_xes == 1 || */n_batch_xes == n_batch, "xes must have a batch dimension equal to the mixture");
    TORCH_CHECK(/*n_layers_xes == 1 || */n_layers_xes == n_layers, "xes must have a layer dimension equal to the mixture");
    TORCH_CHECK(xes.size(3) == n_dims, "xes must have the last dimension equal to the number of dimensions of the mixture");
    return {n_batch, n_layers, n_components, n_dims, n_batch_xes, n_layers_xes, n_xes};
}

torch::Tensor evaluate_inversed_forward(torch::Tensor mixture, torch::Tensor xes) {
    using namespace torch::indexing;
    auto n = check_input_and_get_ns(mixture, xes);

    mixture = mixture.view({n.batch * n.layers, n.components, -1});
    xes = xes.view({n.batch_xes * n.layers_xes, n.xes, n.dims});
    torch::Tensor sum = torch::zeros({n.batch * n.layers, n.xes}, torch::dtype(mixture.dtype()).device(mixture.device()));

    TORCH_CHECK(mixture.device().is_cpu(), "this one is just for cpu..");


    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "eval_inversed_omp", ([&] {
        auto mixture_a = mixture.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto xes_a = xes.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto sum_a = sum.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();

        execute_parallel_forward<scalar_t, 2>(mixture_a, xes_a, sum_a, n);
    }));
    return sum.view({n.batch, n.layers, n.xes});
}


template <typename scalar_t, int DIMS>
void execute_parallel_backward(const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>& mixture_a,
                      const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>& xes_a,
                      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>& grad_mixture_a,
                      torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits>& grad_xes_a,
                      torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>& grad_output_a,
                      const Ns& n, bool requires_grad_mixture, bool requires_grad_xes) {

    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < n.batch * n.layers * n.xes; ++i) {
        auto batch_layer_index = i / n.xes;
        auto xes_index = i % n.xes;

        const glm::vec<2, scalar_t>& x_pos = reinterpret_cast<const glm::vec<2, scalar_t>&>(xes_a[batch_layer_index][xes_index][0]);
    //            scalar_t& qx = xes_a[batch_layer_index][xes_index][0]; // todo: 3d
    //            scalar_t& qy = xes_a[batch_layer_index][xes_index][1]; // todo: 3d

        glm::vec<2, scalar_t>& grad_xes = reinterpret_cast<glm::vec<2, scalar_t>&>(grad_xes_a[batch_layer_index][xes_index][0]);
        for (int c = 0; c < n.components; ++c) {
            scalar_t& grad_c_weight = grad_mixture_a[batch_layer_index][c][0];
            glm::vec<2, scalar_t>& grad_c_pos = reinterpret_cast<glm::vec<2, scalar_t>&>(grad_mixture_a[batch_layer_index][c][1]);
            glm::mat<2, 2, scalar_t>& grad_c_cov = reinterpret_cast<glm::mat<2, 2, scalar_t>&>(grad_mixture_a[batch_layer_index][c][3]);

            const scalar_t& c_weight = mixture_a[batch_layer_index][c][0];
            const glm::vec<2, scalar_t>& c_pos = reinterpret_cast<const glm::vec<2, scalar_t>&>(mixture_a[batch_layer_index][c][1]);
            const glm::mat<2, 2, scalar_t>& c_cov = reinterpret_cast<const glm::mat<2, 2, scalar_t>&>(mixture_a[batch_layer_index][c][3]);

            const auto t = x_pos - c_pos;
            const auto v = scalar_t(-0.5) * glm::dot(t, (c_cov * t));
            const auto exp = std::exp(v);
            const auto weighted_exp = c_weight * exp;
            const auto local_grad_c_pos = weighted_exp * t * c_cov;

            if (requires_grad_xes) {
                grad_xes += -local_grad_c_pos;
            }
            if (requires_grad_mixture) {
                grad_c_weight += exp * grad_output_a[batch_layer_index][xes_index];
                grad_c_pos += local_grad_c_pos * grad_output_a[batch_layer_index][xes_index];
                grad_c_cov += - c_weight * scalar_t(0.5) * exp * grad_output_a[batch_layer_index][xes_index] * glm::outerProduct(t, t);
            }
            /// TODO: templetise for 2d/3d
        }
        grad_xes *= grad_output_a[batch_layer_index][xes_index];
    }
}

std::vector<torch::Tensor> evaluate_inversed_backward(torch::Tensor grad_output, torch::Tensor mixture, torch::Tensor xes, bool requires_grad_mixture, bool requires_grad_xes) {
    gm::check_mixture(mixture);
    auto n = check_input_and_get_ns(mixture, xes);

    TORCH_CHECK(mixture.device().is_cpu(), "this one is just for cpu..");
    TORCH_CHECK(grad_output.device().is_cpu(), "grad_output must be on cpu..");
    TORCH_CHECK(grad_output.dim() == 3, "grad_output has wrong number of dimensions");
    TORCH_CHECK(grad_output.size(0) == n.batch, "grad_output has wrong batch dimension");
    TORCH_CHECK(grad_output.size(1) == n.layers, "grad_output has wrong layer dimension");
    TORCH_CHECK(grad_output.size(2) == n.xes, "grad_output has wrong xes dimension");
    TORCH_CHECK(grad_output.dtype() == mixture.dtype(), "grad_output dtype does not match with mixture dtype")


    torch::Tensor grad_mixture = torch::zeros({n.batch * n.layers, n.components, mixture.size(3)}, torch::dtype(mixture.dtype()).device(mixture.device()));
    torch::Tensor grad_xes = torch::zeros({n.batch_xes * n.layers_xes, n.xes, n.dims}, torch::dtype(mixture.dtype()).device(mixture.device()));

    grad_output = grad_output.view({n.batch * n.layers, n.xes});
    mixture = mixture.view({n.batch * n.layers, n.components, -1});
    xes = xes.view({n.batch_xes * n.layers_xes, n.xes, n.dims});

    AT_DISPATCH_FLOATING_TYPES(mixture.scalar_type(), "eval_inversed_omp_backward", ([&] {
        auto mixture_a = mixture.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto xes_a = xes.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_mixture_a = grad_mixture.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_xes_a = grad_xes.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>();
        auto grad_output_a = grad_output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>();

        execute_parallel_backward<scalar_t, 2>(mixture_a, xes_a, grad_mixture_a, grad_xes_a, grad_output_a, n, requires_grad_mixture, requires_grad_xes);
    }));

    return {grad_mixture.view({n.batch, n.layers, n.components, -1}), grad_xes.view({n.batch_xes, n.layers_xes, n.xes, n.dims})};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &evaluate_inversed_forward, "evaluate_inversed forward");
  m.def("backward", &evaluate_inversed_backward, "evaluate_inversed backward");
}
