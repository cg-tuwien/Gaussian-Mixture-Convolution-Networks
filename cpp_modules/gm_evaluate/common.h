#ifndef COMMON_H
#define COMMON_H

#include <torch/extension.h>
#include <vector>

#define GLM_FORCE_INLINE
#include <glm/glm.hpp>

namespace gm {


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

int n_batch(torch::Tensor mixture) {
    return mixture.size(0);
}

int n_layers(torch::Tensor mixture) {
    return mixture.size(1);
}

int n_components(torch::Tensor mixture) {
    return mixture.size(2);
}

int n_dimensions(torch::Tensor mixture) {
    auto vector_length = mixture.size(3);
    if (vector_length == 7)
        return 2;
    if (vector_length == 13)
        return 3;

    TORCH_CHECK(false, "mixture must have 7 or 13 elements in the last dimension")
    return 0;
}

torch::Tensor weights(torch::Tensor mixture) {
    using namespace torch::indexing;
    return mixture.index({Slice(), Slice(), Slice(), 0});
}

torch::Tensor positions(torch::Tensor mixture) {
    using namespace torch::indexing;
    return mixture.index({Slice(), Slice(), Slice(), Slice(1, n_dimensions(mixture) + 1)});
}

torch::Tensor covariances(torch::Tensor mixture) {
    using namespace torch::indexing;
    auto n_dims = n_dimensions(mixture);
    std::vector<int64_t> new_shape = mixture.sizes().vec();
    new_shape.back() = n_dims;
    new_shape.push_back(n_dims);

    return mixture.index({Slice(), Slice(), Slice(), Slice(n_dimensions(mixture) + 1, None)}).view(new_shape);
}


template <int DIMS, typename scalar_t>
typename std::conditional<std::is_const<scalar_t>::value, const glm::vec<DIMS, std::remove_cv_t<scalar_t>>, glm::vec<DIMS, scalar_t>>::type& vec(scalar_t& memory_location) {
    return reinterpret_cast<typename std::conditional<std::is_const<scalar_t>::value, const glm::vec<DIMS, std::remove_cv_t<scalar_t>>, glm::vec<DIMS, scalar_t>>::type&>(memory_location);
}

template <int DIMS, typename scalar_t>
typename std::conditional<std::is_const<scalar_t>::value, const glm::mat<DIMS, DIMS, std::remove_cv_t<scalar_t>>, glm::mat<DIMS, DIMS, scalar_t>>::type& mat(scalar_t& memory_location) {
    return reinterpret_cast<typename std::conditional<std::is_const<scalar_t>::value, const glm::mat<DIMS, DIMS, std::remove_cv_t<scalar_t>>, glm::mat<DIMS, DIMS, scalar_t>>::type&>(memory_location);
}


template <typename Gaussian>
auto weight(Gaussian gaussian) -> decltype (gaussian[0]) {
    return gaussian[0];
}

template <int DIMS, typename Gaussian>
auto position(Gaussian gaussian) -> decltype (vec<DIMS>(gaussian[1])) {
    return vec<DIMS>(gaussian[1]);
}

template <int DIMS, typename Gaussian>
auto covariance(Gaussian gaussian) -> decltype (mat<DIMS>(gaussian[1 + DIMS])) {
    return mat<DIMS>(gaussian[1 + DIMS]);
}

void check_mixture(torch::Tensor mixture) {
    TORCH_CHECK(!torch::isnan(mixture).any().item<bool>(), "mixture contains NaNs");
    TORCH_CHECK(!torch::isinf(mixture).any().item<bool>(), "mixture contains infinities");
    TORCH_CHECK(mixture.dim() == 4, "mixture must have 4 dimensions");
    auto n_dims = n_dimensions(mixture);
    TORCH_CHECK(n_dims == 2 || n_dims == 3);
    TORCH_CHECK(torch::all(covariances(mixture).det() > 0).item<bool>(), "mixture contains non positive definite covariances");
}


gm::Ns check_input_and_get_ns(torch::Tensor mixture, torch::Tensor xes) {

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

}

#endif // COMMON_H
