#include "evaluate_inversed/evaluate_inversed.h"

#include <iostream>
#include <string>

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include <torch/types.h>
#include <torch/script.h>

#include <cuda_runtime.h>

#include "common.h"
#include "util/mixture.h"

struct use_cuda_type {
    constexpr static bool value = true;
};

struct use_cpu_type {
    constexpr static bool value = false;
};

constexpr uint N_CONVOLUTION_LAYERS = 3;

namespace {
template<bool USE_CUDA, typename ForwardFun, typename BackwardFun>
void run(torch::Tensor mixture, const std::string& name, ForwardFun forward_fun, BackwardFun backward_fun) {
    if (USE_CUDA)
        mixture = mixture.cuda();

    const auto weights = gpe::weights(mixture);
    torch::Tensor positions = gpe::positions(mixture);
    const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
    mixture = gpe::pack_mixture(weights, positions, invCovs.contiguous()).clone().contiguous();
    cudaDeviceSynchronize();

    BENCHMARK(name + "_forward") {
        auto eval_values = forward_fun(mixture, positions.contiguous());
        cudaDeviceSynchronize();
        return eval_values;
    };
    auto forward_out = forward_fun(mixture, positions.contiguous());

    torch::Tensor positions_clone = gpe::positions(mixture).clone();
    auto grad_out = torch::rand_like(std::get<0>(forward_out));

    cudaDeviceSynchronize();
    BENCHMARK(name + "_backward") {
        auto grads = backward_fun(grad_out, mixture, positions_clone, forward_out, true, true);
        cudaDeviceSynchronize();
        return grads;
    };
}
}


TEMPLATE_TEST_CASE("evaluate inversed forward and backward benchmark parallel", "[evaluate_inversed]", use_cuda_type) {
    using namespace torch::indexing;
    torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/after_fixed_point_batch0.pt");
    auto list = container.attributes();

    for (uint i = 0; i < N_CONVOLUTION_LAYERS; i++) {
        run<TestType::value>(container.attr(std::to_string(i)).toTensor(), "layer_" + std::to_string(i), evaluate_inversed::parallel_forward, evaluate_inversed::parallel_backward);
    }
};

TEST_CASE("evaluate inversed forward and backward benchmark bvh", "[evaluate_inversed]") {
    torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/after_fixed_point_batch0.pt");
    auto list = container.attributes();

    for (uint i = 0; i < N_CONVOLUTION_LAYERS; i++) {
        run<use_cuda_type::value>(container.attr(std::to_string(i)).toTensor(), "layer_" + std::to_string(i), evaluate_inversed::cuda_bvh_forward, evaluate_inversed::cuda_bvh_backward);
    }
};
