#include <iostream>
#include <string>

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <cuda_runtime.h>

#include "common.h"
#include "convolution/implementation.h"
#include "convolution_fitting/implementation.h"
#include "evaluate_inversed/evaluate_inversed.h"
#include "util/mixture.h"

constexpr uint N_BATCHES = 1;
constexpr uint CONVOLUTION_LAYER_START = 0;
constexpr uint CONVOLUTION_LAYER_END = 4;
constexpr uint LIMIT_N_BATCH = 5;
constexpr bool USE_CUDA = true;
constexpr bool BACKWARD = true;
constexpr bool RENDER = true;
constexpr uint RESOLUTION = 128;
constexpr bool DO_STATS = true;
constexpr uint N_FITTING_COMPONENTS = 8;


torch::Tensor toPdfMixture(const torch::Tensor& data) {
    const auto weights = pieces::integrate(data.view({-1, 1, 1, data.size(-1)})).contiguous().view({data.size(0), data.size(1), data.size(2)});
    return gpe::pack_mixture(weights, gpe::positions(data), gpe::covariances(data));
}

torch::Tensor toAmplitudeMixture(const torch::Tensor& data) {
    const auto uniAmpM = gpe::pack_mixture(torch::ones_like(gpe::weights(data)), gpe::positions(data), gpe::covariances(data));
    const auto normFactors = pieces::integrate(uniAmpM.view({-1, 1, 1, data.size(-1)})).contiguous().view({data.size(0), data.size(1), data.size(2)});
    return gpe::pack_mixture(gpe::weights(data) / normFactors, gpe::positions(data), gpe::covariances(data));
}

TEST_CASE("convolution_fitting forward and backward benchmark") {
    using namespace torch::indexing;

    for (uint b = 0; b < N_BATCHES; ++b) {
        torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/mnist_intermediate_data/conv_inputs_" + std::to_string(b) + ".pt");
        auto list = container.attributes();

        for (uint l = 0; l < CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START; l++) {
            torch::Tensor data = container.attr("conv_layer_" + std::to_string(l) + "_data").toTensor().contiguous();
            torch::Tensor kernels = container.attr("conv_layer_" + std::to_string(l) + "_kernels").toTensor().contiguous();
            if (USE_CUDA) {
                data = data.cuda();
                kernels = kernels.cuda();
            }

            convolution_fitting::Config config;
            config.n_components_fitting = unsigned(data.size(2));
            std::cout << "layer " << l << " data: " << data.sizes() << " device: " << data.device() << std::endl;
            std::cout << "layer " << l << " kernels: " << kernels.sizes() << " device: " << kernels.device() << std::endl;
            std::cout << "target number of gaussians: " << data.size(1) * data.size(2) * kernels.size(2) << ", fitting number of gaussians: " << config.n_components_fitting << std::endl;

            data = toPdfMixture(data);
            kernels = toPdfMixture(kernels);

            BENCHMARK("forward layer " + std::to_string(l)) {
                const auto forward_output = convolution_fitting::forward_impl(data, kernels, config);
                cudaDeviceSynchronize();
                return forward_output;
            };
            const auto forward_output = convolution_fitting::forward_impl(data, kernels, config);
            auto gradient_fitting = torch::ones_like(forward_output.fitting);
            cudaDeviceSynchronize();

            BENCHMARK("backward layer " + std::to_string(l)) {
                auto gradient_target = convolution_fitting::backward_impl(torch::rand_like(forward_output.fitting), data, kernels, forward_output, config);
                cudaDeviceSynchronize();
                return gradient_target;
            };
        }
    }
}
