#include <iostream>
#include <string>

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/catch.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <cuda_runtime.h>

#include "common.h"
#include "bvh_mhem_fit/implementation.h"
#include "evaluate_inversed/parallel_binding.h"
#include "integrate/binding.h"
#include "util/mixture.h"

constexpr uint N_BATCHES = 1;
constexpr uint CONVOLUTION_LAYER_START = 0;
constexpr uint CONVOLUTION_LAYER_END = 3;
constexpr uint LIMIT_N_BATCH = 100;
constexpr bool USE_CUDA = true;
constexpr bool BACKWARD = false;
constexpr bool RENDER = true;
constexpr uint RESOLUTION = 128;
constexpr bool DO_STATS = false;
constexpr uint N_FITTING_COMPONENTS = 32;


TEST_CASE("bvh_mhem_fit forward and backward benchmark") {
    using namespace torch::indexing;

    std::array<std::vector<torch::Tensor>, CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START> error_data;
    std::array<std::vector<std::chrono::milliseconds>, CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START> time_data;

    // test specific configuration:
#ifndef GPE_LIMIT_N_REDUCTION
    std::vector<int> reduction_n_options = {16};
#else
    std::vector<int> reduction_n_options = {4};
#endif
    std::vector<lbvh::Config::MortonCodeAlgorithm> morton_code_options = {
        lbvh::Config::MortonCodeAlgorithm::Old
    };
    std::vector<float> em_kl_div_threshold_options {0.5f};


    std::vector<std::pair<std::string, bvh_mhem_fit::Config>> configs;
    for (auto reduction_n : reduction_n_options) {
        for (auto morton_code_algorithm : morton_code_options) {
            for (auto em_kl_div_threshold : em_kl_div_threshold_options) {
//                configs.emplace_back("red_" + std::to_string(reduction_n) +
//                                     "_morton_" + std::to_string(int(morton_code_algorithm)) +
//                                     "_emkldivth_" + std::to_string(em_kl_div_threshold),
//                                     bvh_mhem_fit::Config{reduction_n, lbvh::Config{morton_code_algorithm}, em_kl_div_threshold});
                configs.emplace_back(std::to_string(reduction_n) +
                                     ", " + std::to_string(int(morton_code_algorithm)) +
                                     ", " + std::to_string(int(em_kl_div_threshold * 10)),
                                     bvh_mhem_fit::Config{reduction_n, lbvh::Config{morton_code_algorithm}, em_kl_div_threshold, N_FITTING_COMPONENTS});
            }
        }
    }

    for (const auto& named_config : configs) {
        for (uint i = 0; i < N_BATCHES; ++i) {
            torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");
            auto list = container.attributes();

            for (uint i = 0; i < CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START; i++) {
                assert(i + CONVOLUTION_LAYER_START < 3);
                auto mixture = container.attr(std::to_string(i + CONVOLUTION_LAYER_START)).toTensor();
//                mixture = mixture.index({Slice(0,10), Slice(), Slice(), Slice()});
                mixture = gpe::pack_mixture(torch::abs(gpe::weights(mixture)), gpe::positions(mixture), gpe::covariances(mixture));
                if (USE_CUDA)
                    mixture = mixture.cuda();
//                std::cout << "layer " << i + CONVOLUTION_LAYER_START << ": " << mixture.sizes() << " device: " << mixture.device() << std::endl;
//                auto t0 = std::chrono::high_resolution_clock::now();

//                cudaDeviceSynchronize();

                BENCHMARK("forward") {
                    auto forward_out = bvh_mhem_fit::forward_impl(mixture, named_config.second);
                    cudaDeviceSynchronize();
                    return forward_out;
                };
                auto forward_out = bvh_mhem_fit::forward_impl(mixture, named_config.second);
                auto gradient_fitting = torch::ones_like(forward_out.fitting);
                cudaDeviceSynchronize();

                BENCHMARK("backward") {
                    auto gradient_target = bvh_mhem_fit::backward_impl(gradient_fitting, forward_out.clone(), named_config.second);
                    cudaDeviceSynchronize();
                    return gradient_target;
                };
            }
        }
    }
}
