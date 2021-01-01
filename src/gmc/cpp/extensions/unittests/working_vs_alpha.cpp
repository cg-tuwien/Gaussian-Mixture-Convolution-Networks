#include <iostream>
#include <string>

#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/script.h>


#include "bvh_mhem_fit/implementation.h"
#include "bvh_mhem_fit_alpha/implementation.h"
#include "common.h"
#include "evaluate_inversed/parallel_binding.h"
#include "integrate/binding.h"
#include "util/mixture.h"
#include "unittests/support.h"

constexpr uint N_FITTING_COMPONENTS = 2;

TEST_CASE( "testing working against alpha", "[bvh_mhem_fit]" ) {
    using namespace torch::indexing;
    using scalar_t = double;

    // test specific configuration:
    auto config = bvh_mhem_fit::Config{2,
                                   lbvh::Config{lbvh::Config::MortonCodeAlgorithm::Old},
                                   bvh_mhem_fit::Config::FitInitialDisparityMethod::CentroidDistance,
                                   bvh_mhem_fit::Config::FitInitialClusterMergeMethod::MaxWeight,
                                   20.5f,
                                   N_FITTING_COMPONENTS};
    auto reference_config = bvh_mhem_fit_alpha::Config{
            config.reduction_n, config.bvh_config,
            bvh_mhem_fit_alpha::Config::FitInitialDisparityMethod(int(config.fit_initial_disparity_method)),
            bvh_mhem_fit_alpha::Config::FitInitialClusterMergeMethod(int(config.fit_initial_cluster_merge_method)),
            config.em_kl_div_threshold, config.n_components_fitting};

    const auto threshold = scalar_t(0.0001) / (1 + (sizeof(scalar_t) - 4) * 25000);
//    std::cout << "test error threshold is " << threshold << std::endl;

    auto testCases = _combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                           {_collectionOf2dMixtures_with4Gs(),
                                                            _collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                            _collectionOf2dMixtures_with8GsForLongAutoDiff(),
                                                           _collectionOf2dMixtures_with8GsTooLongForAutodiff(),
                                                           _collectionOf2dMixtures_with16Gs()});
    for (const auto& test_case : testCases) {
        torch::Tensor mixture = test_case.first;
        torch::Tensor gradient_fitting = test_case.second;
        if (sizeof(scalar_t) > 4) {
            mixture = mixture.to(torch::ScalarType::Double);
            gradient_fitting = gradient_fitting.to(torch::ScalarType::Double);
        }

        auto forward_out = bvh_mhem_fit::forward_impl(mixture, config);
        auto gradient_target = bvh_mhem_fit::backward_impl(gradient_fitting, forward_out, config);
        auto reference_forward_out = bvh_mhem_fit_alpha::forward_impl(mixture, reference_config);
        auto reference_gradient_target = bvh_mhem_fit_alpha::backward_impl(gradient_fitting, reference_forward_out, reference_config);

        {
            auto gradient = gradient_target.contiguous();
            auto gradient_reference = reference_gradient_target.contiguous();
            auto n_Gs = size_t(gradient_reference.size(-2));
            for (size_t i = 0; i < n_Gs * 7; ++i) {
                auto similar = are_similar(gradient_reference.data_ptr<scalar_t>()[i], gradient.data_ptr<scalar_t>()[i], threshold);
                if (!similar) {
                    std::cout << "target: " << mixture << std::endl;
                    std::cout << "fitting: " << forward_out.fitting << std::endl;
                    std::cout << "gradient target: " << gradient << std::endl;
                    std::cout << "gradient target (reference): " << gradient_reference << std::endl;
                    std::cout << "gradient_fitting: " << gradient_fitting << std::endl;
                    std::cout << "i = " << i << "; difference: " << gradient_reference.data_ptr<scalar_t>()[i] - gradient.data_ptr<scalar_t>()[i] << std::endl;
                }
                REQUIRE(similar);
            }
        }
    }
}
