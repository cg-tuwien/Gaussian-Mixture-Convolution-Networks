#include <iostream>
#include <string>

#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/script.h>


#include "bvh_mhem_fit_alpha/implementation.h"
#include "bvh_mhem_fit_alpha/implementation_autodiff_backward.h"
#include "common.h"
#include "evaluate_inversed/parallel_binding.h"
#include "integrate/binding.h"
#include "util/mixture.h"

#include "support.h"

constexpr uint N_FITTING_COMPONENTS = 2;






TEST_CASE( "testing alpha against autodiff", "[bvh_mhem_fit]" ) {
    using namespace torch::indexing;
    using scalar_t = double;

    // test specific configuration:
    auto alpha_config = bvh_mhem_fit_alpha::Config{2,
                                   lbvh::Config{lbvh::Config::MortonCodeAlgorithm::Old},
                                   bvh_mhem_fit_alpha::Config::FitInitialDisparityMethod::CentroidDistance,
                                   bvh_mhem_fit_alpha::Config::FitInitialClusterMergeMethod::MaxWeight,
                                   20.5f,
                                   N_FITTING_COMPONENTS};

    const auto threshold = scalar_t(0.0001) / (1 + (sizeof(scalar_t) - 4) * 25000);
//    std::cout << "test error threshold is " << threshold << std::endl;

    auto testCases = _combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                           {_collectionOf2dMixtures_with4Gs(),
                                                            _collectionOf2dMixtures_with8GsForQuickAutoDiff()
                                                            /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/});
    for (const auto& test_case : testCases) {
        torch::Tensor mixture = test_case.first;
        torch::Tensor gradient_fitting = test_case.second;
        if (sizeof(scalar_t) > 4) {
            mixture = mixture.to(torch::ScalarType::Double);
            gradient_fitting = gradient_fitting.to(torch::ScalarType::Double);
        }

//        auto start = std::chrono::high_resolution_clock::now();
        auto autodiff_out = bvh_mhem_fit_alpha::implementation_autodiff_backward<2, scalar_t, 2>(mixture, gradient_fitting, alpha_config);
//        auto end = std::chrono::high_resolution_clock::now();
        auto forward_out = bvh_mhem_fit_alpha::forward_impl(mixture, alpha_config);
        auto gradient_target = bvh_mhem_fit_alpha::backward_impl(gradient_fitting, forward_out, alpha_config);

        {
            auto gradient_an = gradient_target.contiguous();
            auto gradient_ad = autodiff_out.mixture_gradient.contiguous();
            auto n_Gs = size_t(gradient_ad.size(-2));
            for (size_t i = 0; i < n_Gs * 7; ++i) {
                auto similar = are_similar(gradient_ad.data_ptr<scalar_t>()[i], gradient_an.data_ptr<scalar_t>()[i], threshold);
                if (!similar) {
                    std::cout << "target: " << mixture << std::endl;
                    std::cout << "fitting: " << forward_out.fitting << std::endl;
                    std::cout << "gradient target (analytical): " << gradient_target << std::endl;
                    std::cout << "gradient target (autodiff): " << autodiff_out.mixture_gradient << std::endl;
                    std::cout << "gradient_fitting: " << gradient_fitting << std::endl;
                    std::cout << "i = " << i << "; difference: " << gradient_ad.data_ptr<scalar_t>()[i] - gradient_an.data_ptr<scalar_t>()[i] << std::endl;
                    bvh_mhem_fit_alpha::implementation_autodiff_backward<2, scalar_t, 2>(mixture, gradient_fitting, alpha_config);
                }
                REQUIRE(similar);
            }
        }
//        std::cout << "test " << ++i << "/" << testCases.size() << " finished (ad gradient took " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms)" << std::endl;

    }
}
