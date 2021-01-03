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

template <int N_REDUCTION, typename scalar_t, int N_FITTING_COMPONENTS = N_REDUCTION>
void runTest(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& testCases, double threshold) {
    using namespace torch::indexing;

    // test specific configuration:
    auto config = bvh_mhem_fit::Config{N_REDUCTION,
                                   lbvh::Config{lbvh::Config::MortonCodeAlgorithm::Old},
                                   bvh_mhem_fit::Config::FitInitialDisparityMethod::CentroidDistance,
                                   bvh_mhem_fit::Config::FitInitialClusterMergeMethod::MaxWeight,
                                   1.5f,
                                   N_FITTING_COMPONENTS};
    auto reference_config = bvh_mhem_fit_alpha::Config{
            config.reduction_n, config.bvh_config,
            bvh_mhem_fit_alpha::Config::FitInitialDisparityMethod(int(config.fit_initial_disparity_method)),
            bvh_mhem_fit_alpha::Config::FitInitialClusterMergeMethod(int(config.fit_initial_cluster_merge_method)),
            config.em_kl_div_threshold, config.n_components_fitting};

//    std::cout << "test error threshold is " << threshold << std::endl;

    for (const auto& test_case : testCases) {
        torch::Tensor mixture = test_case.first;
        torch::Tensor gradient_fitting = test_case.second.repeat({mixture.size(0), mixture.size(1), 1, 1});
        auto reference_mixture = mixture.to(torch::ScalarType::Double);
        auto reference_gradient_fitting = gradient_fitting.to(torch::ScalarType::Double);
        if (sizeof(scalar_t) > 4) {
            mixture = mixture.to(torch::ScalarType::Double);
            gradient_fitting = gradient_fitting.to(torch::ScalarType::Double);
        }

        auto forward_out = bvh_mhem_fit::forward_impl(mixture, config);
        auto gradient_target = bvh_mhem_fit::backward_impl(gradient_fitting, forward_out, config);
        auto reference_forward_out = bvh_mhem_fit_alpha::forward_impl(reference_mixture, reference_config);
        auto reference_gradient_target = bvh_mhem_fit_alpha::backward_impl(reference_gradient_fitting, reference_forward_out, reference_config);

        {
            auto gradient = gradient_target.contiguous();
            auto gradient_reference = reference_gradient_target.contiguous();
            auto n_Gs = size_t(gradient_reference.size(-2));
            for (size_t i = 0; i < n_Gs * 7; ++i) {
                auto similar = are_similar(gradient_reference.data_ptr<double>()[i], double(gradient.data_ptr<scalar_t>()[i]), threshold);
                if (!similar) {
                    std::cout << "target: " << mixture << std::endl;
                    std::cout << "fitting: " << forward_out.fitting << std::endl;
                    std::cout << "gradient target: " << gradient << std::endl;
                    std::cout << "gradient target (reference): " << gradient_reference << std::endl;
                    std::cout << "gradient_fitting: " << gradient_fitting << std::endl;
                    std::cout << "i = " << i << "; difference: " << gradient_reference.data_ptr<double>()[i] - double(gradient.data_ptr<scalar_t>()[i]) << std::endl;
//                    WARN(std::string("grad = ") + std::to_string(gradient.data_ptr<scalar_t>()[i]) + "; ref: " + std::to_string(gradient_reference.data_ptr<double>()[i]));
//                    WARN(std::string("i = ") + std::to_string(i) + "; difference: " + std::to_string(gradient_reference.data_ptr<double>()[i] - double(gradient.data_ptr<scalar_t>()[i])));
                }
                REQUIRE(similar);
            }
        }
    }
}


TEST_CASE( "testing working against alpha reference", "[bvh_mhem_fit]" ) {
    SECTION("2 component fitting double _collectionOf2dMixtures_with4Gs") {
        runTest<2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with4Gs()}),
                           0.0000000000000000001);
    }
    SECTION("2 component fitting double _collectionOf2dMixtures_with8GsForQuickAutoDiff") {
        runTest<2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForQuickAutoDiff()}),
                           0.0000000000000000001);
    }
    SECTION("2 component fitting double _collectionOf2dMixtures_with8GsForLongAutoDiff") {
        runTest<2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForLongAutoDiff()}),
                           0.0000000000000000001);
    }
    SECTION("2 component fitting double _collectionOf2dMixtures_with8GsTooLongForAutodiff") {
        runTest<2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsTooLongForAutodiff()}),
                           0.0000000000000000001);
    }
    SECTION("2 component fitting double _collectionOf2dMixtures_with16Gs") {
        runTest<2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with16Gs()}),
                           0.0000000000000000001);
    }

    SECTION("2 component fitting float _collectionOf2dMixtures_with4Gs") {
        runTest<2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with4Gs()}),
                           0.00002);
    }
    SECTION("2 component fitting float _collectionOf2dMixtures_with8GsForQuickAutoDiff") {
        runTest<2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForQuickAutoDiff()}),
                           0.00002);
    }
    SECTION("2 component fitting float _collectionOf2dMixtures_with8GsForLongAutoDiff") {
        runTest<2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForLongAutoDiff()}),
                           0.00002);
    }
    SECTION("2 component fitting float _collectionOf2dMixtures_with8GsTooLongForAutodiff") {
        runTest<2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsTooLongForAutodiff()}),
                           0.00002);
    }
    SECTION("2 component fitting float _collectionOf2dMixtures_with16Gs") {
        runTest<2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with16Gs()}),
                           0.00003);
    }

    SECTION("2 component fitting float _collectionOf2dMixtures_causingNumericalProblems") {
        runTest<2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                {_collectionOf2dMixtures_causingNumericalProblems()}),
                          0.002);
    }

    SECTION("4 component fitting double") {
        runTest<4, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                  _collectionOf2dMixtures_with8GsForLongAutoDiff(),
                                                                  _collectionOf2dMixtures_with8GsTooLongForAutodiff(),
                                                                  _collectionOf2dMixtures_with16Gs()}),
                           0.0000000000000000001);
    }

    SECTION("4 component fitting float") {
        runTest<4, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                {_collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                 _collectionOf2dMixtures_with8GsForLongAutoDiff(),
                                                                 _collectionOf2dMixtures_with8GsTooLongForAutodiff(),
                                                                 _collectionOf2dMixtures_with16Gs()}),
                          0.00001);
    }

    SECTION("4 component fitting float with numerical problems") {
        runTest<4, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                {_collectionOf2dMixtures_causingNumericalProblems()}),
                          0.002);
    }

    SECTION("32 component fitting double of real world") {
        using namespace torch::indexing;
        std::vector<torch::Tensor> mixtures;
        for (int i = 0; i < 1/*10*/; ++i) {
            torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");

            for (uint i = 0; i < 3; i++) {
                auto mixture = container.attr(std::to_string(i)).toTensor();
                mixture = mixture.index({Slice(0,2), Slice(0,2), Slice(), Slice()});
                mixture = gpe::pack_mixture(torch::abs(gpe::weights(mixture)), gpe::positions(mixture), gpe::covariances(mixture));
                mixtures.push_back(mixture);
            }
        }

        runTest<4, double, 32>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d32GsGrads()}, {mixtures}), 0.0000000000000000001);
    }
}
