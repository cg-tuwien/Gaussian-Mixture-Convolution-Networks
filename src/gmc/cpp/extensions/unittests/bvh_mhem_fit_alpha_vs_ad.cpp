#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include <torch/torch.h>
#include <torch/script.h>

#include "bvh_mhem_fit_alpha/implementation.h"
#include "bvh_mhem_fit_alpha/implementation_autodiff_backward.h"
#include "common.h"
#include "evaluate_inversed/evaluate_inversed.h"
#include "unittests/support.h"
#include "util/mixture.h"


namespace {
template <int N_REDUCTION, typename scalar_t>
void runTest(scalar_t threshold, const std::vector<std::pair<torch::Tensor, torch::Tensor>>& testCases, unsigned fitting_n_mixtures = N_REDUCTION) {
    auto alpha_config = bvh_mhem_fit_alpha::Config{N_REDUCTION,
                                   lbvh::Config{lbvh::Config::MortonCodeAlgorithm::Old},
                                   20.5f,
                                   fitting_n_mixtures};
    for (const auto& test_case : testCases) {
        torch::Tensor mixture = test_case.first;
        torch::Tensor gradient_fitting = test_case.second;
        if (sizeof(scalar_t) > 4) {
            mixture = mixture.to(torch::ScalarType::Double);
            gradient_fitting = gradient_fitting.to(torch::ScalarType::Double);
        }

        auto autodiff_out = bvh_mhem_fit_alpha::implementation_autodiff_backward<N_REDUCTION, scalar_t, 2>(mixture, gradient_fitting, alpha_config);
        auto forward_out = bvh_mhem_fit_alpha::forward_impl(mixture, alpha_config);
        auto gradient_target = bvh_mhem_fit_alpha::backward_impl(gradient_fitting, forward_out, alpha_config);
//        std::cout << "bvh nodes: " << forward_out.bvh_nodes << std::endl;

        auto gradient_an = gradient_target.contiguous();
        auto gradient_ad = autodiff_out.mixture_gradient.contiguous();
        auto n_Gs = size_t(gradient_ad.size(-2));
        for (size_t i = 0; i < n_Gs * 7; ++i) {
            auto similar = are_similar(gradient_ad.template data_ptr<scalar_t>()[i], gradient_an.data_ptr<scalar_t>()[i], threshold);
            if (!similar) {
                std::cout << "target: " << mixture << std::endl;
                std::cout << "fitting: " << forward_out.fitting << std::endl;
                std::cout << "gradient target (analytical): " << gradient_target << std::endl;
                std::cout << "gradient target (autodiff): " << autodiff_out.mixture_gradient << std::endl;
                std::cout << "gradient_fitting: " << gradient_fitting << std::endl;
                std::cout << "i = " << i << "; difference: " << gradient_ad.template data_ptr<scalar_t>()[i] - gradient_an.data_ptr<scalar_t>()[i] << std::endl;
                bvh_mhem_fit_alpha::implementation_autodiff_backward<N_REDUCTION, scalar_t, 2>(mixture, gradient_fitting, alpha_config);
                auto forward_out = bvh_mhem_fit_alpha::forward_impl(mixture, alpha_config);
                bvh_mhem_fit_alpha::backward_impl(gradient_fitting, forward_out, alpha_config);
            }
            REQUIRE(similar);
        }
    }
}
} // anonymous namespace

TEST_CASE( "testing alpha against autodiff" ) {
    using namespace torch::indexing;
    using scalar_t = double;

    SECTION("reduce 2 fit 2 (double)") {
        runTest<2, double>(0.0000000001, _combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                               {_collectionOf2dMixtures_with4Gs(),
                                                                                _collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                                _collectionOf2dMixtures_with128Gs_red2()
                                                                                /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}));
    }

    SECTION("reduce 2 fit 2 (double)") {
        runTest<2, float>(0.0015f, _combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                         {_collectionOf2dMixtures_with4Gs(),
                                                                          _collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                          _collectionOf2dMixtures_with128Gs_red2()
                                                                          /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}));
    }

    SECTION("reduce 2 fit 4 (double)") {
        runTest<2, double>(0.0000000001, _combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                               {_collectionOf2dMixtures_with4Gs(),
                                                                                _collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                                _collectionOf2dMixtures_with128Gs_red2()
                                                                                /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}), 4);
    }
    SECTION("reduce 2 fit 4 (float)") {
        runTest<2, float>(0.0015f, _combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                         {_collectionOf2dMixtures_with4Gs(),
                                                                          _collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                          _collectionOf2dMixtures_with128Gs_red2()
                                                                          /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}), 4);
    }

    SECTION("reduce 2 fit 8 (double)") {
        runTest<2, double>(0.0000000001, _combineCollectionsOfGradsAndMixtures({_collectionOf2d8GsGrads()},
                                                                               {_collectionOf2dMixtures_with4Gs(),
                                                                                _collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                                _collectionOf2dMixtures_with128Gs_red2(),
                                                                                _collectionOf2dMixtures_with128Gs_red4()
                                                                                /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}), 8);
    }
    SECTION("reduce 2 fit 8 (float)") {
        runTest<2, float>(0.0015f, _combineCollectionsOfGradsAndMixtures({_collectionOf2d8GsGrads()},
                                                                         {_collectionOf2dMixtures_with4Gs(),
                                                                          _collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                          _collectionOf2dMixtures_with128Gs_red2(),
                                                                          _collectionOf2dMixtures_with128Gs_red4()
                                                                          /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}), 8);
    }

    SECTION("reduce 4 fit 4 (double)") {
        runTest<4, double>(0.0000000001, _combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                               {_collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                                _collectionOf2dMixtures_with8GsForMediumLongAutoDiff(),
                                                                                _collectionOf2dMixtures_with128Gs_red2(),
                                                                                _collectionOf2dMixtures_with128Gs_red4()
                                                                                 /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}));
    }
    SECTION("reduce 4 fit 4 (float)") {
        runTest<4, float>(0.0015f, _combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                         {_collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                          _collectionOf2dMixtures_with8GsForMediumLongAutoDiff(),
                                                                          _collectionOf2dMixtures_with128Gs_red2(),
                                                                          _collectionOf2dMixtures_with128Gs_red4()
                                                                           /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}));
    }

    SECTION("reduce 4 fit 32 (double)") {
        runTest<4, double>(0.0000000001, _combineCollectionsOfGradsAndMixtures({_collectionOf2d32GsGradsWellBehaving()},
                                                                               {_collectionOf2dMixtures_with128Gs_red2(),
                                                                                _collectionOf2dMixtures_with128Gs_red4(),
                                                                                _collectionOf2dMixtures_with128Gs_red8()
                                                                                 /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}), 32);
    }
    SECTION("reduce 4 fit 32 (float)") {
        runTest<4, float>(0.0015f, _combineCollectionsOfGradsAndMixtures({_collectionOf2d32GsGradsWellBehaving()},
                                                                         {_collectionOf2dMixtures_with128Gs_red2(),
                                                                          _collectionOf2dMixtures_with128Gs_red4(),
                                                                          _collectionOf2dMixtures_with128Gs_red8()
                                                                           /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}), 32);
    }


#ifndef GPE_LIMIT_N_REDUCTION
    SECTION("reduce 8 fit 8 (double)") {
        runTest<8, double>(0.0000000001, _combineCollectionsOfGradsAndMixtures({_collectionOf2d8GsGrads()},
                                                                               {_collectionOf2dMixtures_with128Gs_red2(),
                                                                                _collectionOf2dMixtures_with128Gs_red4(),
                                                                                _collectionOf2dMixtures_with128Gs_red8()
                                                                                 /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}));
    }
    SECTION("reduce 8 fit 8 (float)") {
        runTest<8, float>(0.0015f, _combineCollectionsOfGradsAndMixtures({_collectionOf2d8GsGrads()},
                                                                         {_collectionOf2dMixtures_with128Gs_red2(),
                                                                          _collectionOf2dMixtures_with128Gs_red4(),
                                                                          _collectionOf2dMixtures_with128Gs_red8()
                                                                           /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}));
    }

    SECTION("reduce 8 fit 32 (double)") {
        runTest<8, double>(0.0000000001, _combineCollectionsOfGradsAndMixtures({_collectionOf2d32GsGradsWellBehaving()},
                                                                               {_collectionOf2dMixtures_with128Gs_red2(),
                                                                                _collectionOf2dMixtures_with128Gs_red4(),
                                                                                _collectionOf2dMixtures_with128Gs_red8()
                                                                                 /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}), 32);
    }
    SECTION("reduce 8 fit 32 (float)") {
        runTest<8, float>(0.0015f, _combineCollectionsOfGradsAndMixtures({_collectionOf2d32GsGradsWellBehaving()},
                                                                         {_collectionOf2dMixtures_with128Gs_red2(),
                                                                          _collectionOf2dMixtures_with128Gs_red4(),
                                                                          _collectionOf2dMixtures_with128Gs_red8()
                                                                           /*, _collectionOf2dMixtures_with8GsForLongAutoDiff()*/}), 32);
    }
#endif
}
