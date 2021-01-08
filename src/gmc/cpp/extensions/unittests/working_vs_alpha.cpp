#include <iostream>
#include <string>
#include <type_traits>


#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include <QString>
#include <torch/torch.h>
#include <torch/script.h>


#include "bvh_mhem_fit/implementation.h"
#include "bvh_mhem_fit_alpha/implementation.h"
#include "common.h"
#include "evaluate_inversed/parallel_binding.h"
#include "integrate/binding.h"
#include "util/mixture.h"
#include "unittests/support.h"

struct use_cuda_type {
    constexpr static bool value = true;
};

struct use_cpu_type {
    constexpr static bool value = false;
};

template <bool USE_CUDA, int N_REDUCTION, typename scalar_t, int N_FITTING_COMPONENTS = N_REDUCTION>
void runTest(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& test_cases, double threshold) {
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

    double rrmse_forward = 0;
    double rrmse_backward = 0;
    double max_error_forward = 0;
    double max_error_backward = 0;
//    if (USE_CUDA)
//        std::cout << "cuda, ";
//    else
//        std::cout << "cpu,  ";

//    if (sizeof (scalar_t) == 4)
//        std::cout << "float,  ";
//    else
//        std::cout << "double, ";

    for (const auto& test_case : test_cases) {
        torch::Tensor mixture = test_case.first;
        torch::Tensor gradient_fitting = test_case.second.repeat({mixture.size(0), mixture.size(1), 1, 1});
        auto reference_mixture = mixture.to(torch::ScalarType::Double);
        auto reference_gradient_fitting = gradient_fitting.to(torch::ScalarType::Double);
        if (sizeof(scalar_t) > 4) {
            mixture = mixture.to(torch::ScalarType::Double);
            gradient_fitting = gradient_fitting.to(torch::ScalarType::Double);
        }
        if (USE_CUDA) {
            mixture = mixture.cuda();
            gradient_fitting = gradient_fitting.cuda();
        }

        auto forward_out = bvh_mhem_fit::forward_impl(mixture, config);
        auto gradient_target = bvh_mhem_fit::backward_impl(gradient_fitting, forward_out, config);
        auto reference_forward_out = bvh_mhem_fit_alpha::forward_impl(reference_mixture, reference_config);
        auto reference_gradient_target = bvh_mhem_fit_alpha::backward_impl(reference_gradient_fitting, reference_forward_out, reference_config);

        {
            auto fitting = forward_out.fitting.contiguous().cpu();
            auto fitting_reference = reference_forward_out.fitting.contiguous();
            REQUIRE(fitting_reference.size(-2) == config.n_components_fitting);
            double forward_error = 0;
            for (size_t i = 0; i < config.n_components_fitting * 7; ++i) {
                const auto a = fitting_reference.data_ptr<double>()[i];
                const auto b = double(fitting.data_ptr<scalar_t>()[i]);
                const auto e = (a - b) * (a - b) / std::max(a * a, 1.);
                forward_error += e;
                max_error_forward = std::max(max_error_forward, e);

                auto similar = are_similar(a, b, threshold);
                if (!similar) {
//                    std::cout << "fitting: " << forward_out.fitting << std::endl;
//                    std::cout << "reference: " << reference_forward_out.fitting << std::endl;
//                    std::cout << "i = " << i << "; difference: " << a << " - " << b << " = " << a - b << std::endl;
                    WARN(std::string("forward difference: ") + std::to_string(a) + " - " + std::to_string(b) + " = " + std::to_string(a - b));
//                    WARN(QString("forward difference: %1 - %2 = %3").arg(a).arg(b).arg(a-b).toStdString());
                }
                REQUIRE(similar);
            }
            forward_error /= config.n_components_fitting * 7;
            rrmse_forward += forward_error;

            auto gradient = gradient_target.contiguous().cpu();
            auto gradient_reference = reference_gradient_target.contiguous();
            auto n_Gs = size_t(gradient_reference.size(-2));
            double backward_error = 0;
            for (size_t i = 0; i < n_Gs * 7; ++i) {
                const auto a = gradient_reference.data_ptr<double>()[i];
                const auto b = double(gradient.data_ptr<scalar_t>()[i]);
                const auto e = (a - b) * (a - b) / std::max(a * a, 1.);
                backward_error += e;
                max_error_backward = std::max(max_error_backward, e);


                auto similar = are_similar(a, b, threshold);
                if (!similar) {
//                    std::cout << "target: " << mixture << std::endl;
//                    std::cout << "fitting: " << forward_out.fitting << std::endl;
//                    std::cout << "gradient target: " << gradient << std::endl;
//                    std::cout << "gradient target (reference): " << gradient_reference << std::endl;
//                    std::cout << "gradient_fitting: " << gradient_fitting << std::endl;
//                    std::cout << "i = " << i << "; difference: " << a << " - " << b << " = " << a - b << std::endl;
                    WARN(QString("backward difference: %1 - %2 = %3").arg(a).arg(b).arg(a-b).toStdString());
//                    WARN(std::string("backward difference: ") + std::to_string(a) + " - " + std::to_string(b) + " = " + std::to_string(a - b));
                }
                REQUIRE(similar);
            }
            backward_error /= double(n_Gs * 7);
            rrmse_backward += backward_error;
        }
    }
    rrmse_forward /= double(test_cases.size());
    rrmse_backward /= double(test_cases.size());
//    std::cout << QString("   forward rrmse: %1,  max: %2,   backward rrmse: %3,  max: %4")
//                 .arg(std::sqrt(rrmse_forward), 8, 'e', 2)
//                 .arg(std::sqrt(max_error_forward), 8, 'e', 2)
//                 .arg(std::sqrt(rrmse_backward), 8, 'e', 2)
//                 .arg(std::sqrt(max_error_backward), 8, 'e', 2)
//                 .toStdString() << std::endl;
}

constexpr double gpe_float_exploding_precision = 4e-3;
constexpr double gpe_float_precision = 4e-5;
constexpr double gpe_double_precision = 1e-11;

TEMPLATE_TEST_CASE( "testing working against alpha reference", "[bvh_mhem_fit]", use_cuda_type, use_cpu_type) {
//    std::cout << "01. ";
    SECTION("2 component fitting double _collectionOf2dMixtures_with4Gs") {
        runTest<TestType::value, 2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with4Gs()}),
                           gpe_double_precision);
    }
//    std::cout << "02. ";
    SECTION("2 component fitting double _collectionOf2dMixtures_with8GsForQuickAutoDiff") {
        runTest<TestType::value, 2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForQuickAutoDiff()}),
                           gpe_double_precision);
    }
//    std::cout << "03. ";
    SECTION("2 component fitting double _collectionOf2dMixtures_with8GsForLongAutoDiff") {
        runTest<TestType::value, 2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForLongAutoDiff()}),
                           gpe_double_precision);
    }
//    std::cout << "04. ";
    SECTION("2 component fitting double _collectionOf2dMixtures_with8GsTooLongForAutodiff") {
        runTest<TestType::value, 2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsTooLongForAutodiff()}),
                           gpe_double_precision);
    }
//    std::cout << "05. ";
    SECTION("2 component fitting double _collectionOf2dMixtures_with16Gs") {
        runTest<TestType::value, 2, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with16Gs()}),
                           gpe_double_precision);
    }

//    std::cout << "06. ";
    SECTION("2 component fitting float _collectionOf2dMixtures_with4Gs") {
        runTest<TestType::value, 2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with4Gs()}),
                           gpe_float_precision);
    }
//    std::cout << "07. ";
    SECTION("2 component fitting float _collectionOf2dMixtures_with8GsForQuickAutoDiff") {
        runTest<TestType::value, 2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForQuickAutoDiff()}),
                           gpe_float_precision);
    }
//    std::cout << "08. ";
    SECTION("2 component fitting float _collectionOf2dMixtures_with8GsForLongAutoDiff") {
        runTest<TestType::value, 2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForLongAutoDiff()}),
                           gpe_float_precision);
    }
//    std::cout << "09. ";
    SECTION("2 component fitting float _collectionOf2dMixtures_with8GsTooLongForAutodiff") {
        runTest<TestType::value, 2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsTooLongForAutodiff()}),
                           gpe_float_precision);
    }
//    std::cout << "10. ";
    SECTION("2 component fitting float _collectionOf2dMixtures_with16Gs") {
        runTest<TestType::value, 2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                 {_collectionOf2dMixtures_with16Gs()}),
                           gpe_float_precision);
    }

//    std::cout << "11. ";
    SECTION("2 component fitting float _collectionOf2dMixtures_causingNumericalProblems") {
        runTest<TestType::value, 2, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d2GsGrads()},
                                                                {_collectionOf2dMixtures_causingNumericalProblems()}),
                          gpe_float_exploding_precision);
    }

//    std::cout << "12. ";
    SECTION("4 component fitting double") {
        runTest<TestType::value, 4, double>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                 {_collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                  _collectionOf2dMixtures_with8GsForLongAutoDiff(),
                                                                  _collectionOf2dMixtures_with8GsTooLongForAutodiff(),
                                                                  _collectionOf2dMixtures_with16Gs()}),
                           gpe_double_precision);
    }

//    std::cout << "13. ";
    SECTION("4 component fitting float") {
        runTest<TestType::value, 4, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                {_collectionOf2dMixtures_with8GsForQuickAutoDiff(),
                                                                 _collectionOf2dMixtures_with8GsForLongAutoDiff(),
                                                                 _collectionOf2dMixtures_with8GsTooLongForAutodiff(),
                                                                 _collectionOf2dMixtures_with16Gs()}),
                          gpe_float_precision);
    }

//    std::cout << "14. ";
    SECTION("4 component fitting float with numerical problems") {
        runTest<TestType::value, 4, float>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d4GsGrads()},
                                                                {_collectionOf2dMixtures_causingNumericalProblems()}),
                          gpe_float_exploding_precision);
    }

//    std::cout << "15. ";
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

        runTest<TestType::value, 4, double, 32>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d32GsGradsWellBehaving(), _collectionOf2d32GsGradsExploding()}, {mixtures}), gpe_double_precision);
    }

//    std::cout << "16. ";
    SECTION("32 component fitting float of real world with well behaved gradients") {
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

        runTest<TestType::value, 4, float, 32>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d32GsGradsWellBehaving()}, {mixtures}), gpe_float_precision);
    }

//    std::cout << "17. ";
    SECTION("32 component fitting float of real world with exploding gradients", "[]") {
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

        runTest<TestType::value, 4, float, 32>(_combineCollectionsOfGradsAndMixtures({_collectionOf2d32GsGradsExploding()}, {mixtures}), gpe_float_exploding_precision);
    }
}
