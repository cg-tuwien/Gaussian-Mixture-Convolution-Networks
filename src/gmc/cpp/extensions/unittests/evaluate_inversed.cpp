#include <iostream>
#include <string>
#include <type_traits>

#include <catch2/catch.hpp>
#include <cuda_runtime.h>
#include <QString>
#include <torch/torch.h>
#include <torch/script.h>

#include "evaluate_inversed/implementations.h"
#include "common.h"
#include "unittests/support.h"
#include "util/mixture.h"


constexpr uint N_BATCHES = 1;
constexpr uint N_CONVOLUTION_LAYERS = 3;

constexpr double gpe_float_precision = 4e-5;
constexpr double gpe_double_precision = 1e-10;

template <typename scalar_t>
void requireEqual(const torch::Tensor& a, const torch::Tensor& b, double threshold) {
    const auto ap = a.view(-1);
    const auto bp = b.view(-1);
    REQUIRE(ap.size(0) == bp.size(0));
    for (int64_t i = 0; i < ap.size(0); ++i) {
        const auto av = double(ap.data_ptr<scalar_t>()[i]);
        const auto bv = double(bp.data_ptr<scalar_t>()[i]);
        const auto similar = are_similar(av, bv, threshold);
        if (!similar) {
//            const auto batch_id = i / (ap.size(0) / a.size(0));
//            const auto layer_id = (i - batch_id * (ap.size(0) / a.size(0))) / (ap.size(0) / (a.size(0) * a.size(1)));
            std::cout << "i = " << i << ", i % 7 = " << i % 7 << ", a value = " << av << ", b value = " << bv << " (a - b) / a = " << (av - bv) / av << ", threshold=" << threshold << std::endl;
            std::cout << "a.sizes() = " << a.sizes() << ", b.sizes() = " << b.sizes() << std::endl;
//            std::cout << "a = " << a[batch_id][layer_id] << std::endl;
//            std::cout << "b = " << b[batch_id][layer_id] << std::endl;
//            std::cout << "(a - b) / a = " << (a[batch_id][layer_id] - b[batch_id][layer_id]) / a[batch_id][layer_id] << std::endl;
        }
        REQUIRE(similar);
    }
}

template <typename scalar_t, typename GenFun>
void runTest(const GenFun& make_fun, double precision_forward, double precision_grad) {
    using namespace torch::indexing;
    constexpr auto dtype = (sizeof(scalar_t) == 4) ? torch::ScalarType::Float : torch::ScalarType::Double;
    for (uint i = 0; i < N_BATCHES; ++i) {
        torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");
        auto list = container.attributes();

        for (uint i = 0; i < N_CONVOLUTION_LAYERS; i++) {
            torch::Tensor mixture = container.attr(std::to_string(i)).toTensor();

            const auto weights = gpe::weights(mixture);
            torch::Tensor positions = gpe::positions(mixture).contiguous();
            const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
            mixture = gpe::pack_mixture(weights, positions, invCovs.contiguous()).clone().contiguous();

            const auto forward_reference =  parallel_forward_impl(mixture.toType(torch::ScalarType::Double), positions.toType(torch::ScalarType::Double));
            auto grad_out = torch::rand_like(forward_reference);
            grad_out.index_put_({Ellipsis, 0}, grad_out.index({Ellipsis, 0}) + torch::ones({1}));
            grad_out.index_put_({Ellipsis, 5}, grad_out.index({Ellipsis, 4}));

            const auto grads_reference = parallel_backward_impl(grad_out, mixture.toType(torch::ScalarType::Double), positions.toType(torch::ScalarType::Double), true, true);

            {
                torch::Tensor forward, grad_mixture, grad_xes;
                std::tie(forward, grad_mixture, grad_xes) = make_fun(mixture.toType(dtype), positions.toType(dtype), grad_out.toType(dtype));
//
                requireEqual<double>(forward_reference, forward.toType(torch::ScalarType::Double), precision_forward);
                requireEqual<double>(std::get<0>(grads_reference), grad_mixture.toType(torch::ScalarType::Double), precision_grad);
                requireEqual<double>(std::get<1>(grads_reference), grad_xes.toType(torch::ScalarType::Double), precision_grad);
            }
        }
    }
}

TEMPLATE_TEST_CASE( "testing all against cpu parallel", "[evaluate_inversed]", float, double) {
    using scalar_t = TestType;
    const auto precision = (sizeof(scalar_t) == 4) ? gpe_float_precision : gpe_double_precision;

    SECTION("test parallel cuda") {
        runTest<scalar_t>([](const torch::Tensor& mixture, const torch::Tensor& positions, const torch::Tensor& grad_out) {
            const auto forward = parallel_forward_impl(mixture.cuda(), positions.cuda());
            const auto grads = parallel_backward_impl(grad_out.cuda(), mixture.cuda(), positions.cuda(), true, true);
            return std::make_tuple(forward.cpu(), std::get<0>(grads).cpu(), std::get<1>(grads).cpu());
        }, precision, precision);
    }

    SECTION("test optimised") {
        runTest<scalar_t>([](const torch::Tensor& mixture, const torch::Tensor& positions, const torch::Tensor& grad_out) {
            const auto forward = parallel_forward_optimised_impl(mixture.cuda(), positions.cuda());
            const auto grads = parallel_backward_optimised_impl(grad_out.cuda(), mixture.cuda(), positions.cuda(), true, true);
            return std::make_tuple(forward.cpu(), std::get<0>(grads).cpu(), std::get<1>(grads).cpu());
        }, precision, precision);
    }
}
