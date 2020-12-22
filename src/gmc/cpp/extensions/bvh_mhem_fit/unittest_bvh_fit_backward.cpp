#include <iostream>
#include <string>

#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QScrollArea>

#include <torch/torch.h>
#include <torch/script.h>

#include <cuda_runtime.h>

#include "bvh_mhem_fit/implementation.h"
#include "bvh_mhem_fit/implementation_autodiff_backward.h"
#include "common.h"
#include "evaluate_inversed/parallel_binding.h"
#include "integrate/binding.h"
#include "util/mixture.h"

constexpr bool RENDER = false;
constexpr uint RESOLUTION = 128;
constexpr bool DO_STATS = false;
constexpr uint N_FITTING_COMPONENTS = 2;

torch::Tensor render(torch::Tensor mixture, const int resolution, const int n_batch_limit) {
    using namespace torch::indexing;
    mixture = mixture.cuda();
    const auto n_batch = std::min(gpe::n_batch(mixture), n_batch_limit);
    mixture = mixture.index({Slice(None, n_batch)});
    const auto n_layers = gpe::n_layers(mixture);

    //    const auto weights = gpe::weights(mixture);
    const auto positions = gpe::positions(mixture);
    //    const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
    //    mixture = gpe::pack_mixture(weights, positions, invCovs.contiguous());

    //    const auto minPos = positions.min().item().toFloat() - 1.1f;
    //    const auto maxPos = positions.max().item().toFloat() + 1.1f;

    const auto minPos = -1.0f;
    const auto maxPos = 33.0f;

    const auto mesh = torch::meshgrid({torch::arange(minPos, maxPos, 1.00001f * (maxPos - minPos) / float(resolution), mixture.device()),
                                       torch::arange(minPos, maxPos, 1.00001f * (maxPos - minPos) / float(resolution), mixture.device())});
    auto xv = mesh[0];
    auto yv = mesh[1];
    auto xes = torch::cat({xv.reshape({-1, 1}), yv.reshape({-1, 1})}, 1).view({1, 1, -1, 2});

    return parallel_forward(mixture, xes).cpu().view({n_batch, n_layers, resolution, resolution});
}

void show(torch::Tensor rendering, const int resolution, const int n_batch_limit) {
    rendering = rendering.clone();
    const auto n_layers = gpe::n_layers(rendering);
    const auto n_batch = std::min(gpe::n_batch(rendering), n_batch_limit);
    rendering -= rendering.min();
    rendering /= rendering.max();
    rendering *= 255;
    rendering = rendering.to(torch::ScalarType::Char);
    rendering = rendering.transpose(2, 3).transpose(1, 2).contiguous();
    QImage qRendering(reinterpret_cast<uchar*>(rendering.data_ptr()), resolution * n_layers, resolution * n_batch, QImage::Format_Grayscale8);
    auto* myLabel = new QLabel();
    QPixmap pixmap = QPixmap::fromImage(qRendering);
    pixmap.setDevicePixelRatio(myLabel->devicePixelRatioF());
    myLabel->setPixmap(pixmap);

    auto* scrollarea = new QScrollArea();
    scrollarea->setWidget(myLabel);
    scrollarea->show();
}

int main(int argc, char *argv[]) {
    using namespace torch::indexing;
    using scalar_t = double;
    QApplication a(argc, argv);

    // test specific configuration:
    auto config = BvhMhemFitConfig{2,
                                   lbvh::Config{lbvh::Config::MortonCodeAlgorithm::Old},
                                   BvhMhemFitConfig::FitInitialDisparityMethod::CentroidDistance,
                                   BvhMhemFitConfig::FitInitialClusterMergeMethod::MaxWeight,
                                   20.5f,
                                   N_FITTING_COMPONENTS};

    std::vector<std::pair<torch::Tensor, torch::Tensor>> test_cases;
    test_cases.emplace_back(torch::tensor({{0.5f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.5f,  0.0f,  0.0f,  2.0f,  0.0f,  0.0f,  2.0f},
                                         {0.5f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f},
                                         {0.5f, 10.0f, 10.0f,  1.0f,  0.0f,  0.0f,  1.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                         {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7}));

    test_cases.emplace_back(torch::tensor({{0.5f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.5f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.5f, 20.0f, 10.0f,  5.0f,  0.0f,  0.0f,  7.0f},
                                         {0.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                         {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7}));

    test_cases.emplace_back(torch::tensor({{1.0f,  5.0f,  5.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.5f,  5.0f,  5.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                         {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7}));

    test_cases.emplace_back(torch::tensor({{1.0f,  4.0f,  3.0f,  2.0f, -1.5f, -1.5f,  4.0f},
                                         {0.8f,  5.0f,  6.0f,  3.0f, -2.5f, -2.5f,  5.5f},
                                         {0.5f, 18.0f, 19.0f,  4.0f,  0.4f,  0.4f,  7.0f},
                                         {1.5f, 20.0f, 21.0f,  5.0f,  0.5f,  0.5f,  8.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{0.7f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                         {1.3f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7}));
    test_cases.emplace_back(torch::tensor({{1.0f,  4.0f,  3.0f,  2.0f, -1.5f, -1.5f,  4.0f},
                                           {0.8f,  5.0f,  6.0f,  3.0f, -2.5f, -2.5f,  5.5f},
                                           {0.5f,  6.0f,  5.0f,  4.0f, -0.4f, -0.4f,  7.0f},
                                           {1.5f,  5.5f,  3.0f,  5.0f,  0.5f,  0.5f,  8.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                         {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f}}).view({1, 1, 2, 7}));

    test_cases.emplace_back(torch::tensor({{1.0f,  4.0f,  3.0f,  2.0f, -1.5f, -1.5f,  4.0f},
                                           {0.8f,  5.0f,  6.0f,  3.0f, -2.5f, -2.5f,  5.5f},
                                           {0.5f, 18.0f, 22.0f,  4.0f, -0.4f, -0.4f,  7.0f},
                                           {1.5f, 20.0f, 18.0f,  5.0f,  0.5f,  0.5f,  8.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                         {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f}}).view({1, 1, 2, 7}));

    test_cases.emplace_back(torch::tensor({{1.0f,  4.0f,  3.0f,  2.0f, -1.5f, -1.5f,  4.0f},
                                           {0.8f,  5.0f,  6.0f,  3.0f, -2.5f, -2.5f,  5.5f},
                                           {0.5f, 18.0f, 19.0f,  4.0f, -0.4f, -0.4f,  7.0f},
                                           {1.5f, 20.0f, 21.0f,  5.0f,  0.5f,  0.5f,  8.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{1.1f,  1.2f,  1.3f,  1.4f,  1.5f,  1.6f,  1.7f},
                                         {1.8f,  1.9f,  0.1f,  0.2f,  0.3f,  0.4f,  5.5f}}).view({1, 1, 2, 7}));

    for (const auto& test_case : test_cases) {
        torch::Tensor mixture = test_case.first;
        torch::Tensor gradient_fitting = test_case.second;
        if (sizeof(scalar_t) > 4) {
            mixture = mixture.to(torch::ScalarType::Double);
            gradient_fitting = gradient_fitting.to(torch::ScalarType::Double);
        }

        if (RENDER) {
            torch::Tensor gt_rendering = render(gpe::mixture_with_inversed_covariances(mixture), RESOLUTION, 1);
            show(gt_rendering, RESOLUTION, 1);
        }

        auto autodiff_out = bvh_mhem_fit::implementation_autodiff_backward<2, scalar_t, 2>(mixture, gradient_fitting, config);
        auto forward_out = bvh_mhem_fit::forward_impl(mixture, config);
        auto gradient_target = bvh_mhem_fit::backward_impl(gradient_fitting, forward_out, config);

//        std::cout << "target: " << mixture << std::endl;
//        std::cout << "fitting: " << forward_out.fitting << std::endl;
        std::cout << "gradient target (analytical): " << gradient_target << std::endl;
        std::cout << "gradient target (autodiff): " << autodiff_out.mixture_gradient << std::endl;

        std::cout << "gradient_fitting: " << gradient_fitting << std::endl;
        {
            auto gradient_an = gradient_target.contiguous();
            auto gradient_ad = autodiff_out.mixture_gradient.contiguous();
            for (size_t i = 0; i < 4*7; ++i) {
                assert(std::abs(gradient_ad.data_ptr<scalar_t>()[i] - gradient_an.data_ptr<scalar_t>()[i]) < scalar_t(0.0001));
            }
        }
        std::cout << "=========" << std::endl;

        if (RENDER) {
            torch::Tensor gt_rendering = render(gpe::mixture_with_inversed_covariances(forward_out.fitting), RESOLUTION, 1);
            show(gt_rendering, RESOLUTION, 1);
        }
    }

    std::cout << std::endl << "DONE" << std::endl;
    if (RENDER)
        return QApplication::exec();
    return 0;
}
