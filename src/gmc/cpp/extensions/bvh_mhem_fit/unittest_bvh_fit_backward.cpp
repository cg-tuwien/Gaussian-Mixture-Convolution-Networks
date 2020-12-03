#include <iostream>
#include <string>

#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QScrollArea>

#include <torch/torch.h>
#include <torch/script.h>

#include <cuda_runtime.h>

#include "common.h"
#include "mixture.h"
#include "evaluate_inversed/parallel_binding.h"
#include "bvh_mhem_fit/implementation.h"
#include "bvh_mhem_fit/implementation_autodiff_backward.h"
#include "integrate/binding.h"

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
    QLabel* myLabel = new QLabel();
    QPixmap pixmap = QPixmap::fromImage(qRendering);
    pixmap.setDevicePixelRatio(myLabel->devicePixelRatioF());
    myLabel->setPixmap(pixmap);

    QScrollArea* scrollarea = new QScrollArea();
    scrollarea->setWidget(myLabel);
    scrollarea->show();
}

int main(int argc, char *argv[]) {
    using namespace torch::indexing;
    QApplication a(argc, argv);

    // test specific configuration:
    auto config = BvhMhemFitConfig{2,
                                   lbvh::Config{lbvh::Config::MortonCodeAlgorithm::Old},
                                   BvhMhemFitConfig::FitInitialDisparityMethod::CentroidDistance,
                                   BvhMhemFitConfig::FitInitialClusterMergeMethod::MaxWeight,
                                   0.5f,
                                   N_FITTING_COMPONENTS};

    std::vector<std::pair<torch::Tensor, torch::Tensor>> test_cases;
    test_cases.push_back({torch::tensor({{0.5f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                         {0.5f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.5f, 20.0f, 10.0f,  5.0f,  0.0f,  0.0f,  7.0f},
                                         {0.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                         {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7})});
    test_cases.push_back({torch::tensor({{1.0f,  5.0f,  5.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.5f,  5.0f,  5.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                         {0.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f},
                                         {1.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7}),
                          torch::tensor({{1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
                                         {1.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f,  0.0f}}).view({1, 1, 2, 7})});

    for (const auto& test_case : test_cases) {
        const torch::Tensor& mixture = test_case.first;
        const torch::Tensor& gradient_fitting = test_case.second;

        if (RENDER) {
            torch::Tensor gt_rendering = render(gpe::mixture_with_inversed_covariances(mixture), RESOLUTION, 1);
            show(gt_rendering, RESOLUTION, 1);
        }

        auto autodiff_out = bvh_mhem_fit::implementation_autodiff_backward<2, float, 2>(mixture/*.to(torch::ScalarType::Double)*/,
                                                                                         gradient_fitting/*.to(torch::ScalarType::Double)*/,
                                                                                         config);
        auto forward_out = bvh_mhem_fit::forward_impl(mixture, config);
        auto gradient_target = bvh_mhem_fit::backward_impl(gradient_fitting, forward_out, config);

//        std::cout << "target: " << mixture << std::endl;
//        std::cout << "fitting: " << forward_out.fitting << std::endl;

        std::cout << "gradient_fitting: " << gradient_fitting << std::endl;
        std::cout << "gradient target (analytical): " << gradient_target << std::endl;
        std::cout << "gradient target (autodiff): " << autodiff_out.mixture_gradient << std::endl;
        std::cout << "=========" << std::endl;

        if (RENDER) {
            torch::Tensor gt_rendering = render(gpe::mixture_with_inversed_covariances(forward_out.fitting), RESOLUTION, 1);
            show(gt_rendering, RESOLUTION, 1);
        }
    }

    std::cout << std::endl << "DONE" << std::endl;
    if (RENDER)
        return a.exec();
    else
        return 0;
}
