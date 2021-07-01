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
#include "convolution_fitting/implementation.h"
#include "evaluate_inversed/evaluate_inversed.h"
#include "integrate/binding.h"
#include "util/mixture.h"

constexpr uint N_BATCHES = 10;
constexpr uint CONVOLUTION_LAYER_START = 0;
constexpr uint CONVOLUTION_LAYER_END = 3;
constexpr uint LIMIT_N_BATCH = 100;
constexpr bool USE_CUDA = false;
constexpr bool BACKWARD = false;
constexpr bool RENDER = false;
constexpr uint RESOLUTION = 128;
constexpr bool DO_STATS = true;
constexpr uint N_FITTING_COMPONENTS = 32;

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

    return std::get<0>(evaluate_inversed::parallel_forward(mixture, xes)).cpu().view({n_batch, n_layers, resolution, resolution});
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

    std::array<std::vector<torch::Tensor>, CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START> error_data;
    std::array<std::vector<std::chrono::milliseconds>, CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START> time_data;

    // test all configurations:
//    std::vector<int> reduction_n_options = {2, 4, 8, 16};
//    std::vector<lbvh::Config::MortonCodeAlgorithm> morton_code_options = {
//        lbvh::Config::MortonCodeAlgorithm::Old,
//        lbvh::Config::MortonCodeAlgorithm::Cov1_12p36pc16i,
//        lbvh::Config::MortonCodeAlgorithm::Cov2_54pc10i,
//        lbvh::Config::MortonCodeAlgorithm::Cov3_27p27c10i,
//        lbvh::Config::MortonCodeAlgorithm::Cov4_27c27p10i
//    };
//    std::vector<BvhMhemFitConfig::FitInitialDisparityMethod> fit_initial_disparity_options = {
//        BvhMhemFitConfig::FitInitialDisparityMethod::CentroidDistance,
//        BvhMhemFitConfig::FitInitialDisparityMethod::Likelihood,
//        BvhMhemFitConfig::FitInitialDisparityMethod::KLDivergence
//    };
//    std::vector<BvhMhemFitConfig::FitInitialClusterMergeMethod> fit_initial_cluster_merge_options = {
//        BvhMhemFitConfig::FitInitialClusterMergeMethod::Average,
//        BvhMhemFitConfig::FitInitialClusterMergeMethod::AverageCorrected,
//        BvhMhemFitConfig::FitInitialClusterMergeMethod::MaxWeight,
//        BvhMhemFitConfig::FitInitialClusterMergeMethod::MaxIntegral
//    };
//    std::vector<float> em_kl_div_threshold_options {0.5, 1.0f, 1.5f, 2.0f, 2.5f};

    // test specific configuration:
#ifndef GPE_LIMIT_N_REDUCTION
    std::vector<int> reduction_n_options = {4, 8, 16};
#else
    std::vector<int> reduction_n_options = {4};
#endif
    std::vector<lbvh::Config::MortonCodeAlgorithm> morton_code_options = {
        lbvh::Config::MortonCodeAlgorithm::Old
    };
    std::vector<float> em_kl_div_threshold_options {0.5f};


    std::vector<std::pair<std::string, convolution_fitting::Config>> configs;
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
                                     convolution_fitting::Config{reduction_n, lbvh::Config{morton_code_algorithm}, em_kl_div_threshold, N_FITTING_COMPONENTS});
            }
        }
    }
    std::cout << "n_red, morton, em_kldiv, layer_0, time_0, layer_1, time_1, layer_2, time_2" << std::endl;

    for (const auto& named_config : configs) {
        for (uint i = 0; i < N_BATCHES; ++i) {

        }

    }

//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");
    std::cout << std::endl << "DONE" << std::endl;
    if (RENDER)
        return a.exec();
    else
        return 0;
}
