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
#include "bvh_mhem_fit/implementation.h"
#include "evaluate_inversed/evaluate_inversed.h"
#include "util/mixture.h"
#include "pieces/pieces.h"

constexpr uint N_BATCHES = 10;
constexpr uint CONVOLUTION_LAYER_START = 0;
constexpr uint CONVOLUTION_LAYER_END = 3;
constexpr uint LIMIT_N_BATCH = 100;
constexpr bool USE_CUDA = false;
constexpr bool BACKWARD = false;
constexpr bool RENDER = true;
constexpr uint RESOLUTION = 128;
constexpr bool DO_STATS = true;
constexpr uint N_FITTING_COMPONENTS = 32;

torch::Tensor render(torch::Tensor mixture, const int resolution, const int n_batch_limit) {
    using namespace torch::indexing;
    mixture = mixture.cuda();
    const auto n_batch = std::min(gpe::n_batch(mixture), n_batch_limit);
    mixture = mixture.index({Slice(None, n_batch)});
    const auto n_layers = gpe::n_layers(mixture);

    const auto weights = gpe::weights(mixture);
    const auto positions = gpe::positions(mixture);
    const auto invCovs =  pieces::matrix_inverse(gpe::covariances(mixture)).transpose(-1, -2);
    mixture = gpe::pack_mixture(weights, positions, invCovs.contiguous());

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
    std::cout << "n_red, morton, em_kldiv, layer_0, time_0, layer_1, time_1, layer_2, time_2" << std::endl;

    for (const auto& named_config : configs) {
        for (uint i = 0; i < N_BATCHES; ++i) {
            torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/after_fixed_point_batch" + std::to_string(i) + ".pt");
            auto list = container.attributes();

            for (uint i = 0; i < CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START; i++) {
                assert(i + CONVOLUTION_LAYER_START < 3);
                auto mixture = container.attr(std::to_string(i + CONVOLUTION_LAYER_START)).toTensor();

                mixture = gpe::pack_mixture(torch::abs(gpe::weights(mixture)), gpe::positions(mixture), gpe::covariances(mixture));
                if (USE_CUDA)
                    mixture = mixture.cuda();
//                std::cout << "layer " << i + CONVOLUTION_LAYER_START << ": " << mixture.sizes() << " device: " << mixture.device() << std::endl;
//                auto t0 = std::chrono::high_resolution_clock::now();

                torch::Tensor gt_rendering;
                if (DO_STATS || RENDER)
                    gt_rendering = render(mixture, RESOLUTION, LIMIT_N_BATCH);

//                cudaDeviceSynchronize();
//                auto t1 = std::chrono::high_resolution_clock::now();

                if (RENDER)
                    show(gt_rendering, RESOLUTION, LIMIT_N_BATCH);

                if (BACKWARD) {
                    auto mixture_copy = mixture.clone();
                    for (int i = 0; i < 100; ++i) {
                        std::cout << "step " << i << std::endl;
                        auto forward_out = bvh_mhem_fit::forward_impl(mixture_copy, named_config.second);
                        std::cout << "forward_out.fitting: " << forward_out.fitting << std::endl;
//                        auto target = torch::tensor({{0.5f,  5.0f,  5.0f,  4.0f, -2.5f, -2.5f,  4.0f},
//                                                     {2.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 2, 7});
//                        auto gradient_fitting = forward_out.fitting.cpu() - target;
                        auto gradient_fitting = torch::tensor({{1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
                                                               {1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f,  1.0f}}).view({1, 1, 2, 7});
                        if (USE_CUDA)
                            gradient_fitting = gradient_fitting.cuda();
                        std::cout << "gradient_fitting: " << gradient_fitting << std::endl;
                        auto gradient_target = bvh_mhem_fit::backward_impl(gradient_fitting, forward_out, named_config.second);
                        std::cout << "gradient_target: " << gradient_target << std::endl;
                        mixture_copy -= gradient_target * 0.1f;
                        std::cout << "mixture_copy: " << mixture_copy << std::endl;
                        std::cout << "=========" << std::endl;
                    }
//                    assert(gpe::positions(gradient_target).abs().sum().item<float>() < 0.000000001f);
//                    assert(gpe::covariances(gradient_target).abs().sum().item<float>() < 0.000000001f);
//                    assert((gpe::weights(gradient_target) <= 0).all().item<bool>());
                }

                cudaDeviceSynchronize();
                auto t2 = std::chrono::high_resolution_clock::now();

                auto forward_out = bvh_mhem_fit::forward_impl(mixture, named_config.second);
                cudaDeviceSynchronize();
                auto t3 = std::chrono::high_resolution_clock::now();
                torch::Tensor fitted_mixture = forward_out.fitting;
                torch::Tensor fitted_rendering;
                if (DO_STATS || RENDER)
                        fitted_rendering = render(fitted_mixture, RESOLUTION, LIMIT_N_BATCH);
//                cudaDeviceSynchronize();
//                auto t4 = std::chrono::high_resolution_clock::now();
                if (RENDER) {
                    show(fitted_rendering, RESOLUTION, LIMIT_N_BATCH);
                }
                if (DO_STATS) {
                    auto diff = gt_rendering - fitted_rendering;
                    error_data[i].push_back(diff.cpu().view({1, -1, RESOLUTION * RESOLUTION}));
                    time_data[i].push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2));
                }
                else {
                    std::cout << std::fixed << std::setw( 14 ) << std::setprecision( 12 );
                    std::cout << "target mixture.sizes()" << mixture.sizes() << "  integral: " << gpe::weights(mixture).sum().item<float>() << std::endl;
                    std::cout << "fitted_mixture.sizes()" << fitted_mixture.sizes() << "  integral: " << gpe::weights(fitted_mixture).sum().item<float>() << std::endl;
                }
//                auto rmse = torch::sqrt(torch::mean(diff * diff)).item<float>();
//                std::cout << "elapsed time gt rendering=" << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << "ms, "
//                             "fit=" << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << "ms,"
//                             "fitted rendering=" << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << "ms" << std::endl;
            }
        }

        if (DO_STATS) {
//            std::cout << std::fixed << std::setw(5) << std::setprecision(4) << named_config.first << "; ";
            std::cout << std::fixed << std::setw( 11 ) << std::setprecision( 9 );
            std::cout << named_config.first;
            for (uint i = 0; i < CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START; i++) {
//                std::cout << "layer " << i << "; ";
                torch::Tensor d = torch::cat(error_data[i], 0);
//                std::cout << "RMSE=" << torch::sqrt(torch::mean(d * d)).item<float>() * 1000 << "e-3; ";
                std::cout << ", " << torch::sqrt(torch::mean(d * d)).item<float>();
                std::cout << ", " << float(std::accumulate(time_data[i].begin(), time_data[i].end(), std::chrono::milliseconds(0)).count()) / float(time_data[i].size());
                d = d.view({-1, RESOLUTION * RESOLUTION});
//                std::cout << "std(RMSE)=" << torch::sqrt(torch::sqrt(torch::mean(d * d, 1)).var() / d.size(0)).item<float>() * 1000 << "e-3; ";
                error_data[i].clear();
                time_data[i].clear();
            }
            std::cout << std::endl;
        }
    }

//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");
    std::cout << std::endl << "DONE" << std::endl;
    if (RENDER)
        return a.exec();
    else
        return 0;
}
