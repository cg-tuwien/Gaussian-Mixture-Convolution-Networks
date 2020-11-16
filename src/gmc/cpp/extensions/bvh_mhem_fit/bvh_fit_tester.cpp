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
#include "bvh_mhem_fit/bindings.h"

constexpr uint N_BATCHES = 10;
constexpr uint CONVOLUTION_LAYER_START = 0;
constexpr uint CONVOLUTION_LAYER_END = 3;
constexpr uint LIMIT_N_BATCH = 100;
constexpr bool USE_CUDA = true;
//constexpr bool BACKWARD = false;
constexpr bool RENDER = false;
constexpr uint RESOLUTION = 128;
constexpr bool DO_STATS = true;

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

    std::array<std::vector<torch::Tensor>, CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START> error_data;

    // test all configurations:
    std::vector<int> reduction_n_options = {2, 4, 8, 16};
    std::vector<lbvh::Config::MortonCodeAlgorithm> morton_code_options = {
        lbvh::Config::MortonCodeAlgorithm::Old,
        lbvh::Config::MortonCodeAlgorithm::Cov1_12p36pc16i,
        lbvh::Config::MortonCodeAlgorithm::Cov2_54pc10i,
        lbvh::Config::MortonCodeAlgorithm::Cov3_27p27c10i,
        lbvh::Config::MortonCodeAlgorithm::Cov4_27c27p10i
    };
    std::vector<BvhMhemFitConfig::FitInitialDisparityMethod> fit_initial_disparity_options = {
        BvhMhemFitConfig::FitInitialDisparityMethod::CentroidDistance,
        BvhMhemFitConfig::FitInitialDisparityMethod::Likelihood,
        BvhMhemFitConfig::FitInitialDisparityMethod::KLDivergence
    };
    std::vector<BvhMhemFitConfig::FitInitialClusterMergeMethod> fit_initial_cluster_merge_options = {
        BvhMhemFitConfig::FitInitialClusterMergeMethod::Average,
        BvhMhemFitConfig::FitInitialClusterMergeMethod::AverageCorrected,
        BvhMhemFitConfig::FitInitialClusterMergeMethod::MaxWeight,
        BvhMhemFitConfig::FitInitialClusterMergeMethod::MaxIntegral
    };
    std::vector<float> em_kl_div_threshold_options {1.5f, 2.0f, 2.5f};

    // test specific configuration:
//    std::vector<int> reduction_n_options = {4};
//    std::vector<lbvh::Config::MortonCodeAlgorithm> morton_code_options = {
//        lbvh::Config::MortonCodeAlgorithm::Cov3_27p27c10i
//    };
//    std::vector<BvhMhemFitConfig::FitInitialDisparityMethod> fit_initial_disparity_options = {
//        BvhMhemFitConfig::FitInitialDisparityMethod::CentroidDistance
//    };
//    std::vector<BvhMhemFitConfig::FitInitialClusterMergeMethod> fit_initial_cluster_merge_options = {
//        BvhMhemFitConfig::FitInitialClusterMergeMethod::MaxWeight
//    };
//    std::vector<float> em_kl_div_threshold_options {1.5f};


    std::vector<std::pair<std::string, BvhMhemFitConfig>> configs;
    for (auto reduction_n : reduction_n_options) {
        for (auto morton_code_algorithm : morton_code_options) {
            for (auto fit_initial_disparity_method : fit_initial_disparity_options) {
                for (auto fit_initial_cluster_merge_method : fit_initial_cluster_merge_options) {
                    for (auto em_kl_div_threshold : em_kl_div_threshold_options) {
                        configs.emplace_back("red_" + std::to_string(reduction_n) +
                                             "_morton_" + std::to_string(int(morton_code_algorithm)) +
                                             "_fidispr_" + std::to_string(int(fit_initial_disparity_method)) +
                                             "_ficlstrm_" + std::to_string(int(fit_initial_cluster_merge_method)) +
                                             "_emkldivth_" + std::to_string(em_kl_div_threshold),
                                             BvhMhemFitConfig{reduction_n, lbvh::Config{morton_code_algorithm}, fit_initial_disparity_method, fit_initial_cluster_merge_method, em_kl_div_threshold});
//                        goto outoutoutoutout;
                    }
                }
            }

        }
    }
//    outoutoutoutout:

    for (const auto& named_config : configs) {
        for (uint i = 0; i < N_BATCHES; ++i) {
            torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");
            auto list = container.attributes();

            for (uint i = CONVOLUTION_LAYER_START; i < CONVOLUTION_LAYER_END; i++) {
                assert(i < 3);
                auto mixture = container.attr(std::to_string(i)).toTensor();
    //            mixture = mixture.index({Slice(7, 8), Slice(0,1), Slice(), Slice()});
    //            auto mixture = torch::tensor({{0.5f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
    //                                          {0.5f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
    //                                          {0.5f, 20.0f, 10.0f,  5.0f,  0.0f,  0.0f,  7.0f},
    //                                          {0.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7});
                mixture = gpe::pack_mixture(torch::abs(gpe::weights(mixture)), gpe::positions(mixture), gpe::covariances(mixture));
                if (USE_CUDA)
                    mixture = mixture.cuda();
    //            std::cout << "layer " << i << ": " << mixture.sizes() << " device: " << mixture.device() << std::endl;
//                auto t0 = std::chrono::high_resolution_clock::now();

                torch::Tensor gt_rendering;
                if (DO_STATS || RENDER)
                    gt_rendering = render(gpe::mixture_with_inversed_covariances(mixture), RESOLUTION, LIMIT_N_BATCH);

//                cudaDeviceSynchronize();
//                auto t1 = std::chrono::high_resolution_clock::now();

                if (RENDER)
                    show(gt_rendering, RESOLUTION, LIMIT_N_BATCH);

//                cudaDeviceSynchronize();
//                auto t2 = std::chrono::high_resolution_clock::now();

                torch::Tensor fitted_mixture, nodes, aabbs;
                std::tie(fitted_mixture, nodes, aabbs) = bvh_mhem_fit_forward(mixture, named_config.second, 32);
//                cudaDeviceSynchronize();
//                auto t3 = std::chrono::high_resolution_clock::now();
                torch::Tensor fitted_rendering;
                if (DO_STATS || RENDER)
                        fitted_rendering = render(gpe::mixture_with_inversed_covariances(fitted_mixture), RESOLUTION, LIMIT_N_BATCH);
//                cudaDeviceSynchronize();
//                auto t4 = std::chrono::high_resolution_clock::now();
                if (RENDER) {
                    show(fitted_rendering, RESOLUTION, LIMIT_N_BATCH);
                }
                if (DO_STATS) {
                    auto diff = gt_rendering - fitted_rendering;
                    error_data[i].push_back(diff.cpu().view({1, -1, RESOLUTION * RESOLUTION}));
                }
                else {
                    std::cout << "fitted_mixture.sizes()" << fitted_mixture.sizes() << "  something: " << fitted_mixture.sum() << std::endl;
                }
//                auto rmse = torch::sqrt(torch::mean(diff * diff)).item<float>();
//                std::cout << "elapsed time gt rendering=" << std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count() << "ms, "
//                             "fit=" << std::chrono::duration_cast<std::chrono::milliseconds>(t3-t2).count() << "ms,"
//                             "fitted rendering=" << std::chrono::duration_cast<std::chrono::milliseconds>(t4-t3).count() << "ms" << std::endl;
            }
        }

        if (DO_STATS) {
            std::cout << std::fixed << std::setw(5) << std::setprecision(4) << named_config.first << "; ";
            for (uint i = CONVOLUTION_LAYER_START; i < CONVOLUTION_LAYER_END; i++) {
                std::cout << "layer " << i << "; ";
                torch::Tensor d = torch::cat(error_data[i], 0);
                std::cout << "RMSE=" << torch::sqrt(torch::mean(d * d)).item<float>() * 1000 << "e-3; ";
                d = d.view({-1, RESOLUTION * RESOLUTION});
                std::cout << "std(RMSE)=" << torch::sqrt(torch::sqrt(torch::mean(d * d, 1)).var() / d.size(0)).item<float>() * 1000 << "e-3; ";
                error_data[i].clear();
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
