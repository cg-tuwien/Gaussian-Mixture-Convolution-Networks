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

constexpr uint N_BATCHES = 1;
constexpr uint N_CONVOLUTION_LAYERS = 1;
constexpr uint LIMIT_N_BATCH = 100;
constexpr bool USE_CUDA = false;
//constexpr bool BACKWARD = false;
constexpr bool RENDER = true;

void show(torch::Tensor mixture, const int resolution, const int n_batch_limit) {
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

    cudaDeviceSynchronize();
    auto rendering = parallel_forward(mixture, xes).cpu().view({n_batch, n_layers, resolution, resolution});
    cudaDeviceSynchronize();
    rendering -= rendering.min();
    rendering /= rendering.max();
    rendering *= 255;
    rendering = rendering.to(torch::ScalarType::Char);
    rendering = rendering.transpose(2, 3).transpose(1, 2).contiguous();
    QImage qRendering(reinterpret_cast<uchar*>(rendering.data_ptr()), resolution * n_layers, resolution * n_batch, QImage::Format_Grayscale8);
    QLabel* myLabel = new QLabel();
    myLabel->setPixmap(QPixmap::fromImage(qRendering));

    QScrollArea* scrollarea = new QScrollArea();
    scrollarea->setWidget(myLabel);
    scrollarea->show();
}

int main(int argc, char *argv[]) {
    using namespace torch::indexing;
    QApplication a(argc, argv);

    for (uint i = 0; i < N_BATCHES; ++i) {
        torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");
        auto list = container.attributes();

        for (uint i = 0; i < N_CONVOLUTION_LAYERS; i++) {
//            auto mixture = container.attr(std::to_string(i)).toTensor();//.index({Slice(5, 6), Slice(0, 1), Slice(), Slice()});
            auto mixture = torch::tensor({{0.5f,  5.0f,  5.0f,  4.0f, -0.5f, -0.5f,  4.0f},
                                          {0.5f,  8.0f,  8.0f,  4.0f, -2.5f, -2.5f,  4.0f},
                                          {0.5f, 20.0f, 10.0f,  5.0f,  0.0f,  0.0f,  7.0f},
                                          {0.5f, 20.0f, 20.0f,  5.0f,  0.5f,  0.5f,  7.0f}}).view({1, 1, 4, 7});
            mixture = gpe::pack_mixture(torch::abs(gpe::weights(mixture)), gpe::positions(mixture), gpe::covariances(mixture));
            if (USE_CUDA)
                mixture = mixture.cuda();
            std::cout << "layer " << i << ": " << mixture.sizes() << " device: " << mixture.device() << std::endl;
            if (RENDER)
                show(gpe::mixture_with_inversed_covariances(mixture), 128, LIMIT_N_BATCH);

            cudaDeviceSynchronize();

            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor fitted_mixture, nodes, aabbs;
            std::tie(fitted_mixture, nodes, aabbs) = bvh_mhem_fit_forward(mixture, 2);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            if (RENDER) {
                show(gpe::mixture_with_inversed_covariances(fitted_mixture), 128, LIMIT_N_BATCH);
            }
            std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";
        }
    }


//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");
    std::cout << "DONE" << std::endl;
    return a.exec();
    return 0;
}
