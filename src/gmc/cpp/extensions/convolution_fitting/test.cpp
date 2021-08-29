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
#include "convolution/implementation.h"
#include "convolution_fitting/implementation.h"
#include "evaluate_inversed/evaluate_inversed.h"
#include "util/mixture.h"

constexpr uint N_BATCHES = 1;
constexpr uint CONVOLUTION_LAYER_START = 1;
constexpr uint CONVOLUTION_LAYER_END = 2;
constexpr uint LIMIT_N_BATCH = 5;
constexpr bool USE_CUDA = true;
constexpr bool BACKWARD = false;
constexpr bool RENDER = true;
constexpr uint RESOLUTION = 128;
constexpr bool DO_STATS = true;
constexpr uint N_FITTING_COMPONENTS = 8;

torch::Tensor render(torch::Tensor mixture, const int resolution, const int n_batch_limit) {
    using namespace torch::indexing;
    mixture = mixture.cuda();
    const auto n_batch = std::min(gpe::n_batch(mixture), n_batch_limit);
    mixture = mixture.index({Slice(None, n_batch)});
    const auto n_layers = gpe::n_layers(mixture);

    const auto weights = gpe::weights(mixture);
    const auto positions = gpe::positions(mixture);
//    std::cout << mixture << std::endl;
    const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
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

void show(torch::Tensor rendering, const int resolution, const int n_batch_limit, const QString& name = "") {
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
    scrollarea->setWindowTitle(name);
    scrollarea->show();
}

torch::Tensor toPdfMixture(const torch::Tensor& data) {
    const auto weights = pieces::integrate(data.view({-1, 1, 1, data.size(-1)})).contiguous().view({data.size(0), data.size(1), data.size(2)});
    return gpe::pack_mixture(weights, gpe::positions(data), gpe::covariances(data));
}

torch::Tensor toAmplitudeMixture(const torch::Tensor& data) {
    const auto uniAmpM = gpe::pack_mixture(torch::ones_like(gpe::weights(data)), gpe::positions(data), gpe::covariances(data));
    const auto normFactors = pieces::integrate(uniAmpM.view({-1, 1, 1, data.size(-1)})).contiguous().view({data.size(0), data.size(1), data.size(2)});
    return gpe::pack_mixture(gpe::weights(data) / normFactors, gpe::positions(data), gpe::covariances(data));
}

int main(int argc, char *argv[]) {
    using namespace torch::indexing;
    QApplication a(argc, argv);

    std::array<std::vector<torch::Tensor>, CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START> error_data;
    std::array<std::vector<std::chrono::milliseconds>, CONVOLUTION_LAYER_END - CONVOLUTION_LAYER_START> time_data;

    convolution_fitting::Config config;
    config.n_components_fitting = N_FITTING_COMPONENTS;

    for (uint b = 0; b < N_BATCHES; ++b) {
        torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/mnist_intermediate_data/conv_inputs_" + std::to_string(b) + ".pt");
        auto list = container.attributes();

        for (uint l = CONVOLUTION_LAYER_START; l < CONVOLUTION_LAYER_END; l++) {
            torch::Tensor data = container.attr("conv_layer_" + std::to_string(l) + "_data").toTensor()/*.index({Slice(0, 1), Slice(0, 2), Slice(0, 5), Slice()})*/.contiguous();
            torch::Tensor kernels = container.attr("conv_layer_" + std::to_string(l) + "_kernels").toTensor()/*.index({Slice(0, 1), Slice(0, 2), Slice(0, 3), Slice()})*/.contiguous();
//            auto mixture = torch::tensor({{0.02f, 0.f, 0.f, 1.01f, 1.f, 1.f, 1.0f},
//                                          {0.02f, 5.f, 5.f, 1.01f, 0.5f, 0.5f, 4.0f}}).view({1, 1, 2, 7});
            if (USE_CUDA) {
                data = data.cuda();
                kernels = kernels.cuda();
            }
            std::cout << "layer " << l << " data: " << data.sizes() << " device: " << data.device() << std::endl;
            std::cout << "layer " << l << " kernels: " << kernels.sizes() << " device: " << kernels.device() << std::endl;
            std::cout << "target number of gaussians: " << data.size(1) * data.size(2) * kernels.size(2) << ", fitting number of gaussians: " << config.n_components_fitting << std::endl;
//            show(render(data, 128, LIMIT_N_BATCH), 128, LIMIT_N_BATCH);
//            show(render(kernels, 128, LIMIT_N_BATCH), 128, LIMIT_N_BATCH);

            const auto reference = render(convolution::forward_impl(data, kernels), 128, LIMIT_N_BATCH);
            show(reference, 128, LIMIT_N_BATCH, "reference");
            const auto newTMixtureFitting = convolution_fitting::forward_impl(toPdfMixture(data), toPdfMixture(kernels), config).fitting;
            const auto fitting = render(toAmplitudeMixture(newTMixtureFitting), 128, LIMIT_N_BATCH);
            show(fitting, 128, LIMIT_N_BATCH, "fitting");

            const auto diff = fitting - reference;
            const auto mse = (diff * diff).mean().item();
            std::cout << "MSE batch " << b << " layer " << l << ": " << mse.to<float>() << std::endl;
        }
    }

//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");
    std::cout << std::endl << "DONE" << std::endl;
    if (RENDER)
        return a.exec();
    else
        return 0;
}
