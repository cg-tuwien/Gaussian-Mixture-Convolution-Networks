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

#include "parallel_binding.h"

//torch::Tensor cpu_parallel_forward(const torch::Tensor& mixture, const torch::Tensor& xes);
//torch::Tensor cuda_parallel_forward(const torch::Tensor& mixture, const torch::Tensor& xes);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> cuda_bvh_forward(const torch::Tensor& mixture, const torch::Tensor& xes);

torch::Tensor cuda_bvh_forward_wrapper(const torch::Tensor& mixture, const torch::Tensor& xes) {
    torch::Tensor sum, nodes, aabbs;
    std::tie(sum, nodes, aabbs) = cuda_bvh_forward(mixture, xes);
    return sum;
}

constexpr uint N_BATCHES = 1;
constexpr uint N_CONVOLUTION_LAYERS = 3;
constexpr uint LIMIT_N_BATCH = 100;

void show(torch::Tensor mixture, const uint resolution, const uint n_batch_limit) {
    const auto eval_fun = mixture.is_cuda() ? &cuda_bvh_forward_wrapper : &parallel_forward;
    using namespace torch::indexing;
    const auto n_batch = std::min(gpe::n_batch(mixture), n_batch_limit);
    mixture = mixture.index({Slice(None, n_batch)});
    const auto n_layers = gpe::n_layers(mixture);

    const auto weights = gpe::weights(mixture);
    const auto positions = gpe::positions(mixture);
    const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
    mixture = gpe::pack_mixture(weights, positions, invCovs.contiguous());

    const auto minPos = positions.min().item().toFloat() - 1.1f;
    const auto maxPos = positions.max().item().toFloat() + 1.1f;

    const auto mesh = torch::meshgrid({torch::arange(minPos, maxPos, 1.00001f * (maxPos - minPos) / float(resolution), mixture.device()),
                                       torch::arange(minPos, maxPos, 1.00001f * (maxPos - minPos) / float(resolution), mixture.device())});
    auto xv = mesh[0];
    auto yv = mesh[1];
    auto xes = torch::cat({xv.reshape({-1, 1}), yv.reshape({-1, 1})}, 1).view({1, 1, -1, 2});
    std::cout << "xes.sizes() = " << xes.sizes() << std::endl;

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();
    auto rendering = eval_fun(mixture, xes).cpu().view({n_batch, n_layers, resolution, resolution});
    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";
    std::cout << "rendering.sizes() = " << rendering.sizes()
              << ", min=" << rendering.min().item<float>()
              << ", max=" << rendering.max().item<float>() << std::endl;
    rendering -= rendering.min();
    rendering /= rendering.max();
    rendering *= 255;
    rendering = rendering.to(torch::ScalarType::Char);
    rendering = rendering.transpose(2, 3).transpose(1, 2).contiguous();
    QImage qRendering((uchar*) rendering.data_ptr(), int(resolution * n_layers), int(resolution * n_batch), QImage::Format_Grayscale8);
    QLabel* myLabel = new QLabel();
    myLabel->setPixmap(QPixmap::fromImage(qRendering));

    QScrollArea* scrollarea = new QScrollArea();
    scrollarea->setWidget(myLabel);
    scrollarea->show();
}

int main(int argc, char *argv[]) {
    using namespace torch::indexing;
    QApplication a(argc, argv);

    cudaDeviceSynchronize();
    for (uint i = 0; i < N_BATCHES; ++i) {
        torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");
        auto list = container.attributes();

        for (uint i = 0; i < N_CONVOLUTION_LAYERS; i++) {
            auto mixture = container.attr(std::to_string(i)).toTensor();//.index({Slice(0, 2), Slice(0, 1), Slice(0, 5), Slice()});
//            auto mixture = torch::tensor({{0.02f, 0.f, 0.f, 1.01f, 1.f, 1.f, 1.0f},
//                                          {0.02f, 5.f, 5.f, 1.01f, 0.5f, 0.5f, 4.0f}}).view({1, 1, 2, 7});
//            mixture = mixture.cuda();
            std::cout << "layer " << i << ": " << mixture.sizes() << " device: " << mixture.device() << std::endl;
//            show(mixture, 128, LIMIT_N_BATCH);

            const auto weights = gpe::weights(mixture);
            const auto positions = gpe::positions(mixture);
            const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
            mixture = gpe::pack_mixture(weights, positions, invCovs.contiguous());
//            cudaDeviceSynchronize();

            auto start = std::chrono::high_resolution_clock::now();
            const auto eval_fun = mixture.is_cuda() ? &cuda_bvh_forward_wrapper : &parallel_forward;
            std::cout << "1" << std::endl;
            auto rendering = eval_fun(mixture, positions.contiguous()).cpu();
            std::cout << "2" << std::endl;
//            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::cout << "elapsed time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count() << "ms\n";
        }
    }


//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");
    std::cout << "DONE" << std::endl;
//    return a.exec();
    return 0;
}
