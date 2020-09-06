#include <iostream>
#include <string>

#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QScrollArea>

#include <torch/torch.h>
#include <torch/script.h>

#include "common.h"

torch::Tensor cuda_forward(torch::Tensor mixture, torch::Tensor xes);
torch::Tensor cpu_forward(torch::Tensor mixture, torch::Tensor xes);

constexpr uint N_BATCHES = 1;
constexpr uint N_LAYERS = 1;
constexpr uint LIMIT_N_BATCH = 20;

void show(torch::Tensor mixture, const uint resolution, const uint n_batch_limit) {
    const auto eval_fun = mixture.is_cuda() ? &cuda_forward : &cpu_forward;
    using namespace torch::indexing;
    const auto n_batch = std::min(gm::n_batch(mixture), n_batch_limit);
    mixture = mixture.index({Slice(None, n_batch)});
    const auto n_layers = gm::n_layers(mixture);

    const auto weights = gm::weights(mixture);
    const auto positions = gm::positions(mixture);
    const auto invCovs = gm::covariances(mixture).inverse().transpose(-1, -2);
    mixture = gm::pack_mixture(weights, positions, invCovs.contiguous());

    const auto minPos = positions.min().item().toFloat();
    const auto maxPos = positions.max().item().toFloat();

    const auto mesh = torch::meshgrid({torch::arange(minPos, maxPos, 1.00001f * (maxPos - minPos) / float(resolution), mixture.device()),
                                       torch::arange(minPos, maxPos, 1.00001f * (maxPos - minPos) / float(resolution), mixture.device())});
    auto xv = mesh[0];
    auto yv = mesh[1];
    auto xes = torch::cat({xv.reshape({-1, 1}), yv.reshape({-1, 1})}, 1).view({1, 1, -1, 2});
    std::cout << "xes.sizes() = " << xes.sizes() << std::endl;

    auto rendering = eval_fun(mixture, xes).cpu().view({n_batch, n_layers, resolution, resolution});
    std::cout << "rendering.sizes() = " << rendering.sizes() << std::endl;
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

    for (uint i = 0; i < N_BATCHES; ++i) {
        torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");
        auto list = container.attributes();

        for (uint i = 0; i < N_LAYERS; i++) {
            auto mixture = container.attr(std::to_string(i)).toTensor();
            show(mixture.cuda(), 128, 100);

            std::cout << "layer " << i << ": " << mixture.sizes() << " device: " << mixture.device() << std::endl;

        }
    }


//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");

    return a.exec();
}
