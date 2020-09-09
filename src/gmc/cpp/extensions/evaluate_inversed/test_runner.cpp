#include <iostream>
#include <string>

#include <QApplication>
#include <QImage>
#include <QLabel>
#include <QScrollArea>

#include <torch/torch.h>
#include <torch/script.h>

#include "common.h"
#include "math/symeig.h"

torch::Tensor cpu_parallel_forward(torch::Tensor mixture, torch::Tensor xes);
torch::Tensor cuda_parallel_forward(torch::Tensor mixture, torch::Tensor xes);

torch::Tensor cuda_bvh_forward_impl(torch::Tensor mixture, torch::Tensor xes);

constexpr uint N_BATCHES = 1;
constexpr uint N_LAYERS = 1;
constexpr uint LIMIT_N_BATCH = 100;

void show(torch::Tensor mixture, const uint resolution, const uint n_batch_limit) {
    const auto eval_fun = mixture.is_cuda() ? &cuda_bvh_forward_impl : &cpu_parallel_forward;
    using namespace torch::indexing;
    const auto n_batch = std::min(gpe::n_batch(mixture), n_batch_limit);
    mixture = mixture.index({Slice(None, n_batch)});
    const auto n_layers = gpe::n_layers(mixture);

    const auto weights = gpe::weights(mixture);
    const auto positions = gpe::positions(mixture);
    const auto invCovs = gpe::covariances(mixture).inverse().transpose(-1, -2);
    mixture = gpe::pack_mixture(weights, positions, invCovs.contiguous());

    const auto minPos = positions.min().item().toFloat() - 0.1f;
    const auto maxPos = positions.max().item().toFloat() + 0.1f;

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

    auto t = torch::rand({4, 1, 2, 2});
    auto r = gpe::symeig(t);
    std::cout << r[0] << r[1] << std::endl;
    return 0;
    for (uint i = 0; i < N_BATCHES; ++i) {
        torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");
        auto list = container.attributes();

        for (uint i = 0; i < N_LAYERS; i++) {
            auto mixture = container.attr(std::to_string(i)).toTensor();//.index({Slice(), Slice(0, 1), Slice(0, 50), Slice()});
//            auto mixture = torch::tensor({0.02f, 0.f, 0.f, 1.01f, 1.f, 1.f, 1.0f}).view({1, 1, 1, 7});
            show(mixture.cuda(), 128, LIMIT_N_BATCH);

            std::cout << "layer " << i << ": " << mixture.sizes() << " device: " << mixture.device() << std::endl;

        }
    }


//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");

    return a.exec();
}
