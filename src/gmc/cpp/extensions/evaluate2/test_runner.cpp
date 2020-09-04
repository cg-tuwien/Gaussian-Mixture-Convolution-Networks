#include <iostream>
#include <string>

#include <QApplication>
#include <QImage>
#include <QLabel>

#include <torch/torch.h>
#include <torch/script.h>

#include "common.h"

torch::Tensor evaluate_inversed_forward(torch::Tensor mixture, torch::Tensor xes);

constexpr int N_BATCHES = 1;
constexpr int N_LAYERS = 1;

int main(int argc, char *argv[]) {
    using namespace torch::indexing;
    QApplication a(argc, argv);

    for (int i = 0; i < N_BATCHES; ++i) {
        torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch" + std::to_string(i) + ".pt");
        auto list = container.attributes();

        for (int i = 0; i < N_LAYERS; i++) {
            auto mixture = container.attr(std::to_string(i)).toTensor().cuda().index({Slice(None, 8)});
            auto n_batch = gm::n_batch(mixture);
            auto n_layers = gm::n_layers(mixture);

            auto weights = gm::weights(mixture);
            auto positions = gm::positions(mixture);
            auto invCovs = gm::covariances(mixture).inverse();
            mixture = gm::pack_mixture(weights, positions, invCovs.contiguous());
            auto minPos = positions.min().item().toFloat();
            auto maxPos = positions.max().item().toFloat();

            auto mesh = torch::meshgrid({torch::arange(minPos, maxPos, (maxPos - minPos) / 128.f, mixture.device()),
                                         torch::arange(minPos, maxPos, (maxPos - minPos) / 128.f, mixture.device())});
            auto xv = mesh[0];
            auto yv = mesh[1];
            auto xes = torch::cat({xv.reshape({-1, 1}), yv.reshape({-1, 1})}, 1).view({1, 1, -1, 2});
            std::cout << "xes.sizes() = " << xes.sizes() << std::endl;
            auto rendering = evaluate_inversed_forward(mixture, xes).cpu().view({n_batch, n_layers, 128, 128});
            std::cout << "rendering.sizes() = " << rendering.sizes() << std::endl;
            rendering -= rendering.min();
            rendering /= rendering.max();
            rendering *= 255;
            rendering = rendering.to(torch::ScalarType::Char);
            rendering = rendering.transpose(2, 3).transpose(1, 2).contiguous();
            QImage qRendering((uchar*) rendering.data_ptr(), 128*8, 128*8, QImage::Format_Grayscale8);
            QLabel* myLabel = new QLabel();
            myLabel->setPixmap(QPixmap::fromImage(qRendering));

            myLabel->show();
//            xv, yv = torch.meshgrid([torch.arange(x_low, x_high, (x_high - x_low) / width, dtype=torch.float, device=mixture.device),
//                                     torch.arange(y_low, y_high, (y_high - y_low) / height, dtype=torch.float, device=mixture.device)])
//            m = mixture.detach()[batches[0]:batches[1], layers[0]:layers[1]]
//            c = constant.detach()[batches[0]:batches[1], layers[0]:layers[1]]
//            n_batch = m.shape[0]
//            n_layers = m.shape[1]
//            xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)
//            rendering = (evaluate(m, xes) + c.unsqueeze(-1)).view(n_batch, n_layers, width, height).transpose(2, 3)
//            rendering = rendering.transpose(0, 1).reshape(n_layers * height, n_batch * width)

            std::cout << "layer " << i << ": " << mixture.sizes() << " device: " << mixture.device() << std::endl;

        }
    }


//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");

    return a.exec();
}
