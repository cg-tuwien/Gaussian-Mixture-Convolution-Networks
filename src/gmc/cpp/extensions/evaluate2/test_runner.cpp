#include <iostream>
#include <string>
//#include <torch/torch.h>
#include <torch/script.h>


int main() {
    torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0.pt");
    auto list = container.attributes();
    for (int i = 0; i < 3; i++) {

        auto a = container.attr(std::to_string(i)).toTensor().cuda();
        std::cout << "layer " << i << ": " << a.sizes() << " device: " << a.device() << std::endl;

    }

//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");

    return 0;
}
