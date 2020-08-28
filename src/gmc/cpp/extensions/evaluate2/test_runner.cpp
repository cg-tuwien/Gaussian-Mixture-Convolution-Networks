//#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>

int main() {
    torch::jit::script::Module container = torch::jit::load("/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");
    auto list = container.attributes();
    auto a = container.attr("m").toTensor();

    std::cout << a.sizes() << std::endl;
//    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");

    return 0;
}
