#include <torch/torch.h>


int main() {
    torch::Tensor d;
    torch::load(d, "/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch0_netlayer0.tensor");

    return 0;
}
