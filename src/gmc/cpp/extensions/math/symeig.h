#ifndef MATH_EIGEN_H
#define MATH_EIGEN_H

#include <vector>
#include <torch/script.h>

std::vector<torch::Tensor> eigen_cpu_forward(torch::Tensor matrices);

#endif
