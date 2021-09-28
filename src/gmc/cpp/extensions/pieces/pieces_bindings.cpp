#include "pieces/pieces.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_inverse", &pieces::matrix_inverse, "matrix inverse forward (CPU and CUDA))");
    m.def("symeig", &pieces::symeig, "eigenvalue decomposition for symmetric matrices forward (CPU and CUDA))");
    m.def("symeig_backward", &pieces::symeig_backward, "eigenvalue decomposition for symmetric matrices backward (CPU and CUDA))");
//     m.def("backward", &parallel_backward, "evaluate_inversed parallel backward (CPU and CUDA)");
}

