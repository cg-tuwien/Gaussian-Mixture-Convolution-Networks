#include "pieces/pieces.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("integrate_inversed_forward", &integrate::inversed_forward, "integrate inversed forward (CPU and CUDA))");
    m.def("integrate_forward", &integrate::forward, "integrate forward (CPU and CUDA))");
//     m.def("backward", &parallel_backward, "evaluate_inversed parallel backward (CPU and CUDA)");
}

