#include "evaluate_inversed/evaluate_inversed.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("parallel_forward", &evaluate_inversed::parallel_forward, "evaluate_inversed parallel forward (CPU and CUDA optimised))");
    m.def("parallel_backward", &evaluate_inversed::parallel_backward, "evaluate_inversed parallel backward (CPU and CUDA optimised)");
}
