#include "pieces/pieces.h"

#include <cassert>

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "pieces/matrix_inverse.h"

// We can't use a single implementatino file because the cuda compiler doesn't like pybind11 (i guess)
// Ye, and it certainly doesn't like <torch/extension.h> (depending on the version of pytorch / pybind / cuda / gcc)

namespace pieces {

at::Tensor matrix_inverse(const at::Tensor& matrices)
{
    at::cuda::OptionalCUDAGuard device_guard;
    if (matrices.is_cuda()) {
        assert (device_of(matrices).has_value());
        device_guard.set_device(device_of(matrices).value());
    }
    return matrix_inverse_impl::forward(matrices);

}


}
