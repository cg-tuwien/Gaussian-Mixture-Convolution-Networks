#ifndef GPE_PIECES_MATRIX_INVERSE_H
#define GPE_PIECES_MATRIX_INVERSE_H

#include <torch/types.h>


namespace pieces {
namespace matrix_inverse_impl {

at::Tensor forward(const torch::Tensor& matrices);

} // namespace matrix_inverse_impl
} // namespace pieces

#endif // GPE_PIECES_MATRIX_INVERSE_H
