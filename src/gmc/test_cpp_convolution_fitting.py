import unittest
import time
import typing

import torch.autograd
from torch import Tensor
import matplotlib.pyplot as plt

import gmc.cpp.extensions.convolution_fitting.binding as convolution_fitting
import gmc.mixture as gm
import gmc.render as render

position_radius = 10
covariance_radius = 10


def debug_render(mixture: Tensor, radius: int = 3, image_size: typing.Tuple[int, int] = (200, 200), clamp: typing.Tuple[float, float] = (-1.5, 1.5)):
    mixture = mixture.view(1, gm.n_batch(mixture), gm.n_components(mixture), -1)
    images = render.render(mixture, torch.zeros(1, 1), x_low=-radius, x_high=radius, y_low=-radius, y_high=radius, width=image_size[0], height=image_size[1])
    images = render.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
    return images[:, :, :3]


class CppConvolutionFittingTest(unittest.TestCase):
    def _test_forward(self, n_dims: int):
        n_batches = 100
        gm1 = gm.generate_random_mixtures(n_batches, 1, 5, n_dims=n_dims, pos_radius=1, cov_radius=0.5)
        gm2 = gm.generate_random_mixtures(1, 1, 4, n_dims=n_dims, pos_radius=1, cov_radius=0.5)
        python_result = gm.convolve(gm1, gm2)
        cpp_result = convolution_fitting.apply(gm1, gm2, 1, 1)[0]

        # plt.imshow(debug_render(gm1))
        # plt.show()
        # plt.imshow(debug_render(gm2))
        # plt.show()
        # plt.imshow(debug_render(python_result))
        # plt.show()
        # plt.imshow(debug_render(cpp_result))
        # plt.show()

        sampling_positions = torch.rand(1, 1, 10000, n_dims) * 4 - 2
        python_sampling = gm.evaluate(python_result, sampling_positions)
        cpp_sampling = gm.evaluate(cpp_result, sampling_positions)
        self.assertLess(((python_sampling - cpp_sampling)**2).mean().sqrt(), 0.000001)

    def test_forward_2d(self):
        self._test_forward(2)

    def test_forward_3d(self):
        self._test_forward(3)


sh = CppConvolutionFittingTest()
sh.test_forward_2d()
sh.test_forward_3d()
