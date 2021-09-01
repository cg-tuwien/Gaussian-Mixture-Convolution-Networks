import unittest
import time
import typing

import torch.autograd
from torch import Tensor
import matplotlib.pyplot as plt

import gmc.cpp.extensions.convolution.binding as cpp_convolution
import gmc.cpp.extensions.convolution_fitting.binding as cpp_convolution_fitting
import gmc.mixture as gm
import gmc.render as render

position_radius = 10
covariance_radius = 10


def debug_render(mixture: Tensor, radius: int = 3, image_size: typing.Tuple[int, int] = (200, 200), clamp: typing.Tuple[float, float] = (-1.5, 1.5)):
    mixture = mixture.view(1, gm.n_batch(mixture), gm.n_components(mixture), -1)
    images = render.render(mixture, torch.zeros(1, 1), x_low=-radius, x_high=radius, y_low=-radius, y_high=radius, width=image_size[0], height=image_size[1])
    images = render.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
    return images[:, :, :3]


def cpp_convolution_fitting_wrapper(m1: Tensor, m2: Tensor, n_fitting_components: int):
    m1p = gm.convert_amplitudes_to_priors(m1)
    m2p = gm.convert_amplitudes_to_priors(m2)
    mop = cpp_convolution_fitting.apply(m1p, m2p, n_fitting_components)
    return gm.convert_priors_to_amplitudes(mop)


class CppConvolutionTest(unittest.TestCase):
    def _test_against_python(self, n_dims: int):
        n_batches = 1
        gm1 = gm.generate_random_mixtures(n_batches, 1, 5, n_dims=n_dims, pos_radius=1, cov_radius=0.5)
        gm2 = gm.generate_random_mixtures(1, 1, 4, n_dims=n_dims, pos_radius=1, cov_radius=0.5)
        python_result = gm.convolve(gm1, gm2)
        cpp_result = cpp_convolution.apply(gm1, gm2)
        cpp_result2 = cpp_convolution_fitting_wrapper(gm1, gm2, 4*5)

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
        cpp_sampling2 = gm.evaluate(cpp_result2, sampling_positions)
        self.assertLess(((python_sampling - cpp_sampling)**2).mean().sqrt(), 0.000001)
        self.assertLess(((python_sampling - cpp_sampling2)**2).mean().sqrt(), 0.000001)

    def _test_against_full_convolution(self, n_dims: int):
        n_batches = 10
        n_in_channels = 5
        n_out_channels = 4
        n_in_comps = 32
        n_kernel_comps = 5
        n_out_comps = n_in_channels * n_in_comps * n_kernel_comps
        gm1 = gm.generate_random_mixtures(n_batches, n_in_channels, n_in_comps, n_dims=n_dims, pos_radius=1, cov_radius=0.5)
        gm2 = gm.generate_random_mixtures(n_out_channels, n_in_channels, n_kernel_comps, n_dims=n_dims, pos_radius=1, cov_radius=0.5)

        conv_result = cpp_convolution.apply(gm1, gm2)
        conv_nofit_result = cpp_convolution_fitting_wrapper(gm1, gm2, n_out_comps)
        conv_fit_result = cpp_convolution_fitting_wrapper(gm1, gm2, n_out_comps // 2)

        sampling_positions = torch.rand(1, 1, 10000, n_dims) * 4 - 2
        ref_samples = gm.evaluate(conv_result, sampling_positions)
        nofit_samples = gm.evaluate(conv_nofit_result, sampling_positions)
        fit_samples = gm.evaluate(conv_fit_result, sampling_positions)
        self.assertLess((ref_samples - nofit_samples).abs().max(), 0.0001)
        # render.imshow(conv_result)
        # render.imshow(conv_fit_result)
        self.assertLess(((ref_samples - fit_samples)**2).mean().sqrt(), 0.2)    # the fitting is not very precise..

    def test_forward_2d(self):
        self._test_against_python(2)
        self._test_against_full_convolution(2)

    def test_forward_3d(self):
        self._test_against_python(3)
        self._test_against_full_convolution(3)


sh = CppConvolutionTest()
sh.test_forward_2d()
sh.test_forward_3d()
