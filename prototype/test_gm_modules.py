import unittest
import torch
import numpy as np
import numpy.random as nprnd
import numpy.linalg as npla
import scipy.signal
import matplotlib.pyplot as plt

import gm
import gm_modules


class TestGM(unittest.TestCase):
    # todo: test convolution by covolving discretised versions of GMs

    def test_convolution(self):
        n_batches = 3
        n_layers_in = 4
        n_layers_out = 5
        gm_in = gm.generate_random_mixtures(n_batches, n_layers_in, 3, n_dims=2, pos_radius=1, cov_radius=0.25)
        conv_layer = gm_modules.GmConvolution(n_layers_in = n_layers_in, n_layers_out = n_layers_out, n_dims=2, position_range = 1, covariance_range = 0.25, weight_sd=1)

        gm_out = conv_layer(gm_in)
        samples_per_unit = 50

        xv, yv = torch.meshgrid([torch.arange(-6, 6, 1 / samples_per_unit, dtype=torch.float),
                                 torch.arange(-6, 6, 1 / samples_per_unit, dtype=torch.float)])
        size = xv.size()[0]
        xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)
        gm_in_samples = gm.evaluate(gm_in, xes).numpy()
        gm_out_samples = gm.evaluate(gm_out.detach(), xes).numpy()

        for l in range(n_layers_out):
            gm_kernel_samples = gm.evaluate(conv_layer.kernel(l).detach(), xes).view(n_layers_in, size, size).numpy()

            for b in range(n_batches):
                reference_solution = np.zeros((size, size))
                for k in range(n_layers_in):
                    kernel = gm_kernel_samples[k, :].reshape(size, size)
                    data = gm_in_samples[b, k, :].reshape(size, size)
                    reference_solution += scipy.signal.fftconvolve(data, kernel, 'same') / (samples_per_unit * samples_per_unit)
                    # plt.imshow(reference_solution); plt.colorbar(); plt.show()
                our_solution = gm_out_samples[b, l, :].reshape(size, size)
                reference_solution = reference_solution[1:, 1:]
                our_solution = our_solution[:-1, :-1]
                # plt.imshow(reference_solution); plt.colorbar(); plt.show()
                # plt.imshow(our_solution); plt.colorbar(); plt.show()

                max_l2_err = ((reference_solution - our_solution) ** 2).max()
                # plt.imshow((reference_solution - our_solution)); plt.colorbar(); plt.show();
                assert max_l2_err < 0.0000001


if __name__ == '__main__':
    unittest.main()
