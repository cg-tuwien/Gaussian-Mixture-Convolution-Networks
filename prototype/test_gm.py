import unittest
import torch
import numpy as np
import numpy.random as nprnd
import numpy.linalg as npla
import scipy.signal
import matplotlib.pyplot as plt

import gm

from torch import Tensor


def _triangle_mat_data(dims: int) -> (np.array, np.array, Tensor):
    assert dims == 2 or dims == 3
    A = np.array(([[3., 1., 2.],
                   [1., 8., 3.],
                   [2., 3., 4.]]))
    B = np.array(([[5., 4., 2.],
                   [4., 7., 1.],
                   [2., 1., 6.]]))

    if dims == 2:
        A = A[0:2, 0:2]
        B = B[0:2, 0:2]

    return A, B


class TestGM(unittest.TestCase):
    # todo: test convolution by covolving discretised versions of GMs

    def test_gm_eval(self):
        n_eval_positions = 50
        for dims in range(2, 3):
            weights = nprnd.rand(2) * 2 - 1.0
            positions = nprnd.rand(dims, 2) * 10 - 5.0
            (A, B) = _triangle_mat_data(dims)
            covs = (A, B)

            mixture = gm.single_batch_mixture(torch.tensor(weights, dtype=torch.float32), torch.tensor(positions, dtype=torch.float32).t(), torch.tensor(covs, dtype=torch.float32))

            eval_positions = nprnd.rand(dims, n_eval_positions)
            values_gm = mixture.evaluate_many_xes(torch.tensor(eval_positions, dtype=torch.float32).t().reshape(1, n_eval_positions, dims)).view(-1).numpy()
            values_gm2 = mixture.evaluate_few_xes(torch.tensor(eval_positions, dtype=torch.float32).t().reshape(1, n_eval_positions, dims)).view(-1).numpy()

            for i in range(n_eval_positions):
                np_result = 0
                for j in range(0, 2):
                    x = eval_positions[:, i]
                    pos = positions[:, j]
                    xmp = x - pos
                    cov_i = npla.inv(covs[j])
                    exponent = -0.5 * (xmp @ cov_i @ xmp)
                    np_result += weights[j] * np.exp(exponent)
                self.assertAlmostEqual(np_result, values_gm[i].item(), 5)
                self.assertAlmostEqual(np_result, values_gm2[i].item(), 5)

    def test_polynomMulRepeat(self):
        A: torch.Tensor = torch.tensor([[[1, 2, 3, 4],
                                         [1, 1, 1, 1],
                                         [4, 2, 3, 1],
                                         [4, 2, 3, 1]],
                                        [[10, 20, 30, 40],
                                         [10, 10, 10, 10],
                                         [40, 20, 30, 10],
                                         [40, 20, 30, 10]]], dtype=torch.float32)  # testing with col# = 3 is not propper.
        A = A.transpose(1, 2)

        B: torch.Tensor = torch.tensor([[[1, 2],
                                         [3, 6],
                                         [2, 1],
                                         [1, 2]],
                                        [[10, 20],
                                         [30, 60],
                                         [20, 10],
                                         [10, 20]]], dtype=torch.float32)
        B = B.transpose(1, 2)

        (Ap, Bp) = gm._polynomMulRepeat(A, B)
        Ap = Ap.transpose(1, 2)
        Bp = Bp.transpose(1, 2)

        self.assertEqual(Ap.size()[0], 2)
        self.assertEqual(Bp.size()[0], 2)
        self.assertEqual(Ap.size()[1], 4)
        self.assertEqual(Bp.size()[1], 4)
        self.assertEqual(Ap.size()[2], 8)
        self.assertEqual(Bp.size()[2], 8)

        AtimesB = Ap * Bp
        R = torch.sum(AtimesB, 2)

        self.assertAlmostEqual(R[0, 0].item(), 30.)
        self.assertAlmostEqual(R[0, 1].item(), 4 * 3 + 4 * 6)
        self.assertAlmostEqual(R[0, 2].item(), 30.)
        self.assertAlmostEqual(R[0, 3].item(), 30.)
        self.assertAlmostEqual(R[1, 0].item(), 3000.)
        self.assertAlmostEqual(R[1, 1].item(), 100 * (4 * 3 + 4 * 6))
        self.assertAlmostEqual(R[1, 2].item(), 3000.)
        self.assertAlmostEqual(R[1, 3].item(), 3000.)
        sorted = AtimesB.sort().values
        self.assertAlmostEqual(((sorted[:, 0, :] - sorted[:, 2, :]) ** 2).sum().item(), 0)
        self.assertAlmostEqual(((sorted[:, 3, :] - sorted[:, 2, :]) ** 2).sum().item(), 0)

    def test_convolution(self):
        n_batches = 3
        gm1 = gm.generate_random_mixtures(n_batches, 3, n_dims=2, pos_radius=1, cov_radius=0.5)
        gm2 = gm.generate_random_mixtures(n_batches, 4, n_dims=2, pos_radius=1, cov_radius=0.5)
        gmc = gm.convolve(gm1, gm2)
        samples_per_unit = 50

        xv, yv = torch.meshgrid([torch.arange(-6, 6, 1/samples_per_unit, dtype=torch.float),
                                 torch.arange(-6, 6, 1/samples_per_unit, dtype=torch.float)])
        size = xv.size()[0]
        xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2).expand(n_batches, -1, 2)
        gm1_samples = gm1.evaluate_many_xes(xes).view(n_batches, size, size).numpy()
        gm2_samples = gm2.evaluate_many_xes(xes).view(n_batches, size, size).numpy()
        gmc_samples = gmc.evaluate_many_xes(xes).view(n_batches, size, size).numpy()

        for i in range(n_batches):
            reference_solution = scipy.signal.fftconvolve(gm1_samples[i, :, :], gm2_samples[i, :, :], 'same')\
                                 / (samples_per_unit * samples_per_unit)
            our_solution = gmc_samples[i, :, :]
            # plt.imshow(gm1_samples[i, :, :]); plt.colorbar(); plt.show()
            # plt.imshow(gm2_samples[i, :, :]); plt.colorbar(); plt.show()
            # plt.imshow(reference_solution); plt.colorbar(); plt.show()
            # plt.imshow(our_solution); plt.colorbar(); plt.show()

            max_l2_err = ((reference_solution - our_solution) ** 2).max()
            # plt.imshow((reference_solution - our_solution)); plt.colorbar(); plt.show();
            assert max_l2_err < 0.001



if __name__ == '__main__':
    unittest.main()
