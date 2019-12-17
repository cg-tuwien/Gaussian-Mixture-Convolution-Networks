import unittest
import torch
import numpy as np
import numpy.random as nprnd
import numpy.linalg as npla
import scipy.signal
import matplotlib.pyplot as plt

import gm

from torch import Tensor
from gm import Mixture


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

        xv, yv = torch.meshgrid([torch.arange(-6, 6, 1 / samples_per_unit, dtype=torch.float),
                                 torch.arange(-6, 6, 1 / samples_per_unit, dtype=torch.float)])
        size = xv.size()[0]
        xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2).expand(n_batches, -1, 2)
        gm1_samples = gm1.evaluate_many_xes(xes).view(n_batches, size, size).numpy()
        gm2_samples = gm2.evaluate_many_xes(xes).view(n_batches, size, size).numpy()
        gmc_samples = gmc.evaluate_many_xes(xes).view(n_batches, size, size).numpy()

        for i in range(n_batches):
            reference_solution = scipy.signal.fftconvolve(gm1_samples[i, :, :], gm2_samples[i, :, :], 'same') \
                                 / (samples_per_unit * samples_per_unit)
            our_solution = gmc_samples[i, :, :]
            # plt.imshow(gm1_samples[i, :, :]); plt.colorbar(); plt.show()
            # plt.imshow(gm2_samples[i, :, :]); plt.colorbar(); plt.show()
            # plt.imshow(reference_solution); plt.colorbar(); plt.show()
            # plt.imshow(our_solution); plt.colorbar(); plt.show()

            max_l2_err = ((reference_solution - our_solution) ** 2).max()
            # plt.imshow((reference_solution - our_solution)); plt.colorbar(); plt.show();
            assert max_l2_err < 0.001

    def test_batch_sum(self):
        ms = [Mixture(torch.tensor([[1, 2],
                                    [3, 4]]),


                      torch.tensor([[[1.0, 1.1],
                                     [2.0, 2.1]],

                                    [[3.0, 3.1],
                                     [4.0, 4.1]]]),


                      torch.tensor([[[[1.9, 1.1],
                                      [1.1, 1.2]],

                                     [[2.9, 2.1],
                                      [2.1, 2.2]]],

                                    [[[3.9, 3.1],
                                      [3.1, 3.2]],

                                     [[4.9, 4.1],
                                      [4.1, 4.2]]]])),

              Mixture(torch.tensor([[5, 6],
                                    [7, 8]]),


                      torch.tensor([[[5.0, 5.1],
                                     [6.0, 6.1]],

                                    [[7.0, 7.1],
                                     [8.0, 8.1]]]),


                      torch.tensor([[[[5.9, 5.1],
                                      [5.1, 5.2]],

                                     [[6.9, 6.1],
                                      [6.1, 6.2]]],

                                    [[[7.9, 7.1],
                                      [7.1, 7.2]],

                                     [[8.9, 8.1],
                                      [8.1, 8.2]]]]))
              ]

        m = gm.batch_sum(ms)

        self.assertAlmostEqual((m.weights - torch.tensor([[1, 2, 3, 4],
                                                          [5, 6, 7, 8]])).abs().sum(), 0)
        self.assertAlmostEqual((m.positions - torch.tensor([[[1.0000, 1.1000],
                                                             [2.0000, 2.1000],
                                                             [3.0000, 3.1000],
                                                             [4.0000, 4.1000]],

                                                            [[5.0000, 5.1000],
                                                             [6.0000, 6.1000],
                                                             [7.0000, 7.1000],
                                                             [8.0000, 8.1000]]])).abs().sum(), 0)
        self.assertAlmostEqual((m.covariances - torch.tensor([[[[1.9000, 1.1000],
                                                                [1.1000, 1.2000]],

                                                               [[2.9000, 2.1000],
                                                                [2.1000, 2.2000]],

                                                               [[3.9000, 3.1000],
                                                                [3.1000, 3.2000]],

                                                               [[4.9000, 4.1000],
                                                                [4.1000, 4.2000]]],

                                                              [[[5.9000, 5.1000],
                                                                [5.1000, 5.2000]],

                                                               [[6.9000, 6.1000],
                                                                [6.1000, 6.2000]],

                                                               [[7.9000, 7.1000],
                                                                [7.1000, 7.2000]],

                                                               [[8.9000, 8.1000],
                                                                [8.1000, 8.2000]]]])).abs().sum(), 0)

    def test_mixture_normalisation(self):
        data_in = gm.MixtureReLUandBias(gm.generate_random_mixtures(10, 3, 2, pos_radius=3, cov_radius=0.3), torch.rand(10) + 0.2)
        data_in.mixture.weights[0:4, 0] = 2
        data_in.mixture.positions[0, 0, 0] = -10
        data_in.mixture.positions[1, 0, 1] = 10
        data_in.mixture.covariances[2, 0, 0, 0] = 400
        data_in.mixture.covariances[3, 0, 1, 1] = 400
        data_in.mixture.update_inverted_covariance()

        data_normalised, norm_factors = gm.normalise(data_in)
        data_out = gm.de_normalise(data_normalised.mixture, norm_factors)

        self.assertEqual(data_in.bias.shape, data_normalised.bias.shape)
        self.assertAlmostEqual((data_in.mixture.weights / data_in.bias.view(-1, 1) - data_normalised.mixture.weights / data_normalised.bias.view(-1, 1)).abs().mean().item(), 0, places=4)
        self.assertAlmostEqual((data_in.mixture.weights - data_out.weights).abs().mean().item(), 0, places=5)
        self.assertAlmostEqual((data_in.mixture.positions - data_out.positions).abs().mean().item(), 0, places=5)
        self.assertAlmostEqual((data_in.mixture.covariances - data_out.covariances).abs().mean().item(), 0, places=5)

        # for i in range(10):
        #     data_in.mixture.debug_show(i, -10, -10, 10, 10, 0.1)
        #     data_normalised.mixture.debug_show(i, -1.5, -1.5, 1.5, 1.5, 0.01)
        #     data_out.debug_show(i, -10, -10, 10, 10, 0.1)
        #     input("Press enter to continue!")

    def test_component_select(self):
        n_batches = 10
        gm_large = gm.generate_random_mixtures(n_batches, n_components=200, n_dims=2)
        gm_a = gm_large.select_components(0, 100)
        gm_b = gm_large.select_components(100, gm_large.n_components())

        self.assertEqual(gm_a.n_layers(), n_batches)
        self.assertEqual(gm_b.n_layers(), n_batches)
        self.assertEqual(gm_a.n_components(), 100)
        self.assertEqual(gm_b.n_components(), 100)

        gm_concatenated = gm.cat((gm_a, gm_b), dim=1)
        self.assertEqual(gm_concatenated.n_layers(), n_batches)
        self.assertEqual(gm_concatenated.n_components(), gm_large.n_components())
        self.assertEqual(gm_concatenated.n_dimensions(), gm_large.n_dimensions())

        self.assertTrue(((gm_concatenated.weights - gm_large.weights).abs() < 0.00001).all())
        self.assertTrue(((gm_concatenated.positions - gm_large.positions).abs() < 0.00001).all())
        self.assertTrue(((gm_concatenated.covariances - gm_large.covariances).abs() < 0.00001).all())


if __name__ == '__main__':
    unittest.main()
