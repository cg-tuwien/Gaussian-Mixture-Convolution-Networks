import unittest

import numpy as np
import numpy.random as nprnd
import numpy.linalg as npla
import scipy.signal
import torch
from torch import Tensor

import gmc.mixture as gm


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
    def test_gm_eval(self):
        n_eval_positions = 50
        for dims in range(2, 3):
            weights = nprnd.rand(2) * 2 - 1.0
            positions = nprnd.rand(dims, 2) * 10 - 5.0
            (A, B) = _triangle_mat_data(dims)
            covs = (A, B)

            mixture = gm.pack_mixture(torch.tensor(weights, dtype=torch.float32).view(1, 1, -1),
                                      torch.tensor(positions, dtype=torch.float32).t().view(1, 1, -1, dims),
                                      torch.tensor(covs, dtype=torch.float32).view(1, 1, -1, dims, dims))

            eval_positions = nprnd.rand(dims, n_eval_positions)
            values_gm = gm.evaluate(mixture, torch.tensor(eval_positions, dtype=torch.float32).t().reshape(1, 1, n_eval_positions, dims)).view(-1).numpy()

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

    def test_polynomMulRepeat(self):
        A: torch.Tensor = torch.tensor([[[[1, 2, 3, 4],
                                          [1, 1, 1, 1],
                                          [4, 2, 3, 1],
                                          [4, 2, 3, 1]],
                                         [[10, 20, 30, 40],
                                          [10, 10, 10, 10],
                                          [40, 20, 30, 10],
                                          [40, 20, 30, 10]]]], dtype=torch.float32)  # testing with col# = 3 is not propper.
        A = A.transpose(2, 3)

        B: torch.Tensor = torch.tensor([[[[1, 2],
                                          [3, 6],
                                          [2, 1],
                                          [1, 2]],
                                         [[10, 20],
                                          [30, 60],
                                          [20, 10],
                                          [10, 20]]]], dtype=torch.float32)
        B = B.transpose(2, 3)

        (Ap, Bp) = gm._polynomial_mul_repeat(A, B)
        Ap = Ap.transpose(2, 3)
        Bp = Bp.transpose(2, 3)

        self.assertEqual(Ap.shape[0], 1)
        self.assertEqual(Bp.shape[0], 1)
        self.assertEqual(Ap.shape[1], 2)
        self.assertEqual(Bp.shape[1], 2)
        self.assertEqual(Ap.shape[2], 4)
        self.assertEqual(Bp.shape[2], 4)
        self.assertEqual(Ap.shape[3], 8)
        self.assertEqual(Bp.shape[3], 8)

        AtimesB = Ap * Bp
        AtimesB = AtimesB.view(2, 4, 8)
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
        gm1 = gm.generate_random_mixtures(n_batches, 1, 3, n_dims=2, pos_radius=1, cov_radius=0.5)
        gm2 = gm.generate_random_mixtures(n_batches, 1, 4, n_dims=2, pos_radius=1, cov_radius=0.5)
        gmc = gm.convolve(gm1, gm2)
        samples_per_unit = 50

        xv, yv = torch.meshgrid([torch.arange(-6, 6, 1 / samples_per_unit, dtype=torch.float),
                                 torch.arange(-6, 6, 1 / samples_per_unit, dtype=torch.float)])
        size = xv.size()[0]
        xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2).expand(n_batches, 1, -1, 2)
        gm1_samples = gm.evaluate(gm1, xes).view(n_batches, size, size).numpy()
        gm2_samples = gm.evaluate(gm2, xes).view(n_batches, size, size).numpy()
        gmc_samples = gm.evaluate(gmc, xes).view(n_batches, size, size).numpy()

        for i in range(n_batches):
            reference_solution = scipy.signal.fftconvolve(gm1_samples[i, :, :], gm2_samples[i, :, :], 'same') \
                                 / (samples_per_unit * samples_per_unit)
            our_solution = gmc_samples[i, :, :]
            # plt.imshow(gm1_samples[i, :, :]); plt.colorbar(); plt.show()
            # plt.imshow(gm2_samples[i, :, :]); plt.colorbar(); plt.show()
            reference_solution = reference_solution[1:, 1:]
            our_solution = our_solution[:-1, :-1]
            # plt.imshow(reference_solution); plt.colorbar(); plt.show()
            # plt.imshow(our_solution); plt.colorbar(); plt.show()

            max_l2_err = ((reference_solution - our_solution) ** 2).max()
            # plt.imshow((reference_solution - our_solution)); plt.colorbar(); plt.show();
            self.assertLess(max_l2_err, 0.0000001)

    def test_spatial_scale(self):
        n_batch = 10
        n_layers = 8
        n_components = 3

        # different scale per dimensions
        for n_dims in range(2, 4):
            mixture_in = gm.generate_random_mixtures(n_batch=n_batch, n_layers=n_layers, n_components=n_components, n_dims=n_dims, pos_radius=1, cov_radius=1, weight_min=-0.5, weight_max=4)
            scaling_factors = torch.rand(1, n_layers, n_dims) * 20

            xv, yv, zv = torch.meshgrid([torch.arange(-1, 1, 0.25, dtype=torch.float),
                                     torch.arange(-1, 1, 0.25, dtype=torch.float),
                                     torch.arange(-1, 1, 0.25, dtype=torch.float)])
            xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)), 1).view(1, 1, -1, n_dims)

            eval_reference = gm.evaluate(mixture_in, xes)
            mixture_scaled = gm.spatial_scale(mixture_in, scaling_factors)
            xes_scaled = xes.repeat(n_batch, n_layers, 1, 1) * scaling_factors.view(1, n_layers, 1, n_dims)
            eval_scaled = gm.evaluate(mixture_scaled, xes_scaled)

            self.assertLess((eval_reference - eval_scaled).abs().mean().item(), 0.000001)

        #uniform scale
        for n_dims in range(2, 4):
            mixture_in = gm.generate_random_mixtures(n_batch=n_batch, n_layers=n_layers, n_components=n_components, n_dims=n_dims, pos_radius=1, cov_radius=1, weight_min=-0.5, weight_max=4)
            scaling_factors = torch.rand(n_batch, n_layers, 1) * 20

            xv, yv, zv = torch.meshgrid([torch.arange(-1, 1, 0.25, dtype=torch.float),
                                     torch.arange(-1, 1, 0.25, dtype=torch.float),
                                     torch.arange(-1, 1, 0.25, dtype=torch.float)])
            xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1)), 1).view(1, 1, -1, n_dims)

            eval_reference = gm.evaluate(mixture_in, xes)
            mixture_scaled = gm.spatial_scale(mixture_in, scaling_factors)
            xes_scaled = xes.repeat(n_batch, n_layers, 1, 1) * scaling_factors.view(n_batch, n_layers, 1, 1)
            eval_scaled = gm.evaluate(mixture_scaled, xes_scaled)

            self.assertLess((eval_reference - eval_scaled).abs().mean().item(), 0.000001)

    def test_mixture_integration(self):
        for n_dims in range(2, 4):
            print(n_dims)
            n_batch = 2
            n_layers = 4
            gm1 = gm.generate_random_mixtures(n_batch=n_batch, n_layers=n_layers, n_components=3, n_dims=n_dims, pos_radius=0.5, cov_radius=0.2)
            gm1_covs = gm.covariances(gm1)
            gm1_covs += (torch.eye(n_dims) * 0.05).view(1, 1, 1, n_dims, n_dims)  # tested: changes gm1
            n_samples = 40000000
            integration_area_side_length = 10

            xes = (torch.rand(1, 1, n_samples, n_dims) - 0.5) * integration_area_side_length

            gm1_samples = gm.evaluate(gm1, xes)

            reference_solution = gm1_samples.sum(2).cpu() / (n_samples / (integration_area_side_length**n_dims))
            my_solution = gm.integrate(gm1)

            for b in range(n_batch):
                for l in range(n_layers):
                    self.assertLess(abs(my_solution[b, l].item() - reference_solution[b, l].item()), 0.016)

    def test_normal_amplitude(self):
        for n_dims in range(2, 4):
            print(n_dims)
            n_batch = 2
            n_layers = 4
            gm1 = gm.generate_random_mixtures(n_batch=n_batch, n_layers=n_layers, n_components=3, n_dims=n_dims, pos_radius=0.5, cov_radius=0.2)
            gm1_covs = gm.covariances(gm1)
            gm1_covs += (torch.eye(n_dims) * 0.05).view(1, 1, 1, n_dims, n_dims)
            gm1 = gm.pack_mixture(gm.normal_amplitudes(gm1_covs), gm.positions(gm1), gm1_covs)

            self.assertAlmostEqual(gm.integrate_components(gm1).mean().item(), 1.0, places=6)
            self.assertAlmostEqual(gm.integrate_components(gm1).var().item(), 0.0, places=6)

            gm1_cov_inversed = gm.covariances(gm1).inverse().transpose(-2, -1)
            self.assertTrue((gm.normal_amplitudes(gm.covariances(gm1)) - gm.normal_amplitudes_inversed(gm1_cov_inversed) < 0.000001).all())


if __name__ == '__main__':
    unittest.main()
