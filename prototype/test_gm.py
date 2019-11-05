import unittest
import torch
import numpy as np

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

    M = np.array([A[np.triu_indices(dims)],
                  B[np.triu_indices(dims)]]).transpose()
    M = torch.tensor(M)

    return A, B, M


class TestGM(unittest.TestCase):
    def test_gm_eval(self):
        for dims in range(2, 4):
            factors = nprnd.rand(2) * 2 - 1.0
            positions = nprnd.rand(dims, 2) * 10 - 5.0
            (A, B, covs_torch) = _triangle_mat_data(dims)
            covs = (A, B)

            mixture = gm.Mixture(torch.tensor(factors), torch.tensor(positions), covs_torch)

            np_result = 0
            eval_positions = nprnd.rand(dims, 20)
            values_gm = mixture.evaluate(torch.tensor(eval_positions)).numpy()

            for i in range(20):
                np_result = 0
                for j in range(0, 2):
                    x = eval_positions[:, i]
                    pos = positions[:, j]
                    xmp = x - pos
                    cov_i = npla.inv(covs[j])
                    exponent = -0.5 * (xmp @ cov_i @ xmp)
                    np_result += factors[j] * np.exp(exponent)
                self.assertAlmostEqual(np_result, values_gm[i].item())

    def test_polynomMulRepeat(self):
        A: torch.Tensor = torch.tensor([[1, 2, 3, 4],
                                        [1, 1, 1, 1],
                                        [4, 2, 3, 1],
                                        [4, 2, 3, 1]], dtype=torch.float32)    # testing with col# = 3 is not propper.

        B: torch.Tensor = torch.tensor([[1, 2],
                                        [3, 6],
                                        [2, 1],
                                        [1, 2]], dtype=torch.float32)

        (Ap, Bp) = gm._polynomMulRepeat(A, B)

        self.assertEqual(Ap.size()[0], 4)
        self.assertEqual(Bp.size()[0], 4)
        self.assertEqual(Ap.size()[1], 8)
        self.assertEqual(Bp.size()[1], 8)

        AtimesB = Ap * Bp
        R = torch.sum(AtimesB, 1)
        
        self.assertAlmostEqual(R[0].item(), 30.)
        self.assertAlmostEqual(R[1].item(), 4*3 + 4*6)
        self.assertAlmostEqual(R[2].item(), 30.)
        self.assertAlmostEqual(R[3].item(), 30.)
        sorted = AtimesB.sort().values
        self.assertAlmostEqual(torch.sum(torch.abs(sorted[0]-sorted[2]), 0).item(), 0)
        self.assertAlmostEqual(torch.sum(torch.abs(sorted[3]-sorted[2]), 0).item(), 0)

    def test_gen_cov(self):
        covs = gm._gen_random_covs(100, 2)
        self.assertTrue(torch.all(gm._triangle_determinants(covs) > 0))

        covs = gm._gen_random_covs(100, 3)
        self.assertTrue(torch.all(gm._triangle_determinants(covs) > 0))


if __name__ == '__main__':
    unittest.main()

