import unittest
import torch
import numpy as np
import numpy.random as nprnd
import numpy.linalg as npla

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
    M = torch.tensor(M, dtype=torch.float32)

    return A, B, M


class TestGM(unittest.TestCase):
    # todo: test convolution by covolving discretised versions of GMs

    def test_gm_eval(self):
        n_eval_positions = 50
        for dims in range(2, 3):
            factors = nprnd.rand(2) * 2 - 1.0
            positions = nprnd.rand(dims, 2) * 10 - 5.0
            (A, B, covs_torch) = _triangle_mat_data(dims)
            covs = (A, B)

            mixture = gm.Mixture(torch.tensor(factors, dtype=torch.float32), torch.tensor(positions, dtype=torch.float32), covs_torch)

            eval_positions = nprnd.rand(dims, n_eval_positions)
            values_gm = mixture.evaluate_many_xes(torch.tensor(eval_positions, dtype=torch.float32)).numpy()
            values_gm2 = mixture.evaluate_few_xes(torch.tensor(eval_positions, dtype=torch.float32)).numpy()

            for i in range(n_eval_positions):
                np_result = 0
                for j in range(0, 2):
                    x = eval_positions[:, i]
                    pos = positions[:, j]
                    xmp = x - pos
                    cov_i = npla.inv(covs[j])
                    exponent = -0.5 * (xmp @ cov_i @ xmp)
                    np_result += factors[j] * np.exp(exponent)
                self.assertAlmostEqual(np_result, values_gm[i].item(), 5)
                self.assertAlmostEqual(np_result, values_gm2[i].item(), 5)

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


if __name__ == '__main__':
    unittest.main()

