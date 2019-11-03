import unittest
import torch
import numpy as np
import numpy.linalg as npla
import numpy.random as nprnd

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

    return (A, B, M)


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

    def test_triangle_xAx(self):
        for dims in range(2, 4):
            (A, B, M) = _triangle_mat_data(dims)

            xes = nprnd.rand(dims, 5)
            xesAxes = gm._xAx_withTriangleA(M[:, 0], torch.tensor(xes))
            xesBxes = gm._xAx_withTriangleA(M[:, 1], torch.tensor(xes))

            self.assertEqual(xesAxes.size()[0], 5)
            self.assertEqual(xesBxes.size()[0], 5)

            for i in range(5):
                np_result_a = xes[:, i] @ A @ xes[:, i]
                self.assertAlmostEqual(np_result_a, xesAxes[i].item())

                np_result_b = xes[:, i] @ B @ xes[:, i]
                self.assertAlmostEqual(np_result_b, xesBxes[i].item())

    
    def test_triangle_det(self):
        for dims in range(2, 4):
            # do not use gm.gen* here!
            (A, B, M) = _triangle_mat_data(dims)
                
            dets = gm._triangle_determinants(M)
            self.assertEqual(dets.size()[0], 2)
            self.assertAlmostEqual(npla.det(A), dets[0].item())
            self.assertAlmostEqual(npla.det(B), dets[1].item())

        def test_triangle_matmul(self):
            for dims in range(2, 4):
                # do not use gm.gen* here!
                (A, B, M) = _triangle_mat_data(dims)

                result = gm._triangle_matmul(M[:, 0].view(-1, 1), M[:, 1].view(-1, 1))
                self.assertEqual(result.size()[1], 1)
                np_result = A @ B
                if dims == 2:
                    self.assertAlmostEqual(np_result[0, 0], result[0])
                    self.assertAlmostEqual(np_result[0, 1], result[1])
                    self.assertAlmostEqual(np_result[1, 1], result[2])
                else:
                    self.assertAlmostEqual(np_result[0, 0], result[0])
                    self.assertAlmostEqual(np_result[0, 1], result[1])
                    self.assertAlmostEqual(np_result[0, 2], result[2])
                    self.assertAlmostEqual(np_result[1, 1], result[3])
                    self.assertAlmostEqual(np_result[1, 2], result[4])
                    self.assertAlmostEqual(np_result[2, 2], result[5])

    
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

