import unittest

import torch
import numpy as np
import numpy.linalg as npla
import numpy.random as nprnd
from torch import Tensor

import mat_tools


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


class MatToolsTest(unittest.TestCase):
    def test_triangle_xAx(self):
        for dims in range(2, 4):
            (A, B, M) = _triangle_mat_data(dims)

            xes = nprnd.rand(dims, 5)
            xesAxes = mat_tools.triangle_xAx(M[:, 0], torch.tensor(xes))
            xesBxes = mat_tools.triangle_xAx(M[:, 1], torch.tensor(xes))

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

            dets = mat_tools.triangle_det(M)
            self.assertEqual(dets.size()[0], 2)
            self.assertAlmostEqual(npla.det(A), dets[0].item())
            self.assertAlmostEqual(npla.det(B), dets[1].item())

    def test_triangle_matmul(self):
        for dims in range(2, 4):
            # do not use gm.gen* here!
            (A, B, M) = _triangle_mat_data(dims)

            result = mat_tools.triangle_matmul(M[:, 0].view(-1, 1), M[:, 1].view(-1, 1))
            self.assertEqual(result.size()[1], 1)
            np_result = A @ B
            if dims == 2:
                self.assertAlmostEqual(np_result[0, 0], result[0].item())
                self.assertAlmostEqual(np_result[0, 1], result[1].item())
                self.assertAlmostEqual(np_result[1, 1], result[2].item())
            else:
                self.assertAlmostEqual(np_result[0, 0], result[0].item())
                self.assertAlmostEqual(np_result[0, 1], result[1].item())
                self.assertAlmostEqual(np_result[0, 2], result[2].item())
                self.assertAlmostEqual(np_result[1, 1], result[3].item())
                self.assertAlmostEqual(np_result[1, 2], result[4].item())
                self.assertAlmostEqual(np_result[2, 2], result[5].item())


if __name__ == '__main__':
    unittest.main()
