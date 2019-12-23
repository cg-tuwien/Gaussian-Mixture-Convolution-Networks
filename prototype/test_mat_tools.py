import unittest
import timeit
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
    def test_batched_index_select(self):
        a = torch.tensor([[[1, 1337], [2, 1337], [3, 1337]], [[4, 1337], [5, 1337], [6, 1337]], [[7, 1337], [8, 1337], [9, 1337]]])
        i = torch.tensor([[0, 1], [0, 2], [1, 2]])
        b = mat_tools.batched_index_select(a, 1, i)
        self.assertEqual(len(b.shape), 3)
        self.assertEqual(b.shape[0], a.shape[0])
        self.assertEqual(b.shape[1], i.shape[1])
        self.assertEqual(b.shape[2], a.shape[2])

        target = torch.tensor([[[1, 1337],
                                [2, 1337]],
                               [[4, 1337],
                                [6, 1337]],
                               [[8, 1337],
                                [9, 1337]]])

        self.assertAlmostEqual((b - target).abs().sum().item(), 0)

    def test_flatten_index(self):
        indices = torch.arange(3 * 5 * 7 * 11).view(3, 5, 7, 11)
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    for l in range(11):
                        coordinates = torch.tensor([i, j, k, l], dtype=torch.long).view(1, -1).expand((13, -1))
                        found_index = mat_tools.flatten_index(coordinates, indices.shape)
                        for m in range(13):
                            self.assertEqual(indices[i, j, k, l].item(), found_index[m].item())


    def test_triangle_conv(self):
        for dims in range(2, 4):
            Atri = mat_tools.gen_random_positive_definite_triangle(20, dims)
            Amats = mat_tools.triangle_to_normal(Atri)
            Atri_p = mat_tools.normal_to_triangle(Amats)
            self.assertTrue(torch.all(Atri == Atri_p))

            Bmats = torch.rand((dims, dims, 20)) * 2 - 1
            Bmats += Bmats.transpose(0, 1).contiguous()
            # Bmats is symmetric now, but not necessarily positive definite
            Btri = mat_tools.normal_to_triangle(Bmats)
            Bmats_p = mat_tools.triangle_to_normal(Btri)
            self.assertTrue(torch.all(Bmats == Bmats_p))


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

    def test_triangle_xAx2(self):
        n_tests = 50
        for dims in range(2, 4):
            M = mat_tools.gen_random_positive_definite_triangle(n_tests, dims)
            M_mats = mat_tools.triangle_to_normal(M)

            xes = nprnd.rand(dims)
            xesMxes = mat_tools.triangle_xAx(M, torch.tensor(xes))
            self.assertEqual(n_tests, xesMxes.size()[0])

            for i in range(M.size()[1]):
                np_result = xes @ M_mats[:, :, i].numpy() @ xes
                self.assertAlmostEqual(np_result, xesMxes[i].item(), 5)

    def test_triangle_det(self):
        for dims in range(2, 4):
            # do not use mat_tools.gen* here!
            (A, B, M) = _triangle_mat_data(dims)

            dets = mat_tools.triangle_det(M)
            self.assertEqual(dets.size()[0], 2)
            self.assertAlmostEqual(npla.det(A), dets[0].item())
            self.assertAlmostEqual(npla.det(B), dets[1].item())

    def test_triangle_invert(self):
        for dims in range(2, 4):
            a_matri = mat_tools.gen_random_positive_definite_triangle(20, dims)

            ainv_matri = mat_tools.triangle_invert(a_matri)

            ainv_mat = mat_tools.triangle_to_normal(ainv_matri)
            a_mat = mat_tools.triangle_to_normal(a_matri)
            for i in range(a_mat.size()[2]):
                a = a_mat[:, :, i].numpy()
                good_inv = npla.inv(a)
                quest_inv = ainv_mat[:, :, i].numpy()
                self.assertAlmostEqual(((good_inv - quest_inv)**2).sum(), 0, 5)

            for n_components in (100, 1000, 10000):
                b_matri = mat_tools.gen_random_positive_definite_triangle(n_components, dims)

                def bench():
                    mat_tools.triangle_invert(b_matri)
                print(f"benchmark_triangle_invert ({dims} dims, {n_components} components): {timeit.timeit(bench, number = 5) / 5}")

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

    def test_gen_random_pd_triangle(self):
        covs = mat_tools.gen_random_positive_definite_triangle(100, 2)
        self.assertTrue(torch.all(mat_tools.triangle_det(covs) > 0))

        covs = mat_tools.gen_random_positive_definite_triangle(100, 3)
        self.assertTrue(torch.all(mat_tools.triangle_det(covs) > 0))

if __name__ == '__main__':
    unittest.main()
