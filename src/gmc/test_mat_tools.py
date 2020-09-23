import unittest
import time
import torch
import numpy as np
import numpy.linalg as npla
import numpy.random as nprnd
from torch import Tensor

from gmc import mat_tools


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

    def test_my_matrix_inverse(self):
        def t(n_dims):
            m: Tensor = torch.rand(23, n_dims, n_dims)
            m = m @ m.transpose(-1, -2) + torch.eye(n_dims).view(1, n_dims, n_dims) * 0.01
            m = m.cuda()
            m.requires_grad = True
            torch_inverse = m.inverse()
            torch_inverse.sum().backward()
            torch_grad = m.grad.clone()
            m.grad = None
            my_inverse = mat_tools.inverse(m)
            my_inverse.sum().backward()
            my_grad = m.grad.clone()
            self.assertTrue(((torch_inverse - my_inverse) ** 2).mean().item() < 0.00001)
            self.assertTrue(((torch_grad - my_grad) ** 2).mean().item() < 0.0001)
        t(2)
        t(3)

    def test_my_matrix_inverse_performance(self):
        def t(n_dims):
            m: Tensor = torch.rand(100*10*1000, n_dims, n_dims)
            m = m @ m.transpose(-1, -2) + torch.eye(n_dims).view(1, n_dims, n_dims) * 0.01
            m = m.cuda()
            m.requires_grad = True
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            torch_inverse = m.inverse()
            torch_inverse.sum().backward()

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"torch_inverse + torch_inverse.backward for {m.shape[0]} elements in {n_dims}d: {t1-t0}")
            m.grad = None
            my_inverse = mat_tools.inverse(m)
            my_inverse.sum().backward()

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f"my_inverse + my_inverse.backward for {m.shape[0]} elements in {n_dims}d: {t2-t1}")
        t(2)
        t(3)


if __name__ == '__main__':
    unittest.main()
