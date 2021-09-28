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

    def test_my_symeig_gradcheck(self):

        def symmetrise_and_symeig(m: torch.Tensor):
            orig_shape = m.shape
            if m.shape[-1] == 2:
                d = (m[..., 0, 1] + m[..., 1, 0]) * 0.5
                m = torch.cat([m[..., 0, 0].unsqueeze(-1), d.unsqueeze(-1), d.unsqueeze(-1), m[..., 1, 1].unsqueeze(-1)], dim=-1).view(orig_shape)
            else:
                assert m.shape[-1] == 3
                m00 = (m[..., 0, 0]).unsqueeze(-1)
                m01 = ((m[..., 0, 1] + m[..., 1, 0]) * 0.5).unsqueeze(-1)
                m02 = ((m[..., 0, 2] + m[..., 2, 0]) * 0.5).unsqueeze(-1)
                m11 = (m[..., 1, 1]).unsqueeze(-1)
                m12 = ((m[..., 1, 2] + m[..., 2, 1]) * 0.5).unsqueeze(-1)
                m22 = (m[..., 2, 2]).unsqueeze(-1)
                m = torch.cat([m00, m01, m02, m01, m11, m12, m02, m12, m22], dim=-1).view(orig_shape)
            return mat_tools.symeig(m)

        print("test_my_symeig_gradcheck")
        eps = 1e-9
        for cuda in (False, True):
            for n_dims in (2, 3):
                print(f"cuda={cuda}, n_dims={n_dims}")
                n = 100
                if n_dims == 2:
                    v1 = torch.rand(n, 2) * 2 - 1
                    v1 = v1 / v1.norm(dim=-1).unsqueeze(dim=-1)
                    v2 = v1 @ torch.tensor([[0.0, -1], [1, 0]])
                    V = torch.cat([v1.unsqueeze(-1), v2.unsqueeze(-1)], 2)
                else:
                    v1 = torch.rand(n, 3) * 2 - 1
                    v1 = v1 / v1.norm(dim=-1).unsqueeze(dim=-1)
                    v2 = torch.rand(n, 3) * 2 - 1
                    v2 = v2 / v2.norm(dim=-1).unsqueeze(dim=-1)
                    v3 = torch.cross(v1, v2)
                    v3 = v3 / v3.norm(dim=-1).unsqueeze(dim=-1)
                    v2 = torch.cross(v3, v1)
                    V = torch.cat([v1.unsqueeze(-1), v2.unsqueeze(-1), v3.unsqueeze(-1)], 2)

                L = torch.rand(n, n_dims) + torch.arange(n_dims).unsqueeze(0) * 1.5
                V = V.double()
                L = L.double()
                m = V @ torch.diag_embed(L) @ V.transpose(-1, -2)

                # m: Tensor = torch.rand(10, n_dims, n_dims)
                # m = m @ m.transpose(-1, -2) + torch.eye(n_dims).view(1, n_dims, n_dims) * 0.1
                if cuda:
                    m = m.cuda()
                m.requires_grad = True

                test = torch.autograd.gradcheck(symmetrise_and_symeig, (m), eps=eps, atol=1e-3, nondet_tol=1e-6)
                self.assertTrue(test)

    def test_my_symeig_performance(self):
        print("")
        def t(n_dims):
            m: Tensor = torch.rand(100 * 1000, n_dims, n_dims)
            m = m @ m.transpose(-1, -2) + torch.eye(n_dims).view(1, n_dims, n_dims) * 0.01
            m = m.cuda()
            m.requires_grad = True
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            eigvals, eigvecs = torch.linalg.eigh(m)
            (eigvals.sum() + eigvecs.sum()).backward()

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"torch.linalg.eigh + backward for {m.shape[0]} elements in {n_dims}d: {t1 - t0}")
            m.grad = None
            eigvals, eigvecs = mat_tools.symeig(m)
            (eigvals.sum() + eigvecs.sum()).backward()

            torch.cuda.synchronize()
            t2 = time.perf_counter()
            print(f"my symeig +         backward for {m.shape[0]} elements in {n_dims}d: {t2 - t1}")

        t(2)
        t(3)


if __name__ == '__main__':
    unittest.main()
