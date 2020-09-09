import unittest
import time

import torch

import gmc.cpp.extensions.math.symeig as symeig


class MyTestCase(unittest.TestCase):
    def forward2d(self, device: str):
        print(f"testing {device}")
        t = torch.rand(100, 8, 125, 2, 2, device=device)
        t = t @ torch.transpose(t, -1, -2)
        t += torch.eye(2, device=device).view(1, 1, 2, 2) * 0.000001
        self.assertTrue((torch.det(t) > 0).all().item())

        t0 = time.perf_counter()
        torch_eigenvalues, torch_eigenvectors = torch.symeig(t, True)
        t1 = time.perf_counter()
        my_eigenvalues, my_eigenvectors = symeig.apply(t)
        t2 = time.perf_counter()

        print(f"{device} torch: {t1-t0}, mine: {t2-t1}")

        self.assertEqual(my_eigenvalues.shape, torch_eigenvalues.shape)
        self.assertEqual(my_eigenvectors.shape, torch_eigenvectors.shape)

        # torch eigenvectors are sometimes oriented into the other direction, therefore we can't compare them by value.
        # but we can test the mathematical way
        self.assertTrue(torch.all((my_eigenvectors.norm(dim=-1) - 1)**2 < 0.000000000001).item())
        self.assertTrue(torch.all((my_eigenvalues.unsqueeze(-2) - (t @ my_eigenvectors.transpose(-1, -2) / my_eigenvectors.transpose(-1, -2)))**2 < 0.0000001).item())

    def test_forward2d_cpu(self):
        self.forward2d("cpu")

    def test_forward2d_cuda(self):
        self.forward2d("cuda")

if __name__ == '__main__':
    unittest.main()
