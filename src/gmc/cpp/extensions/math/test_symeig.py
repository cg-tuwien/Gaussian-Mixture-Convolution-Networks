import unittest

import torch

import gmc.cpp.extensions.math.symeig as symeig


class MyTestCase(unittest.TestCase):
    def test_forward2d_cpu(self):
        t = torch.rand(10, 1, 20, 2, 2)
        t = t @ torch.transpose(t, -1, -2)
        t += torch.eye(2).view(1, 1, 2, 2) * 0.000001
        self.assertTrue((torch.det(t) > 0).all().item())

        torch_eigenvalues, torch_eigenvectors = torch.symeig(t, True)
        my_eigenvalues, my_eigenvectors = symeig.cpu.forward(t)
        self.assertEqual(my_eigenvalues.shape, torch_eigenvalues.shape)
        self.assertEqual(my_eigenvectors.shape, torch_eigenvectors.shape)

        # torch eigenvectors are sometimes oriented into the other direction, therefore we can't compare them by value.
        # but we can test the mathematical way
        self.assertTrue(torch.all((my_eigenvectors.norm(dim=-1) - 1)**2 < 0.000000000001).item())
        self.assertTrue(torch.all((my_eigenvalues.unsqueeze(-2) - (t @ my_eigenvectors.transpose(-1, -2) / my_eigenvectors.transpose(-1, -2)))**2 < 0.0000001).item())


if __name__ == '__main__':
    unittest.main()
