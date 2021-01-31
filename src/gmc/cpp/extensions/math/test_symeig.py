import unittest
import time

import torch

import gmc.cpp.extensions.math.symeig as symeig


class MyTestCase(unittest.TestCase):
    def forward(self, N_DIMS: int, device: str, double: bool):
        print(f"testing {device}")
        t = torch.rand(10, 5, 100, N_DIMS, N_DIMS, device=device)
        PRECISION = 1.5E-3                  # warning: indicates possible numerical issues. but it doesn't matter for gmcn since they are only used to compute the AABBs for the BVH in case of evaluation.
        if double:
            PRECISION = 1.0E-10
            t = t.double()
        t = t @ torch.transpose(t, -1, -2)
        t += torch.eye(N_DIMS, device=device).view(1, 1, 1, N_DIMS, N_DIMS) * 0.000001
        self.assertGreater(torch.det(t).min().item(), 0)

        t0 = time.perf_counter()
        torch_eigenvalues, torch_eigenvectors = torch.symeig(t, True)
        t1 = time.perf_counter()
        my_eigenvalues, my_eigenvectors = symeig.apply(t)
        t2 = time.perf_counter()

        print(f"{device} torch: {t1 - t0}, mine: {t2 - t1}")

        self.assertEqual(my_eigenvalues.shape, torch_eigenvalues.shape)
        self.assertEqual(my_eigenvectors.shape, torch_eigenvectors.shape)

        # length 1
        self.assertLess(((torch_eigenvectors.norm(dim=-1) - 1) ** 2).max().item(), PRECISION)
        self.assertLess(((my_eigenvectors.norm(dim=-1) - 1) ** 2).max().item(), PRECISION)

        # orthogonal
        for dim in range(N_DIMS):
            self.assertLess((torch_eigenvectors[..., 0].view(-1, 1, N_DIMS) @ torch_eigenvectors[..., 1].view(-1, N_DIMS, 1)).abs().max(), PRECISION)
            self.assertLess((my_eigenvectors[..., 0].view(-1, 1, N_DIMS) @ my_eigenvectors[..., 1].view(-1, N_DIMS, 1)).abs().max(), PRECISION)
            if N_DIMS > 2:
                self.assertLess((torch_eigenvectors[..., 0].view(-1, 1, N_DIMS) @ torch_eigenvectors[..., 2].view(-1, N_DIMS, 1)).abs().max(), PRECISION)
                self.assertLess((torch_eigenvectors[..., 1].view(-1, 1, N_DIMS) @ torch_eigenvectors[..., 2].view(-1, N_DIMS, 1)).abs().max(), PRECISION)
                self.assertLess((my_eigenvectors[..., 0].view(-1, 1, N_DIMS) @ my_eigenvectors[..., 2].view(-1, N_DIMS, 1)).abs().max(), PRECISION)
                self.assertLess((my_eigenvectors[..., 1].view(-1, 1, N_DIMS) @ my_eigenvectors[..., 2].view(-1, N_DIMS, 1)).abs().max(), PRECISION)

        # A @ eigenvector = eigenvalue * eigenvector
        self.assertLess((((t @ torch_eigenvectors) - torch_eigenvectors * torch_eigenvalues.unsqueeze(-2)).abs().max().item()), PRECISION)

        errors = ((t @ my_eigenvectors) - my_eigenvectors * my_eigenvalues.unsqueeze(-2))
        # print(f"worst precision error: {errors.abs().max()}, root mean square: {torch.sqrt(torch.mean(errors ** 2)).item()}")
        worst_input = t.view(-1, N_DIMS, N_DIMS)[errors.abs().max(dim=-1).values.max(dim=-1).values.view(-1).max(dim=0).indices]
        # print(f"worst input: {worst_input}, it's det = {worst_input.det()}")
        self.assertLess(errors.abs().max().item(), PRECISION)
        self.assertLess(torch.sqrt(torch.mean(errors ** 2)).item(), PRECISION)

    def test_forward2d_cpu_double(self):
        self.forward(2, "cpu", double=True)

    def test_forward2d_cpu_float(self):
        self.forward(2, "cpu", double=False)

    def test_forward2d_cuda_double(self):
        self.forward(2, "cuda", double=True)

    def test_forward2d_cuda_float(self):
        self.forward(2, "cuda", double=False)

    # def test_forward3d_cpu(self):
    #     torch.set_printoptions(precision=8, threshold=None, edgeitems=None, linewidth=None, profile=None, sci_mode=None)
        # t = torch.tensor([[1.4844, 1.7474, 1.8749],
        # [1.7474, 2.1518, 2.3024],
        # [1.8749, 2.3024, 2.4764]]).double()
        # print(f"t.det()={t.det()}")
        # my_eigenvalues, my_eigenvectors = symeig.apply(t)
        # print(f"dots: {torch.dot(my_eigenvectors[0], my_eigenvectors[1])}, {torch.dot(my_eigenvectors[0], my_eigenvectors[2])}, {torch.dot(my_eigenvectors[0], my_eigenvectors[2])}")
        # torch_eigenvalues, torch_eigenvectors = torch.symeig(t.double(), True)
        # errors = ((my_eigenvalues.unsqueeze(-2) - (t @ my_eigenvectors.transpose(-1, -2) / my_eigenvectors.transpose(-1, -2))) ** 2)
        #
        # print(f"values torch: \n{torch_eigenvalues}, \n ours:\n {my_eigenvalues}");
        # print(f"vectors torch: \n{torch_eigenvectors},\n ours:\n{my_eigenvectors}")
        # print("\n")
        # print(f"errors= {errors}")
        # self.forward(3, "cpu", double=True)
        # self.forward(3, "cpu", double=False)

    def test_forward3d_cpu_double(self):
        self.forward(3, "cpu", double=True)

    def test_forward3d_cpu_float(self):
        self.forward(3, "cpu", double=False)

    def test_forward3d_cuda_double(self):
        self.forward(3, "cuda", double=True)

    def test_forward3d_cuda_float(self):
        self.forward(3, "cuda", double=False)


if __name__ == '__main__':
    unittest.main()
