import unittest
import timeit
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



if __name__ == '__main__':
    unittest.main()
