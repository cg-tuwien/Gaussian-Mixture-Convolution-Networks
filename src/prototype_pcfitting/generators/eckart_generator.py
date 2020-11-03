from prototype_pcfitting import GMMGenerator, GMLogger, data_loading
from prototype_pcfitting import TerminationCriterion, MaxIterationTerminationCriterion
from gmc.cpp.extensions.furthest_point_sampling import furthest_point_sampling
import torch
import gmc.mixture as gm
import numpy

class EckartGenerator(GMMGenerator):
    # GMM Generator using Expectation Sparsification by Eckart

    def __init__(self,
                 n_gaussians_per_level: int,
                 n_levels: int,
                 dtype: torch.dtype = torch.float32):
        self._n_gaussians_per_level = n_gaussians_per_level
        self._n_levels = n_levels
        self._dtype = dtype

    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        assert (gmbatch is None), "EckartGenerator cannot improve existing GMMs"

        batch_size = pcbatch.shape[0]

        assert (batch_size is 1), "EckartGenerator currently does not support batchsizes > 1"
        point_count = pcbatch.shape[1]
        pcbatch = pcbatch.to(self._dtype).cuda()

        self._eps = (torch.eye(3, 3, dtype=self._dtype) * 1e-6).view(1, 1, 1, 3, 3) \
            .expand(batch_size, 1, self._n_gaussians, 3, 3).cuda()

        parents = torch.zeros(1, point_count).to(self._dtype).cuda()
        parents[:, :] = -1

        # Iterate over levels
        for l in range(self._n_levels):
            # Iterate over GMMs for this level
            for j in range(self._n_gaussians_per_level ** l):
                # Calculate index of this GMM
                # Find all points that have this GMM as their parent
                # Scale them to the unit cube
                # Create initial GMM
                # Iterate E- and M-steps
                # Calculate the new parent indizes for these points
                pass

