from abc import ABC, abstractmethod

import torch
from prototype_pcfitting import GMLogger


class GMMGenerator(ABC):
    # Abstract class for GMM generators.

    @abstractmethod
    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        # Gets a point cloud batch of size [m,n,3]
        # where m is the batch size and n the point count.
        # Point cloud is given in the right scale already!
        # It might be given an initial gaussian mixture of
        # size [m,1,g,10] where m is the batch size and g
        # the number of Gaussians.
        # It returns two gaussian mixtures of sizes
        # [m,1,g,10], the first being a mixture with amplitudes as weights
        # the second a mixture where the weights describe the priors.
        # Parameters have to be set in the other methods
        # of the class
        pass

    @abstractmethod
    def set_logging(self, logger: GMLogger = None):
        pass
