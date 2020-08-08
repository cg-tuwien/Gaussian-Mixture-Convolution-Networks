from abc import ABC, abstractmethod
import torch


class GMMGenerator(ABC):
    # Abstract class for GMM generators.

    @abstractmethod
    def generate(self, pcbatch: torch.Tensor, gmbatch: torch.Tensor = None) -> torch.Tensor:
        # Gets a point cloud batch of size [m,n,3]
        # where m is the batch size and n the point count.
        # Point cloud is given in already the right scale!
        # It might be given an initial gaussian mixture of
        # size [m,1,g,10] where m is the batch size and g
        # the number of Gaussians.
        # It returns a gaussian mixture batch of size
        # [m,1,g,10].
        # Parameters have to be set in the other methods
        # of the class
        pass
