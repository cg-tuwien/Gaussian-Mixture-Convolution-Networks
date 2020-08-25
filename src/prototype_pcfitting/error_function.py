from abc import ABC, abstractmethod
import torch


class ErrorFunction(ABC):
    # Abstract base class for Error Functions

    @abstractmethod
    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor) -> torch.Tensor:
        # Gets a point cloud batch of size [m,n,3] where m is the batch size
        # and n is the point count, as well as a Gaussian mixture containing
        # g Gaussians, given by their positions [m,1,g,3], covariances [m,1,g,3,3],
        # inverse covariances [m,1,g,3,3] and amplitudes [m,1,g].
        # Returns a tensor of size [m] where each entry contains the
        # score / loss
        pass