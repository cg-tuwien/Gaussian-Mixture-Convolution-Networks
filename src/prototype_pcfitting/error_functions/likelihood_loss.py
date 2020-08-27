from prototype_pcfitting import ErrorFunction
import torch
import gmc.mixture as gm


class LikelihoodLoss(ErrorFunction):
    # Calculates an error by calculating the likelihood of the point cloud given the mixture

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        mixture_with_inversed_cov = gm.pack_mixture(gmamplitudes, gmpositions, gminvcovariances)
        output = gm.evaluate_inversed(mixture_with_inversed_cov, pcbatch).view(batch_size, -1)
        return -torch.mean(torch.log(output + 0.00001), dim=1)
