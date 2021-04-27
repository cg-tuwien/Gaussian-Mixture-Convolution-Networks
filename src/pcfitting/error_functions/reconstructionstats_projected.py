from typing import List

from pcfitting import EvalFunction, GMSampler
import torch
from .reconstructionstats import ReconstructionStats
import gmc.mixture as gm
import trimesh
import trimesh.proximity

class ReconstructionStatsProjected(EvalFunction):

    def __init__(self, recstat: ReconstructionStats = None):
        if recstat is None:
            recstat = ReconstructionStats()
        self._recstat = recstat

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        assert batch_size == 1

        gmm = gm.convert_amplitudes_to_priors(gm.pack_mixture(gmamplitudes, gmpositions, gmcovariances))
        sampled = GMSampler.sampleGMM(gmm, self._recstat._samplepoints)

        return self.calculate_score_on_reconstructed(pcbatch, sampled, modelpath)

    def calculate_score_on_reconstructed(self, pcbatch: torch.Tensor, sampled: torch.Tensor,
                                         modelpath: str = None) -> torch.Tensor:

        mesh = trimesh.load_mesh(modelpath)
        query = trimesh.proximity.ProximityQuery(mesh)
        closest, distance, triangle_id = query.on_surface(sampled[0].cpu().numpy())
        sampled = torch.from_numpy(closest).cuda()

        return self._recstat.calculate_score_on_reconstructed(pcbatch, sampled, modelpath)

    def get_names(self) -> List[str]:
        list = self._recstat.get_names()
        for i in range(len(list)):
            list[i] = "Projected: " + list[i]
        return list
