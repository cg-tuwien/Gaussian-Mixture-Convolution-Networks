from typing import List

from pcfitting import EvalFunction, GMSampler
import torch
from .reconstructionstats import ReconstructionStats
import gmc.mixture as gm
import trimesh
import trimesh.proximity
from pysdf import SDF
import open3d as o3d

class ReconstructionStatsProjected(EvalFunction):
    # Decorator for ReconstructionStats. Adds a projection preprocessing stage that projects reconstructed points
    # onto the surface.

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
        sampled = GMSampler.sampleGMM_ext(gmm, self._recstat._samplepoints)

        return self.calculate_score_on_reconstructed(pcbatch, sampled, modelpath)

    def calculate_score_on_reconstructed(self, pcbatch: torch.Tensor, sampled: torch.Tensor,
                                         modelpath: str = None) -> torch.Tensor:

        # mesh = trimesh.load_mesh(modelpath)
        # query = trimesh.proximity.ProximityQuery(mesh)
        # closest, distance, triangle_id = query.on_surface(sampled[0].cpu().numpy())
        # sampled = torch.from_numpy(closest).cuda()
        meshO = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(modelpath))
        query_point = o3d.core.Tensor(sampled.view(-1, 3).cpu().numpy(), dtype=o3d.core.Dtype.Float32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(meshO)
        ans = scene.compute_closest_points(query_point)
        projected = torch.from_numpy(ans['points'].numpy()).cuda()

        return self._recstat.calculate_score_on_reconstructed(pcbatch, projected, modelpath)

    def get_names(self) -> List[str]:
        list = self._recstat.get_names()
        for i in range(len(list)):
            list[i] = "Projected: " + list[i]
        return list
