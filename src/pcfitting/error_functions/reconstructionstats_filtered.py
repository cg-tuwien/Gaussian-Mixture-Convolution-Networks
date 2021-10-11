from typing import List

from pcfitting import EvalFunction, GMSampler, data_loading
import torch
from .reconstructionstats import ReconstructionStats
from .reconstructionstats_projected import ReconstructionStatsProjected
import gmc.mixture as gm
import trimesh
import trimesh.proximity
from pysdf import SDF
import numpy as np
import time
import open3d as o3d

class ReconstructionStatsFiltered(EvalFunction):

    def __init__(self, recstat: [ReconstructionStats, ReconstructionStatsProjected] = None, thresh = -1.0):
        if recstat is None:
            recstat = ReconstructionStats()
        self._recstat = recstat
        self._thresh = thresh

    def set_threshold(self, thresh: float):
        self._thresh = thresh

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = pcbatch.shape[0]
        assert batch_size == 1

        gmm = gm.convert_amplitudes_to_priors(gm.pack_mixture(gmamplitudes, gmpositions, gmcovariances))
        if hasattr(self._recstat, "_samplepoints"):
            sampled = GMSampler.sampleGMM_ext(gmm, self._recstat._samplepoints)
        else:
            sampled = GMSampler.sampleGMM_ext(gmm, self._recstat._recstat._samplepoints)

        return self.calculate_score_on_reconstructed(pcbatch, sampled, modelpath)

    def calculate_score_on_reconstructed(self, pcbatch: torch.Tensor, sampled: torch.Tensor,
                                         modelpath: str = None) -> torch.Tensor:

        rcn = sampled.view(-1, 3).cpu().numpy()
        thresh = self._thresh
        if self._thresh < 0:
            evn = pcbatch.view(-1, 3).cpu().numpy()
            sdfR = SDF(rcn, np.zeros((0, 3)))
            d2 = torch.linalg.norm(torch.tensor(evn - rcn[sdfR.nn(evn)]), dim=1)
            thresh = d2.max()

        meshO = o3d.t.geometry.TriangleMesh.from_legacy(o3d.io.read_triangle_mesh(modelpath))
        query_point = o3d.core.Tensor(sampled.view(-1, 3).cpu().numpy(), dtype=o3d.core.Dtype.Float32)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(meshO)
        distances = torch.from_numpy(scene.compute_distance(query_point).numpy()).cuda()
        closest = sampled[0, distances.lt(thresh), :]
        print(closest.shape[0])
        #data_loading.write_pc_to_off(modelpath + str(time.time()) + ".recf.off", closest)

        return self._recstat.calculate_score_on_reconstructed(pcbatch, closest, modelpath)

    def get_names(self) -> List[str]:
        list = self._recstat.get_names()
        for i in range(len(list)):
            list[i] = "Filtered: " + list[i]
        return list
