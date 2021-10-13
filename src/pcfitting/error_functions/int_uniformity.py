import math
from typing import List

import trimesh

from pcfitting import EvalFunction
import torch
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import numpy as np
import time
from scipy.integrate import dblquad
from pcfitting.cpp.gmeval import pyeval

class IntUniformity(EvalFunction):
    # This class was an experiment to calculate the true standard deviation of density values a long a surface
    # by applying numerical methods rather than using an evaluation point cloud.
    # It takes way too long.

    def __init__(self, unscaled: bool = True, scaled: bool = True, mean: bool = True):
        self._unscaled = unscaled
        self._scaled = scaled
        self._mean = mean
        pass

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = gmpositions.shape[0]
        assert batch_size == 1
        mixture_with_inversed_cov = gm.pack_mixture(gmamplitudes, gmpositions, gminvcovariances)
        dt = mixture_with_inversed_cov.dtype

        # see https://math.stackexchange.com/questions/3395992/surface-integral-over-an-arbitrary-triangle-in-3d
        # and https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html

        mesh = trimesh.load_mesh(modelpath)
        # Calculate Mean
        totalmean = 0
        for i in range(len(mesh.faces)):
            print((float(i)/len(mesh.faces)))
            p0 = mesh.vertices[mesh.faces[i,0]]
            p1 = mesh.vertices[mesh.faces[i,1]]
            p2 = mesh.vertices[mesh.faces[i,2]]
            n = mesh.face_normals[i]
            x3d = lambda u, v: p0 + u*(p1-p0) + v*(p2-p0)
            x3dt = lambda u, v: torch.tensor(x3d(u, v)).to(dtype=dt).cuda().view(1, 1, 1, 3)
            sub = np.linalg.norm(p1-p0)*np.linalg.norm(p2-p0)
            integrant = lambda u, v: gm.evaluate_inversed_with_amplitude_mixture(mixture_with_inversed_cov, x3dt(u, v)).item()
            mean = sub * dblquad(integrant, 0, 1, lambda x: 0, lambda x: 1-x)[0] / mesh.area_faces[i]
            totalmean += mean
        # Calculate Std
        totalstd = 0
        for i in range(len(mesh.faces)):
            print((float(i) / len(mesh.faces)))
            p0 = mesh.vertices[mesh.faces[i, 0]]
            p1 = mesh.vertices[mesh.faces[i, 1]]
            p2 = mesh.vertices[mesh.faces[i, 2]]
            x3d = lambda u, v: p0 + u*(p1-p0) + v*(p2-p0)
            x3dt = lambda u, v: torch.tensor(x3d(u, v)).to(dtype=dt).cuda().view(1, 1, 1, 3)
            sub = np.linalg.norm(p1-p0)*np.linalg.norm(p2-p0)
            integrant = lambda u, v: (gm.evaluate_inversed_with_amplitude_mixture(mixture_with_inversed_cov, x3dt(u, v)).item() - mean) ** 2
            stabw = sub * dblquad(integrant, 0, 1, lambda x: 0, lambda x: 1-x)[0] / mesh.area_faces[i]
            totalstd += stabw
        res = torch.zeros((self._unscaled + self._scaled)*(1+self._mean), batch_size, device=pcbatch.device, dtype=pcbatch.dtype)
        sfnn, sfnnl = self.calculate_scale_factors_nn(pcbatch)
        i = 0
        if self._unscaled:
            if self._mean:
                res[i, 0] = totalmean
                i += 1
            res[i, 0] = totalstd
            i += 1
        if self._scaled:
            if self._mean:
                res[i, 0] = totalmean * sfnn
                i += 1
            res[i, 0] = totalstd
            i += 1
        return res

    def get_names(self) -> List[str]:
        nlst = []
        if self._unscaled:
            if self._mean:
                nlst.append("Integral based Density Mean")
            nlst.append("Integral based Density Variation")
        if self._scaled:
            if self._mean:
                nlst.append("Normalized Integral based Density Mean")
            nlst.append("Normalized Integral based Density Variation")
        return nlst

    def calculate_scale_factors_nn(self, pcbatch: torch.Tensor) -> (float, float):
        if not hasattr(pcbatch, "nnscalefactor"):
            md = pyeval.calc_rmsd_to_itself(pcbatch.view(-1, 3))[1]
            refdist = 128 / (2*math.sqrt(pcbatch.shape[1]) - 1)
            pcbatch.nnscalefactor = refdist / md
        return (math.pow(pcbatch.nnscalefactor, -3), -3 * math.log(pcbatch.nnscalefactor))
