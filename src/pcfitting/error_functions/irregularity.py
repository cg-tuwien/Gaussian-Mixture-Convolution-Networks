import math
from typing import List

import trimesh

from pcfitting import EvalFunction
import torch
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import numpy as np
import time
import os
from pcfitting.cpp.gmeval import pyeval
from pcfitting.generators.em_tools import EMTools

class Irregularity(EvalFunction):
    # Calculates the smoothness according to our own method

    def __init__(self, nn: int = 17, subsamples: int = -1):
        self._nn = nn
        self._subsamples = subsamples
        pass

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = gmpositions.shape[0]
        assert batch_size == 1
        mixture_with_inversed_cov = gm.pack_mixture(gmamplitudes, gmpositions, gminvcovariances)
        dt = mixture_with_inversed_cov.dtype
        point_count = pcbatch.shape[1]
        points = pcbatch.view(1, 1, -1, 3)

        # shape: (np, nn)
        nngraph = None
        if modelpath is not None:
            file = modelpath + ".nngraph-e" + str(point_count) + "-n" + str(self._nn) + (("_s" + str(self._subsamples)) if self._subsamples > 0 else "") + ".torch"
            if os.path.exists(file):
                nngraph = torch.load(file).cuda()
            else:
                print("Calculating nngraph")
                if self._subsamples > 0:
                    nngraph = pyeval.nn_graph_sub(pcbatch.view(-1, 3), self._subsamples, self._nn).cuda()
                else:
                    nngraph = pyeval.nn_graph(pcbatch.view(-1, 3), self._nn).cuda()
                torch.save(nngraph, file)
                print("Saved nngraph")

        output = torch.zeros(point_count, dtype=dt, device=mixture_with_inversed_cov.device)
        # output[int(point_count/2):] = 1
        subbatches = math.ceil((point_count) / 65535)
        subbatch_pointcount = math.ceil(point_count / subbatches)
        for p in range(subbatches):
            startidx = p * subbatch_pointcount
            endidx = min((p + 1) * subbatch_pointcount, point_count)
            output[startidx:endidx] = \
                gm.evaluate_inversed_with_amplitude_mixture(mixture_with_inversed_cov, points[:, :, startidx:endidx, :]).view(-1)

        if self._subsamples > 0:
            smooth = pyeval.irregularity_sub(output.cpu(), nngraph.cpu())
        else:
            smooth = pyeval.irregularity(output.cpu(), nngraph.cpu())

        res = torch.zeros(1, 1, device=pcbatch.device, dtype=pcbatch.dtype)
        res[0, 0] = smooth
        return res

    def get_names(self) -> List[str]:
        nlst = []
        nlst.append("SoD")
        return nlst

    def calculate_scale_factors_nn(self, pcbatch: torch.Tensor) -> (float, float):
        if not hasattr(pcbatch, "nnscalefactor"):
            md = pyeval.calc_rmsd_to_itself(pcbatch.view(-1, 3))[1]
            refdist = 128 / (2*math.sqrt(pcbatch.shape[1]) - 1)
            pcbatch.nnscalefactor = refdist / md
        return (math.pow(pcbatch.nnscalefactor, -3), -3 * math.log(pcbatch.nnscalefactor))
