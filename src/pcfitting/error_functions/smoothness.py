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

class Smoothness(EvalFunction):
    # Calculates the smoothness according to locally consistent gmm paper

    def __init__(self, nn: int = 17):
        self._nn = 17
        pass

    def calculate_score(self, pcbatch: torch.Tensor, gmpositions: torch.Tensor, gmcovariances: torch.Tensor,
                        gminvcovariances: torch.Tensor, gmamplitudes: torch.Tensor,
                        noisecontribution: torch.Tensor = None, modelpath: str = None) -> torch.Tensor:
        batch_size = gmpositions.shape[0]
        assert batch_size == 1
        mixture_with_inversed_cov = gm.pack_mixture(gmamplitudes, gmpositions, gminvcovariances)
        dt = mixture_with_inversed_cov.dtype
        point_count = pcbatch.shape[1]

        # shape: (np, nn)
        nngraph = None
        if modelpath is not None:
            file = modelpath + ".nngraph-e" + str(point_count) + "-n" + str(self._nn) + ".torch"
            if os.path.exists(file):
                nngraph = torch.load(file).cuda()
            else:
                print("Calculating nngraph")
                nngraph = pyeval.nn_graph(pcbatch.view(-1, 3), self._nn).cuda()
                torch.save(nngraph, file)
                print("Saved nngraph")

        gcount = gmpositions.shape[2]
        gm_data = EMTools.TrainingData(1, gcount, gmpositions.dtype, 1)
        gm_data.set_positions(gmpositions[0:1], torch.tensor([True]))
        gm_data.set_covariances(gmcovariances[0:1], torch.tensor([True]))
        gm_data.set_amplitudes(gmamplitudes[0:1], torch.tensor([True]))
        print("E-Step")
        # shape: (np, ng)
        responsibilities = EMTools.expectation(pcbatch.view(1, 1, -1, 1, 3), gm_data, gcount, torch.tensor([True]), -1, 10000)[0].squeeze()

        del gm_data

        print("R-calc")
        R = pyeval.smoothness(responsibilities.to(torch.double).cpu(), nngraph.cpu())
        # R = 0
        # for i_start in range(0, point_count, 1000):
        #     i_end = i_start + 1000
        #     i_batch_size = min(point_count, i_end) - i_start
        #     for j_start in range(0, point_count, 500):
        #         j_end = j_start + 500
        #         j_batch_size = min(point_count, j_end) - j_start
        #         skds = torch.zeros(i_batch_size, j_batch_size, dtype=responsibilities.dtype, device='cuda')
        #         responsibilities_I = responsibilities[i_start:i_end, :].unsqueeze(1).expand(i_batch_size, j_batch_size, gcount)
        #         responsibilities_J = responsibilities[j_start:j_end, :].unsqueeze(0).expand(i_batch_size, j_batch_size, gcount)
        #         # shape: (npI, npJ)
        #         klds_ij = (responsibilities_I * torch.log(responsibilities_I / responsibilities_J)).sum(dim=2)
        #         klds_ji = (responsibilities_J * torch.log(responsibilities_J / responsibilities_I)).sum(dim=2)
        #         symkldivs = 0.5*(klds_ij + klds_ji)
        #         w = torch.zeros(i_batch_size, j_batch_size, dtype=torch.bool, device='cuda')
        #         # shape: (npI, nn)
        #         subgraph_i = nngraph[i_start:i_end]
        #         # list of indizes
        #         subgraph_i_relevant = ((subgraph_i >= j_start) & (subgraph_i < j_end)).nonzero()
        #         w[subgraph_i_relevant[:, 0], subgraph_i[subgraph_i_relevant[:,0], subgraph_i_relevant[:,1]] - j_start] = 1
        #         subgraph_j = nngraph[j_start:j_end]
        #         subgraph_j_relevant = ((subgraph_j >= i_start) & (subgraph_j < i_end)).nonzero()
        #         w[subgraph_j_relevant[:, 0], subgraph_j[subgraph_j_relevant[:,0], subgraph_j_relevant[:,1]] - i_start] = 1
        #         R += 0.5*(symkldivs * w).sum().item()

        res = torch.zeros(1, batch_size, device=pcbatch.device, dtype=pcbatch.dtype)
        sfnn, sfnnl = self.calculate_scale_factors_nn(pcbatch)
        i = 0
        res[i, 0] = R
        return res

    def get_names(self) -> List[str]:
        nlst = []
        nlst.append("Smoothness (Unscaled)")
        return nlst

    def calculate_scale_factors_nn(self, pcbatch: torch.Tensor) -> (float, float):
        if not hasattr(pcbatch, "nnscalefactor"):
            md = pyeval.calc_rmsd_to_itself(pcbatch.view(-1, 3))[1]
            refdist = 128 / (2*math.sqrt(pcbatch.shape[1]) - 1)
            pcbatch.nnscalefactor = refdist / md
        return (math.pow(pcbatch.nnscalefactor, -3), -3 * math.log(pcbatch.nnscalefactor))
