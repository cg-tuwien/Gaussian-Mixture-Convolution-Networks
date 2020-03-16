import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import typing
import madam_imagetools

import gm
import mat_tools

import pointcloud

import config

from torch import Tensor

"""
pointclouds:    [m,n,3]-Tensor where n is the number of points
                and m the batch size. All pcs have to have the
                same point count. 
"""
def ad_algorithm(pointclouds: Tensor, n_components: int, n_iterations: int = 8, device: torch.device = 'cpu') -> Tensor:
    assert len(pointclouds.shape) == 3
    assert pointclouds.shape[2] == 3
    assert n_components > 0
    batch_size = pointclouds.size()[0]
    point_count = pointclouds.size()[1]

    target = pointclouds.to(device)

    #Find AABBs for each point cloud such that we can initialize the gm in the right area
    bbmin = torch.min(target, dim=1)[0]     #shape: (m, 3)
    bbmax = torch.max(target, dim=1)[0]     #shape: (m, 3)
    extends = bbmax - bbmin                 #shape: (m, 3)

    #Maybe: normalize point cloud to certain area

    # FROM NOW ON FOR NOW I ASSUME THAT batch_size == 1

    #Scale point clouds to [0,1] in the smallest dimension
    scale = torch.min(extends, dim=1)[0]    #shape: (m)
    scale = scale.view(batch_size, 1, 1)    #shape: (m,1,1)
    scale2 = scale ** 2
    target = target / scale

    #-- INITIALIZE GM(M) --
    mixture = gm.generate_random_mixtures(n_layers=batch_size, n_components=n_components, n_dims=3,
                                          pos_radius=0.5, cov_radius=5 / (n_components**(1/3)),
                                          weight_min=0, weight_max=1, device=device)
    #Achtung: Weights hier sind nicht Weights, sondern Amplituden.
    #Wir können also nur sicherstellen, dass die Summe der tatsächlichen Weights = 1
    #ist, indem wir sicherstellen, dass das ganze eine PDF ist (Amplituden > 0, Integral = 1)
    #und das machen wir während des Trainings durch den Penalty Term
    #Ist das ganze eine PDF, so müssen die Weights 1 sein, siehe Notizbuch S.164
    positions = gm.positions(mixture)
    positions += 0.5
    positions *= torch.tensor(extends, dtype=torch.float32, device=device)
    positions += bbmin

    fitting_start = time.time()

    weights = gm.weights(mixture)

    inversed_covariances = gm.covariances(mixture).inverse()
    (eigvals, eigvecs) = torch.symeig(inversed_covariances, eigenvectors=True)
    eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float32, device=device))
    icov_factor = torch.matmul(eigvecs, eigvals.sqrt().diag_embed())
    icov_factor.requires_grad = True

    optimiser = optim.Adam([weights, positions, icov_factor], lr=0.005)

    for k in range(n_iterations):
        print (f"{k}th iteration")
        if k == n_iterations / 2:
            optimiser = optim.Adam([weights, positions, icov_factor], lr=0.0005)
        optimiser.zero_grad()

        #TODO: Check epsilon for covariances
        #Evaluate quality by taking -sum(log(p(x)) + penalty
        #Indizes of sample points. Shape: (s), where s is #samples
        sample_point_idz = (torch.rand(config.eval_pc_n_sample_points, device=device, dtype=torch.float32) * point_count).long()
        sample_points = target[:, sample_point_idz, :]  #shape: (m,s,3)
        sample_points_in = sample_points.view(batch_size, 1, config.eval_pc_n_sample_points, 3) #shape: (m,1,s,3)
        inversed_covariances = icov_factor @ icov_factor.transpose(-2, -1) + torch.eye(3, 3, device=mixture.device) * 0.001 #eps
        assert not torch.isnan(inversed_covariances).any()
        assert not torch.isinf(inversed_covariances).any()
        mixture_with_inversed_cov = gm.pack_mixture(weights, positions, inversed_covariances)
        # shape first (m,1,s), then after view (m,s)
        output = gm.evaluate_inversed(mixture_with_inversed_cov, sample_points_in).view(batch_size, -1)
        loss = -torch.mean(torch.sum(torch.log(output), dim=1)) + abs(gm.integrate(mixture_with_inversed_cov) - 1)

        loss.backward()
        optimiser.step()

        if k % 100 == 0:
            print(f"iterations {k}: loss = {loss.item()}, min det = {torch.min(torch.det(inversed_covariances.detach()))}")
            _weights = weights.detach()
            _positions = positions.detach()
            _covariances = inversed_covariances.detach().inverse().transpose(-1, -2)
            _positions *= scale
            #Scaling of covariance by f@s@f', where f is the diagonal matrix of scalings
            #if all diag entries of f are the same, then this just results in times x^2, where x is the element of f
            _covariances *= scale2
            mixture = gm.pack_mixture(_weights, _positions, _covariances)
            #TODO: SAVE IN READABLE FORMAT
            gm.save(mixture, "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/pcgm-" + str(k).zfill(5) + ".gm")

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    weights = weights.detach()
    positions = positions.detach()
    covariances = inversed_covariances.detach().inverse().transpose(-1,-2)
    #scaling
    positions *= scale
    covariances *= scale2
    return gm.pack_mixture(weights, positions, covariances)

def test():
    pc = pointcloud.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/chair_0030.off")
    m1 = ad_algorithm(pc, n_components=20000, n_iterations=1000, device='cuda')
    #TODO: SAVE IN READABLE FORMAT
    gm.save(m1, "pcgm-final.gm")

test();