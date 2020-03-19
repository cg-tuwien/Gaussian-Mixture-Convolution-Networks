import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import torch.utils.tensorboard
import datetime
import typing
import madam_imagetools
import os

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
def ad_algorithm(pointclouds: Tensor, n_components: int, n_iterations: int = 8, device: torch.device = 'cpu', name: str = '') -> Tensor:
    assert len(pointclouds.shape) == 3
    assert pointclouds.shape[2] == 3
    assert n_components > 0

    save = True

    if name == '':
        name = f'fitPointcloudRegularized_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(
        config.data_base_path / 'tensorboard' / name)

    gm_path = config.data_base_path / 'models' / name
    os.mkdir(gm_path)

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
    positions = gm.positions(mixture) #shape: (m,1,n,3)
    positions += 0.5
    positions.requires_grad = True

    inversed_covariances = gm.covariances(mixture).inverse() #shape: (m,1,n,3,3)
    (eigvals, eigvecs) = torch.symeig(inversed_covariances, eigenvectors=True)
    eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float32, device=device))
    icov_factor = torch.matmul(eigvecs, eigvals.sqrt().diag_embed())
    icov_factor.requires_grad = True

    weights = gm.weights(mixture) #shape: (m,1,n)
    weights = weights / gm.integrate(mixture)
    weights.requires_grad = True

    fitting_start = time.time()

    optimiser = optim.Adam([weights, positions, icov_factor], lr=0.0001)

    for k in range(n_iterations):
        if k == n_iterations / 2:
            optimiser = optim.Adam([weights, positions, icov_factor], lr=0.00005)
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

        covariances = inversed_covariances.inverse()
        mixture_with_inversed_cov = gm.pack_mixture(weights, positions, inversed_covariances)
        mixture_with_regular_cov = gm.pack_mixture(weights, positions, covariances.detach().clone())
        # shape first (m,1,s), then after view (m,s)
        output = gm.evaluate_inversed(mixture_with_inversed_cov, sample_points_in).view(batch_size, -1)
        loss1 = -torch.mean(torch.log(output), dim=1)
        loss2 = torch.abs(gm.integrate(mixture_with_regular_cov) - 1)
        #loss2 = point_count * torch.exp(2 + torch.abs(gm.integrate(mixture_with_regular_cov) - 1))
        loss = loss1 + loss2

        loss.backward()
        optimiser.step()

        tensor_board_writer.add_scalar("0. training loss", loss.item(), k)
        tensor_board_writer.add_scalar("1. likelihood loss", loss1.item(), k)
        tensor_board_writer.add_scalar("2. integral loss", loss2.item(), k)

        print(f"iterations {k}: loss = {loss.item()}, loss1={loss1.item()}, loss2={loss2.item()}")
        if save and k % 10 == 0:
            _positions = positions.detach().clone()
            _positions -= 0.5
            _positions *= scale

            _weights = weights.detach().clone()
            _weights *= covariances.det().sqrt() * 15.74960995

            _covariances = inversed_covariances.detach().inverse().transpose(-1, -2).clone()
            # #Scaling of covariance by f@s@f', where f is the diagonal matrix of scalings
            # #if all diag entries of f are the same, then this just results in times x^2, where x is the element of f
            _covariances *= scale2

            #_weights /= covariances.det().sqrt()

            gm.write_gm_to_ply(_weights, _positions, _covariances, 0,
                               f"{gm_path}/pcgmm-" + str(k).zfill(5) + ".ply")

            # _mixture = gm.pack_mixture(_weights, _positions, _covariances)
            # gm.save(_mixture, "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/pcgm-" + str(k).zfill(5) + ".gm")

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    positions = positions.detach()
    covariances = inversed_covariances.detach().inverse().transpose(-1,-2)
    #scaling
    positions -= 0.5
    positions *= scale
    weights *= covariances.det().sqrt()
    covariances *= scale2
    if save:
        gm.write_gm_to_ply(weights * 15.74960995, positions, covariances, 0,
                           f"{gm_path}/pcgmm-final.ply")
    weights /= covariances.det().sqrt()
    return gm.pack_mixture(weights, positions, covariances)

def test():
    pc = pointcloud.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/chair_0030.off")
    name = input('Name for this training (or empty for auto): ')
    m1 = ad_algorithm(pc, n_components=20000, n_iterations=1000, device='cuda', name=name)
    #pc = pointcloud.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/testdata.off")
    #m1 = ad_algorithm(pc, n_components=1, n_iterations=1000, device='cuda')
    #gm.save(m1, "pcgm-final.gm")

test();