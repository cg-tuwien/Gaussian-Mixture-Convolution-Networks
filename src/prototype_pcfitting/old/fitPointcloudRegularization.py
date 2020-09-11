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
import os


import core.gm as gm
import core.mat_tools as mat_tools
import core.madam_imagetools as madam_imagetools

import pointcloud

import config

from torch import Tensor

import pygmvis

"""
pointclouds:    [m,n,3]-Tensor where n is the number of points
                and m the batch size. All pcs have to have the
                same point count. 
mixture:        [m,1,n_components,13]-Tensor to initialize the 
                Mixture with. Or None, if random initialization
                is preferred. Useful if previous training wants
                to be continued. 
"""


def ad_algorithm(pointclouds: Tensor, n_components: int, mixture: Tensor = None, n_iterations: int = 8, device: torch.device = 'cpu', name: str = '') -> Tensor:
    assert len(pointclouds.shape) == 3
    assert pointclouds.shape[2] == 3
    assert n_components > 0

    if name == '':
        name = f'fitPointcloudRegularized_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(
        config.data_base_path / 'tensorboard' / name)

    vis3d = pygmvis.create_visualizer(async=False, width=500, height=500)
    vis3d.set_camera_auto(True)
    vis3d.set_pointclouds(pointclouds)
    vis3d.set_density_rendering(True, pygmvis.GMDensityRenderMode.ADDITIVE_ACC_PROJECTED)

    gm_path = config.data_base_path / 'models' / name
    os.mkdir(gm_path)

    batch_size = pointclouds.size()[0]
    point_count = pointclouds.size()[1]

    target = pointclouds.to(device)

    # Find AABBs for each point cloud such that we can initialize the gm in the right area
    bbmin = torch.min(target, dim=1)[0]     #shape: (m, 3)
    bbmax = torch.max(target, dim=1)[0]     #shape: (m, 3)
    extends = bbmax - bbmin                 #shape: (m, 3)

    # Scale point clouds to [0,1] in the smallest dimension
    scale = torch.min(extends, dim=1)[0]    # shape: (m)
    scale = scale.view(batch_size, 1, 1)    # shape: (m,1,1)
    scale2 = scale ** 2
    target = target / scale
    target += 0.5
    scale = scale.view(batch_size, 1, 1, 1)  # shape: (m,1,1,1)
    scale2 = scale2.view(batch_size, 1, 1, 1, 1)  # shape: (m,1,1,1,1)

    # -- INITIALIZE GM --
    newmixture = (mixture is None)
    if newmixture:
        mixture = gm.generate_random_mixtures(n_batch=batch_size, n_layers=1, n_components=n_components, n_dims=3,
                                          pos_radius=0.5, cov_radius=0.01 / (n_components**(1/3)),
                                          weight_min=0, weight_max=1, device=device)
    else:
        mixture = mixture.to(device)

    positions = gm.positions(mixture) #shape: (m,1,n,3)
    if not newmixture:
        positions /= scale
    positions += 0.5
    positions.requires_grad = True

    covariances = gm.covariances(mixture)

    weights = gm.weights(mixture) #shape: (m,1,n)
    if not newmixture:
        weights *= covariances.detach().det().sqrt()

    if not newmixture:
        covariances /= scale2
    inversed_covariances = covariances.inverse() #shape: (m,1,n,3,3)
    (eigvals, eigvecs) = torch.symeig(inversed_covariances, eigenvectors=True)
    eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float32, device=device))
    icov_factor = torch.matmul(eigvecs, eigvals.sqrt().diag_embed())
    icov_factor.requires_grad = True

    if not newmixture:
        weights /= covariances.detach().det().sqrt()
    else:
        weights = weights / gm.integrate(mixture).view(batch_size,1,1)
    weights.requires_grad = True

    fitting_start = time.time()

    optimiser = optim.Adam([weights, positions, icov_factor], lr=0.0001)

    for k in range(n_iterations):
        if k == n_iterations / 2:
            optimiser = optim.Adam([weights, positions, icov_factor], lr=0.00005)
        optimiser.zero_grad()

        # Indizes of sample points. Shape: (s), where s is #samples
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
        loss1 = -torch.mean(torch.log(output + 0.001), dim=1).sum() #0.001 to avoid infinite loss
        loss2 = torch.abs(gm.integrate(mixture_with_regular_cov) - 1).view(batch_size).sum()
        loss = loss1 + loss2    # sum of gmm-weights are 1 <=> integral is 1 (s.164)

        loss.backward()
        optimiser.step()

        tensor_board_writer.add_scalar("0. training loss", loss.item(), k)
        tensor_board_writer.add_scalar("1. likelihood loss", loss1.item(), k)
        tensor_board_writer.add_scalar("2. integral loss", loss2.item(), k)

        print(f"iterations {k}: loss = {loss.item()}, loss1={loss1.item()}, loss2={loss2.item()}")
        if k % 100 == 0:
            _positions = positions.detach().clone()
            _positions -= 0.5
            _positions *= scale

            _weights = weights.detach().clone()
            _weights *= covariances.detach().det().sqrt()   # Correct scaling requires probabilities to stay the same

            _covariances = inversed_covariances.detach().inverse().transpose(-1, -2).clone()
            # Scaling of covariance by f@s@f', where f is the diagonal matrix of scalings
            # if all diag entries of f are the same, then this just results in times x^2, where x is the element of f
            _covariances *= scale2

            _weights /= _covariances.det().sqrt()

            _mixture = gm.pack_mixture(_weights, _positions, _covariances)
            vis3d.set_gaussian_mixtures(_mixture.detach().cpu(), isgmm=False)
            res = vis3d.render(k)
            for i in range(res.shape[0]):
                tensor_board_writer.add_image(f"GM {i}, Ellipsoids", res[i, 0, :, :, :], k, dataformats="HWC")
                tensor_board_writer.add_image(f"GM {i}, Density", res[i, 1, :, :, :], k, dataformats="HWC")
            gm.write_gm_to_ply(_weights, _positions, _covariances, i, f"{gm_path}/pcgmm-" + str(k).zfill(5) + ".ply")
            gm.save(_mixture, f"{gm_path}/pcgmm-" + str(k).zfill(5) + ".gm")

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    positions = positions.detach()
    covariances = inversed_covariances.detach().inverse().transpose(-1,-2)
    #scaling
    positions -= 0.5
    positions *= scale
    weights = weights.detach()
    weights *= covariances.det().sqrt()
    covariances *= scale2
    # write gmM: gm.write_gm_to_ply(weights * 15.74960995, positions, covariances, 0,
    #                   f"{gm_path}/pcgmm-final.ply") # GMM-Weights!
    weights /= covariances.det().sqrt()
    _mixture = gm.pack_mixture(weights, positions, covariances)
    for i in range(batch_size):
        gm.write_gm_to_ply(weights, positions, covariances, i, f"{gm_path}/pcgmm-{i}-final.ply")
    gm.save(_mixture, f"{gm_path}/pcgmm-final.gm")
    return _mixture

def test():
    pcs = pointcloud.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/chair_0030.off")
    # pcs = pointcloud.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-lores/ModelNet10/chair/train/chair_0030.off")
    # gms = gm.load(
    #    "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/2020-04-08-30-42-train-cont/pcgmm-36100.gm")[0]
    # gms = gms[0,:,:,:].view(1,1,2000,13)#[m,1,n_components,13]
    gms = gm.load("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/2020-04-09-30LRv2-translfixed/pcgmm-75600.gm")[0]
    gms = gms[0, :, :, :].view(1, 1, 100, 13)  # [m,1,n_components,13]
    #pcs[0,:,:] = pc1
    #pcs[1,:,:] = pc2
    name = input('Name for this training (or empty for auto): ')
    m1 = ad_algorithm(pcs, n_components=100, n_iterations=1000000, device='cuda', name=name)

test();