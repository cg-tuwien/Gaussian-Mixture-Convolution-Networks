import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import torch.utils.tensorboard
import datetime
import typing


import core.gm as gm
import core.mat_tools as mat_tools
import core.madam_imagetools as madam_imagetools

import pointcloud

import config

from torch import Tensor

# THIS ONLY SUPPORTS BATCH SIZE = 1

def debug_render(pointclouds: Tensor, mixture: Tensor, clamp: typing.Tuple[float, float] = (0, 1)):
    images = gm.render(mixture, x_low=0, x_high=1, y_low=0, y_high=1, width=500, height=500)
    images = madam_imagetools.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
    pointcolor = np.array([1,0,0,0])
    pcs = (499*pointclouds).round().int()

    images[pcs[0, :, 1], pcs[0, :, 0], :] = 0.2*images[pcs[0, :, 1], pcs[0, :, 0], :] + 0.8*pointcolor

    return images[:, :, :3]

"""
pointclouds:    [m,n,2]-Tensor where n is the number of points
                and m the batch size. All pcs have to have the
                same point count. 
mixture:        [m,1,n_components,x]-Tensor to initialize the 
                Mixture with. Or None, if random initialization
                is preferred. Useful if previous training wants
                to be continued. 
"""


def ad_algorithm(pointclouds: Tensor, n_components: int, mixture: Tensor = None, n_iterations: int = 8, device: torch.device = 'cpu', name: str = '', startepoch=0) -> Tensor:
    assert len(pointclouds.shape) == 3
    assert pointclouds.shape[2] == 2
    assert n_components > 0

    if name == '':
        name = f'fitPointcloud2D_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(
        config.data_base_path / 'tensorboard' / name)

    gm_path = config.data_base_path / 'models' / name
    os.mkdir(gm_path)

    batch_size = pointclouds.size()[0]
    point_count = pointclouds.size()[1]

    target = pointclouds.to(device)

    #Find AABBs for each point cloud such that we can initialize the gm in the right area
    bbmin = torch.min(target, dim=1)[0]     #shape: (m, 2)
    bbmax = torch.max(target, dim=1)[0]     #shape: (m, 2)
    extends = bbmax - bbmin                 #shape: (m, 2)

    #Scale point clouds to [0,1] in the smallest dimension
    # HERE WE USE THE MAXIMUM FOR SCALING SO IT ALL FITS INTO PICTURE
    scale = torch.max(extends, dim=1)[0]    #shape: (m)
    scale = scale.view(batch_size, 1, 1)    #shape: (m,1,1)
    scale2 = scale ** 2
    target = target / scale
    target += 0.5
    scale = scale.view(batch_size, 1, 1, 1)  # shape: (m,1,1,1)
    scale2 = scale2.view(batch_size, 1, 1, 1, 1)  # shape: (m,1,1,1,1)

    #-- INITIALIZE GM(M) --
    newmixture = (mixture is None)
    if newmixture:
        mixture = gm.generate_random_mixtures(n_batch=batch_size, n_layers=1, n_components=n_components, n_dims=2,
                                          pos_radius=0.5, cov_radius=0.01 / (n_components**(1/2)),
                                          weight_min=0, weight_max=1, device=device)
        indizes = torch.randperm(point_count)[0:n_components]
        positions = target[:, indizes, :].view(batch_size, 1, n_components, 2)
        mixture = gm.pack_mixture(gm.weights(mixture), positions, gm.covariances(mixture))
    else:
        mixture = mixture.to(device)

    positions = gm.positions(mixture) #shape: (m,1,n,2)
    if not newmixture:
        positions /= scale
        positions += 0.5
    positions.requires_grad = True

    covariances = gm.covariances(mixture)

    pi_relative = gm.weights(mixture)  # shape: (m,1,n)
    if not newmixture:
        pi_relative *= covariances.detach().det().sqrt() * 15.74960995
    pi_relative.requires_grad = True

    if not newmixture:
        covariances /= scale2
    inversed_covariances = covariances.inverse() #shape: (m,1,n,2,2)
    (eigvals, eigvecs) = torch.symeig(inversed_covariances, eigenvectors=True)
    eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float32, device=device))
    icov_factor = torch.matmul(eigvecs, eigvals.sqrt().diag_embed())
    icov_factor.requires_grad = True

    fitting_start = time.time()

    optimiser = optim.Adam([pi_relative, positions, icov_factor], lr=0.0001)

    for k in range(startepoch, n_iterations):
        if k == n_iterations / 2:
            optimiser = optim.Adam([pi_relative, positions, icov_factor], lr=0.00005)
        optimiser.zero_grad()

        #Indizes of sample points. Shape: (s), where s is #samples
        sample_point_idz = (torch.rand(config.eval_pc_n_sample_points, device=device, dtype=torch.float32) * point_count).long()
        sample_points = target[:, sample_point_idz, :]  #shape: (m,s,2)
        sample_points_in = sample_points.view(batch_size, 1, config.eval_pc_n_sample_points, 2) #shape: (m,1,s,2)

        #Sample points for empty area
        empty_samples = torch.rand((batch_size, 1, config.eval_pc_n_sample_points,2), device=device, dtype=torch.float32)

        inversed_covariances = icov_factor @ icov_factor.transpose(-2, -1) + torch.eye(2, 2, device=mixture.device) * 0.001 #eps
        assert not torch.isnan(inversed_covariances).any()
        assert not torch.isinf(inversed_covariances).any()

        pi_sum = pi_relative.abs().sum(dim=2).view(batch_size, 1, 1)  # shape: (m,1) -> (m,1,1)
        pi_normalized = pi_relative.abs() / pi_sum  # shape (m,1,n)
        covariances = inversed_covariances.inverse()
        amplitudes = pi_normalized / (covariances.det().sqrt() * 15.74960995)

        mixture_with_inversed_cov = gm.pack_mixture(amplitudes, positions, inversed_covariances)
        # shape first (m,1,s), then after view (m,s)
        output = gm.evaluate_inversed(mixture_with_inversed_cov, sample_points_in).view(batch_size, -1)
        output_empty = gm.evaluate_inversed(mixture_with_inversed_cov, empty_samples).view(batch_size, -1)
        loss1 = -torch.mean(torch.log(output + 0.001), dim=1)
        loss2 = torch.mean(torch.log(output_empty + 0.001), dim=1)
        loss = loss1  # + loss2

        if torch.isnan(loss).any():
            print("NAN detected!")

        #mixture_with_regular_cov = gm.pack_mixture(amplitudes, positions, covariances.detach().clone())
        #integ = gm.integrate(mixture_with_regular_cov)
        #print(f"This should be one: {integ}")

        loss.backward()
        optimiser.step()

        tensor_board_writer.add_scalar("0. training loss", loss.item(), k)
        tensor_board_writer.add_scalar("1. likelihood loss", loss1.item(), k)
        tensor_board_writer.add_scalar("2. unlikelihood loss", loss2.item(), k)

        print(f"iterations {k}: loss = {loss.item()}")
        if k % 100 == 0:
            _positions = positions.detach().clone()
            _positions -= 0.5
            _positions *= scale

            _covariances = inversed_covariances.detach().inverse().transpose(-1, -2).clone()
            #Scaling of covariance by f@s@f', where f is the diagonal matrix of scalings
            #if all diag entries of f are the same, then this just results in times x^2, where x is the element of f
            _covariances *= scale2

            _amplitudes = pi_normalized / (_covariances.det().sqrt() * 15.74960995)

            _mixture = gm.pack_mixture(_amplitudes, _positions, _covariances)
            rendering = debug_render(target.cpu(), mixture)
            tensor_board_writer.add_image(f"2DGM", rendering, k, dataformats="HWC")
            # gm.save(_mixture, f"{gm_path}/pcgm-" + str(k).zfill(5) + ".gm")

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    positions = positions.detach()
    covariances = inversed_covariances.detach().inverse().transpose(-1,-2)
    #scaling
    positions -= 0.5
    positions *= scale
    covariances *= scale2
    pi_sum = pi_relative.abs().sum(dim=2).view(batch_size, 1, 1)  # shape: (m,1) -> (m,1,1)
    pi_normalized = pi_relative.abs() / pi_sum  # shape (m,1,n)
    amplitudes = pi_normalized / covariances.det()
    _mixture = gm.pack_mixture(amplitudes, positions, covariances)
    # for i in range(batch_size):
    #     gm.write_gm_to_ply(amplitudes, positions, covariances, i, f"{gm_path}/pcgmm-{i}-final.ply")
    # gm.save(_mixture, f"{gm_path}/pcgmm-final.gm")
    return _mixture

def test():
    pcs = pointcloud.load_pc_from_off(
        "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-lores/ModelNet10/chair/train/chair_0030.off")
    pcs = pcs[:, :, [0,2]].view(1, -1, 2)
    name = input('Name for this training (or empty for auto): ')
    m1 = ad_algorithm(pcs, n_components=100, n_iterations=1000000, device='cuda', name=name)

test()
