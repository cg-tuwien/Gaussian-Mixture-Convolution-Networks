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
import madam_imagetools

import gm
import mat_tools

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


def ad_algorithm(pointclouds: Tensor,
                 n_components: int,
                 n_iterations: int,
                 device: torch.device = 'cpu',
                 name: str = '',
                 mixture: Tensor = None,
                 start_epoch: int = 0,
                 init_positions_from_pointcloud: bool = False,
                 init_weights_equal: bool = False,
                 penalize_long_gaussians: bool = False,
                 penalize_amplitude_differences: bool = False,
                 learn_rate: float = 0.0001) -> Tensor:

    assert len(pointclouds.shape) == 3
    assert pointclouds.shape[2] == 3
    assert n_components > 0

    # Create Visualizer
    vis3d = pygmvis.create_visualizer(async=False, width=500, height=500)
    vis3d.set_camera_auto(True)
    vis3d.set_pointclouds(pointclouds)
    vis3d.set_density_rendering(True, pygmvis.GMDensityRenderMode.ADDITIVE_ACC_PROJECTED)

    batch_size = pointclouds.size()[0]
    point_count = pointclouds.size()[1]

    target = pointclouds.to(device)

    # Find AABBs for each point cloud such that we can initialize the gm in the right area
    bbmin = torch.min(target, dim=1)[0]     #shape: (m, 3)
    bbmax = torch.max(target, dim=1)[0]     #shape: (m, 3)
    extends = bbmax - bbmin                 #shape: (m, 3)

    # Scale point clouds to [0,1] in the smallest dimension
    scale = torch.min(extends, dim=1)[0]    #shape: (m)
    scale = scale.view(batch_size, 1, 1)    #shape: (m,1,1)
    scale2 = scale ** 2
    target = target / scale
    target += 0.5
    scale = scale.view(batch_size, 1, 1, 1)  # shape: (m,1,1,1)
    scale2 = scale2.view(batch_size, 1, 1, 1, 1)  # shape: (m,1,1,1,1)

    # Initialize Gaussian Mixture
    create_new_mixture = (mixture is None)
    if create_new_mixture:
        mixture = gm.generate_random_mixtures(n_batch=batch_size, n_layers=1, n_components=n_components, n_dims=3,
                                              pos_radius=0.5, cov_radius=0.01 / (n_components**(1/3)),
                                              weight_min=0, weight_max=1, device=device)
        if init_positions_from_pointcloud:
            indizes = torch.randperm(point_count)[0:n_components]
            positions = target[:, indizes, :].view(batch_size, 1, n_components, 3) - 0.5
            mixture = gm.pack_mixture(gm.weights(mixture), positions, gm.covariances(mixture))

        if init_weights_equal:
            weights = torch.ones((batch_size, 1, n_components))
            mixture = gm.pack_mixture(weights, gm.positions(mixture), gm.covariances(mixture))
    else:
        mixture = mixture.to(device)

    positions = gm.positions(mixture) # shape: (m,1,n,3)
    if not create_new_mixture:
        positions /= scale  # Positions need to be downscaled
    positions += 0.5
    positions.requires_grad = True

    covariances = gm.covariances(mixture)

    pi_relative = gm.weights(mixture)  # shape: (m,1,n)
    if not create_new_mixture:
        pi_relative *= covariances.detach().det().sqrt() * 15.74960995  # calculate priors from amplitudes
    pi_relative.requires_grad = True

    if not create_new_mixture:
        covariances /= scale2   # Covariances need to be downscaled
    inversed_covariances = covariances.inverse() # shape: (m,1,n,3,3)
    (eigvals, eigvecs) = torch.symeig(inversed_covariances, eigenvectors=True)
    eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float32, device=device))
    icov_factor = torch.matmul(eigvecs, eigvals.sqrt().diag_embed())
    icov_factor.requires_grad = True


    # Prepare directories
    if name == '':
        name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(
        config.data_base_path / 'tensorboard' / name)

    gm_path = config.data_base_path / 'models' / name
    os.mkdir(gm_path)

    fitting_start = time.time()
    print(datetime.datetime.now().strftime("%H:%M:%S"))

    optimiser = optim.Adam([pi_relative, positions, icov_factor], lr=learn_rate)

    for k in range(start_epoch, n_iterations):
        optimiser.zero_grad()

        #Indizes of sample points. Shape: (s), where s is #samples
        sample_point_idz = torch.randperm(point_count)[0:config.eval_pc_n_sample_points]
        sample_points = target[:, sample_point_idz, :]  #shape: (m,s,3)
        sample_points_in = sample_points.view(batch_size, 1, config.eval_pc_n_sample_points, 3) #shape: (m,1,s,3)
        inversed_covariances = icov_factor @ icov_factor.transpose(-2, -1) + torch.eye(3, 3, device=mixture.device) * 0.001 #eps
        assert not torch.isnan(inversed_covariances).any()
        assert not torch.isinf(inversed_covariances).any()

        # Calculate amplitudes from priors
        pi_sum = pi_relative.abs().sum(dim=2).view(batch_size, 1, 1)  # shape: (m,1) -> (m,1,1)
        pi_normalized = pi_relative.abs() / pi_sum  # shape (m,1,n)
        covariances = inversed_covariances.inverse()
        amplitudes = pi_normalized / (covariances.det().sqrt() * 15.74960995)

        # Main Loss
        mixture_with_inversed_cov = gm.pack_mixture(amplitudes, positions, inversed_covariances)
        # shape first (m,1,s), then after view (m,s)
        output = gm.evaluate_inversed(mixture_with_inversed_cov, sample_points_in).view(batch_size, -1)
        loss1 = -torch.mean(torch.log(output + 0.001), dim=1)

        cov_cost = torch.tensor(0)
        if penalize_long_gaussians:
            eigenvalues: Tensor = torch.symeig(covariances, eigenvectors=True).eigenvalues
            largest_eigenvalue: Tensor = eigenvalues[:, :, :, -1]
            smallest_eigenvalue: Tensor = eigenvalues[:, :, :, 0]
            #cov_cost: Tensor = (0.1 * torch.log(1 + torch.exp(largest_eigenvalue / smallest_eigenvalue - 10))).sum() (leads to nan)
            cov_cost: Tensor = 0.1 * largest_eigenvalue / smallest_eigenvalue - 1
            cov_cost = cov_cost.where(cov_cost > torch.zeros_like(cov_cost), torch.zeros_like(cov_cost)).sum()

        amp_cost = torch.tensor(0)
        if penalize_amplitude_differences:
            amp_cost = amplitudes.max() / amplitudes.min()
            amp_cost = torch.max(amp_cost - 100, 0)[0]

        #mixture_with_regular_cov = gm.pack_mixture(amplitudes, positions, covariances.detach().clone())
        #integ = gm.integrate(mixture_with_regular_cov)
        #print(f"This should be one: {integ}")

        loss = loss1 + cov_cost + amp_cost

        loss.backward()
        optimiser.step()

        tensor_board_writer.add_scalar("0. training loss", loss.item(), k)
        tensor_board_writer.add_scalar("1. likelihood loss", loss1.item(), k)
        if penalize_long_gaussians:
            tensor_board_writer.add_scalar("2. length loss", cov_cost.item(), k)
        if penalize_amplitude_differences:
            tensor_board_writer.add_scalar("3. ampl loss", amp_cost.item(), k)

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
            vis3d.set_gaussian_mixtures(_mixture.detach().cpu(), isgmm=False)
            res = vis3d.render(k)
            for i in range(res.shape[0]):
                tensor_board_writer.add_image(f"GM {i}, Ellipsoids", res[i, 0, :, :, :], k, dataformats="HWC")
                tensor_board_writer.add_image(f"GM {i}, Density", res[i, 1, :, :, :], k, dataformats="HWC")
                gm.write_gm_to_ply(_amplitudes, _positions, _covariances, i, f"{gm_path}/pcgm-{i}.ply")
            gm.save(_mixture, f"{gm_path}/pcgm-.gm")

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
    for i in range(batch_size):
        gm.write_gm_to_ply(amplitudes, positions, covariances, i, f"{gm_path}/pcgm-{i}.ply")
    gm.save(_mixture, f"{gm_path}/pcgm.gm")
    return _mixture

def test():
    torch.manual_seed(0)
    np.random.seed(0)

    pcs = pointcloud.load_pc_from_off(
        "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-lores/ModelNet10/chair/train/chair_0030.off")
    # gms = gm.load(
    #     "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/fitPointcloud_2020-04-09_18-56-05/pcgmm-22200.gm")[
    #     0]
    # gms = gms[0, :, :, :].view(1, 1, 100, 13)  # [m,1,n_components,13]
    name = input('Name for this training (or empty for auto): ')
    ad_algorithm(
        pointclouds=pcs,
        n_components=100,
        n_iterations=1000000,
        device='cuda',
        name=name,
        # mixture=gms,
        # start_epoch=22201,
        init_positions_from_pointcloud=False,
        init_weights_equal=False,
        penalize_long_gaussians=False,
        penalize_amplitude_differences=False,
        learn_rate=0.0001
    )

test()