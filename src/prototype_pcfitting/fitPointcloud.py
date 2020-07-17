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
#import madam_imagetools
import gc
import struct

import gmc.mixture as gm

import pointcloud

import config

import gmc.cpp.gm_vis.pygmvis as pygmvis

from torch import Tensor



"""
pointclouds:    [m,n,3]-Tensor where n is the number of points
                and m the batch size. All pcs have to have the
                same point count. 
mixture:        [m,1,n_components,13]-Tensor to initialize the 
                Mixture with. Or None, if random initialization
                is preferred. Useful if previous training wants
                to be continued. 
validation_pc:  Pointclouds used for validating the result.  
                [m,k,3]-Tensor where k is the number of points
                and m the batch size.
"""




def ad_algorithm(pointclouds: Tensor,
                 n_components: int,
                 n_iterations: int,
                 device: torch.device = 'cpu',
                 name: str = '',
                 mixture: Tensor = None,
                 start_epoch: int = 0,
                 validation_pc: Tensor = None,
                 init_positions_from_pointcloud: bool = False,
                 init_weights_equal: bool = False,
                 penalize_long_gaussians: bool = False,
                 penalize_amplitude_differences: bool = False,
                 penalize_extends_differences: bool = False,
                 penalize_small_determinants: bool = False,
                 weight_softmax: bool = False,
                 constant_weights: bool = False,
                 log_positions: bool = False,
                 cov_train_mode: str = 'cholesky',   #cholesky or eigen
                 learn_rate_pos: float = 0.001,
                 learn_rate_cov: float = 0.02,
                 learn_rate_wei: float = 0.0005) -> Tensor:

    assert len(pointclouds.shape) == 3
    assert pointclouds.shape[2] == 3
    assert n_components > 0

    if cov_train_mode == 'cholesky':
        epsilon = 1e-7
    else:
        epsilon = 0.000000001

    # Create Visualizer
    vis3d = pygmvis.create_visualizer(False, width=500, height=500)
    vis3d.set_camera_auto(True)
    vis3d.set_pointclouds(pointclouds)
    vis3d.set_density_rendering(True, pygmvis.GMDensityRenderMode.ADDITIVE_ACC_PROJECTED)
    vis3d.set_ellipsoid_coloring(pygmvis.GMColoringRenderMode.COLOR_WEIGHT, pygmvis.GMColorRangeMode.RANGE_MINMAX)
    vis3d.set_positions_rendering(True, True)
    vis3d.set_positions_coloring(pygmvis.GMColoringRenderMode.COLOR_WEIGHT, pygmvis.GMColorRangeMode.RANGE_MEDMED)

    batch_size = pointclouds.shape[0]
    point_count = pointclouds.shape[1]

    target = pointclouds.to(device)

    # --SCALING------------------------------------------------------------

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

    # Scale Validation Pointcloud if it exists
    if validation_pc is not None:
        validation_pc = validation_pc.view(batch_size, 1, -1, 3).to(device)
        validation_pc = validation_pc / scale
        validation_pc += 0.5

    # --GM-INITIALIZATION--------------------------------------------------

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
            weights = torch.ones((batch_size, 1, n_components)).to(device)
            mixture = gm.pack_mixture(weights, gm.positions(mixture), gm.covariances(mixture))
    else:
        mixture = mixture.to(device)

    positions = gm.positions(mixture) # shape: (m,1,n,3)
    if not create_new_mixture:
        positions /= scale  # Positions need to be downscaled
    positions += 0.5
    positions.requires_grad = True

    covariances = gm.covariances(mixture)

    if constant_weights:
        pi_relative = torch.ones((batch_size, 1, n_components))
        pi_normalized = torch.nn.functional.softmax(pi_relative,dim=2).to(device)
    else:
        pi_relative = gm.weights(mixture)  # shape: (m,1,n)
        if not create_new_mixture:
            pi_relative *= covariances.detach().det().sqrt() * 15.74960995  # calculate priors from amplitudes
        pi_relative.requires_grad = True
        if weight_softmax:
            pi_normalized = torch.nn.functional.softmax(pi_relative, dim=2)
        else:
            pi_sum = pi_relative.abs().sum(dim=2).view(batch_size, 1, 1)  # shape: (m,1) -> (m,1,1)
            pi_normalized = pi_relative.abs() / pi_sum  # shape (m,1,n)

    if not create_new_mixture:
        covariances /= scale2   # Covariances need to be downscaled

    if cov_train_mode == 'cholesky':
        cov_factor_mat = torch.cholesky(covariances)
        cov_factor_vec = torch.zeros((batch_size, 1, n_components, 6)).to(device)
        cov_factor_vec[:, :, :, 0] = torch.max(cov_factor_mat[:, :, :, 0, 0] - epsilon, 0)[0]
        cov_factor_vec[:, :, :, 1] = torch.max(cov_factor_mat[:, :, :, 1, 1] - epsilon, 0)[0]
        cov_factor_vec[:, :, :, 2] = torch.max(cov_factor_mat[:, :, :, 2, 2] - epsilon, 0)[0]
        cov_factor_vec[:, :, :, 3] = cov_factor_mat[:, :, :, 1, 0]
        cov_factor_vec[:, :, :, 4] = cov_factor_mat[:, :, :, 2, 0]
        cov_factor_vec[:, :, :, 5] = cov_factor_mat[:, :, :, 2, 1]
        cov_train_data = cov_factor_vec
        cov_train_data.requires_grad = True
    elif cov_train_mode == 'eigen':
        inversed_covariances = covariances.inverse() # shape: (m,1,n,3,3)
        (eigvals, eigvecs) = torch.symeig(inversed_covariances - torch.eye(3, 3, device=mixture.device) * epsilon, eigenvectors=True)
        eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float32, device=device))
        icov_factor = torch.matmul(eigvecs, eigvals.sqrt().diag_embed())
        cov_train_data = icov_factor
        cov_train_data.requires_grad = True
    else:
        print("Invalid cov train mode")
        return
    covariances, inversed_covariances, determinants = calculate_covariance_matrices(cov_train_mode, cov_train_data, epsilon)

    amplitudes = pi_normalized / (determinants.sqrt() * 15.74960995)


    # --OPTIMIZER-INITIALIZATION----------------------------------------

    #optimiser_pos = optim.RMSprop([positions], lr=learn_rate_pos, alpha=0.99, momentum=0.0)
    optimiser_pos = optim.RMSprop([positions], lr=learn_rate_pos, alpha=0.7, momentum=0.0)
    #optimiser_pos = optim.Adam([positions], lr=learn_rate_pos)
    #optimiser_pos = optim.SGD([positions], lr=learn_rate_pos)
    #LRadap = lambda epoch: 1 / (1 + 0.0001 * epoch)
    #scheduler_pos = optim.lr_scheduler.LambdaLR(optimiser_pos, LRadap)
    if cov_train_mode == 'cholesky':
        optimiser_cov = optim.Adam([cov_factor_vec], lr=learn_rate_cov)
    else:
        optimiser_cov = optim.Adam([icov_factor], lr=learn_rate_cov)

    if constant_weights:
        optimiser_pi = None
    else:
        optimiser_pi = optim.Adam([pi_relative], lr=learn_rate_wei)
        #optimiser_pi = optim.SGD([pi_relative], lr=learn_rate_wei)


    # --DIRECTORY-PREPERATION-------------------------------------------

    if name == '':
        name = f'fitPointcloud_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(
        config.data_base_path / 'tensorboard' / name)

    gm_path = config.data_base_path / 'models' / name
    os.mkdir(gm_path)

    pos_log_size = 500

    if log_positions:
        # Create Log Files
        pos_log = torch.zeros((batch_size,n_components,pos_log_size,3))
        for b in range(batch_size):
            for g in range(n_components):
                f = open(f"{gm_path}/pos-b{b}-g{g}.bin", "w+")
                f.close()

    # Log initial gm and positions
    if cov_train_mode == 'cholesky':
        _amplitudes, _positions, _covariances = rescale_gmm_to_gm(pi_normalized, positions, covariances, determinants, scale, scale2)
    else:
        _amplitudes, _positions, _covariances = rescale_igmm_to_gm(pi_normalized, positions, inversed_covariances, determinants,
                                                               scale, scale2)
    gm.write_gm_to_ply(_amplitudes, _positions, _covariances, 0, f"{gm_path}/pcgmm-{0}-initial.ply")
    if log_positions:
        _positions = positions.detach().clone()
        _positions -= 0.5
        _positions *= scale
        pos_log[:, :, 0, :] = _positions.view(batch_size, n_components, 3)

    # --START FITTING---------------------------------------------------

    fitting_start = time.time()
    print(datetime.datetime.now().strftime("%H:%M:%S"))

    for k in range(start_epoch, n_iterations):
        optimiser_pos.zero_grad()
        optimiser_cov.zero_grad()
        if optimiser_pi:
            optimiser_pi.zero_grad()

        # Get Sample Points for Loss Evaluation
        sample_point_idz = torch.randperm(point_count)[0:config.eval_pc_n_sample_points] #Shape: (s), where s is #samples
        sample_points = target[:, sample_point_idz, :]  #shape: (m,s,3)
        sample_points_in = sample_points.view(batch_size, 1, min(point_count, config.eval_pc_n_sample_points), 3) #shape: (m,1,s,3)

        # print("icov sym?", (inversed_covariances.transpose(-2,-1) == inversed_covariances).all())
        # print("cov sym?", (covariances.transpose(-2,-1) == covariances).all())
        # print("cov pd?", (covariances.det() > 0).all())

        # Calculate main loss
        mixture_with_inversed_cov = gm.pack_mixture(amplitudes, positions, inversed_covariances) # shape first (m,1,s), then after view (m,s)
        output = gm.evaluate_inversed(mixture_with_inversed_cov, sample_points_in).view(batch_size, -1)
        loss1 = -torch.mean(torch.log(output + 0.00001), dim=1)

        # Calculate possible regularization losses
        cov_cost = torch.tensor(0)
        if penalize_long_gaussians:
            eigenvalues = torch.symeig(covariances, eigenvectors=True).eigenvalues
            largest_eigenvalue = eigenvalues[:, :, :, -1]
            smaller_eigenvalue = eigenvalues[:, :, :, -2]
            cov_cost = 0.1 * largest_eigenvalue / smaller_eigenvalue - 1
            cov_cost = cov_cost.where(cov_cost > torch.zeros_like(cov_cost), torch.zeros_like(cov_cost)).mean() #changed from sum, so it's comparable

        amp_cost = torch.tensor(0)
        if penalize_amplitude_differences:
            amp_cost = pi_normalized.std()
            #amp_cost = pi_normalized.max() / pi_normalized.min()
            amp_cost = amp_cost.where(amp_cost > torch.ones(amp_cost.shape)*1e-4, torch.zeros_like(amp_cost)).sum()*1000

        ext_cost = torch.tensor(0)
        if penalize_extends_differences:
            eigenvalues: Tensor = torch.symeig(covariances, eigenvectors=True).eigenvalues
            largest_eigenvalue = eigenvalues[:, :, :, -1]
            ext_cost = largest_eigenvalue.std()

        det_cost = torch.tensor(0)
        if penalize_small_determinants:
            #det_ref = torch.ones(det_cost.shape).to(device)*1e-12
            #det_cost = (det_ref - det_cost.where(det_cost < det_ref, det_ref)).sum()*1e12
            det_cost = -(torch.sigmoid(5*determinants/1e-12)-1).sum()*2

        loss = loss1 + cov_cost + amp_cost + ext_cost + det_cost
        #loss = loss1 + cov_cost + ext_cost

        assert not torch.isnan(loss).any()

        loss.backward()
        optimiser_pos.step()
        optimiser_cov.step()
        if optimiser_pi:
            optimiser_pi.step()
        #scheduler_pos.step()

        # Reconstruct Covariance Matrix
        covariances, inversed_covariances, determinants = calculate_covariance_matrices(cov_train_mode, cov_train_data, epsilon)

        # Calculate new amplitudes from priors
        if not constant_weights:
            if weight_softmax:
                pi_normalized = torch.nn.functional.softmax(pi_relative, dim=2)
            else:
                pi_sum = pi_relative.abs().sum(dim=2).view(batch_size, 1, 1)  # shape: (m,1) -> (m,1,1)
                pi_normalized = pi_relative.abs() / pi_sum  # shape (m,1,n)

        amplitudes = pi_normalized / (determinants.sqrt() * 15.74960995)

        # Send values to Tensorboard
        tensor_board_writer.add_scalar("0. training loss", loss.item(), k)
        tensor_board_writer.add_scalar("1. likelihood loss", loss1.item(), k)
        if penalize_long_gaussians:
            tensor_board_writer.add_scalar("2. length loss", cov_cost.item(), k)
        if penalize_amplitude_differences:
            tensor_board_writer.add_scalar("3. ampl loss", amp_cost.item(), k)
        if penalize_extends_differences:
            tensor_board_writer.add_scalar("4. extend loss", ext_cost.item(), k)
        if penalize_small_determinants:
            tensor_board_writer.add_scalar("5. det loss", det_cost.item(), k)

        print(f"iterations {k}: loss = {loss.item()}")
        # mixture_with_regular_cov = gm.pack_mixture(pi_normalized, positions, covariances.detach().clone())
        # integ = torch.abs(gm.integrate(mixture_with_regular_cov) - 1).view(batch_size).sum()
        # print(f"integral: {integ}")

        # Evaluate Validation Pointcloud
        if validation_pc is not None and k % 250 == 0:
            v_output = gm.evaluate_inversed(mixture_with_inversed_cov, validation_pc).view(batch_size, -1)
            v_loss = -torch.mean(torch.log(v_output + 0.001), dim=1)
            tensor_board_writer.add_scalar("Validation Likelihood Loss", v_loss.item(), k)
            del v_output
            del v_loss

        # Log Positions
        if log_positions:
            _positions = positions.detach().clone()
            _positions -= 0.5
            _positions *= scale
            pos_log[:,:,(k+1)%pos_log_size,:] = _positions.view(batch_size, n_components, 3)
            if _positions.max() > bbmax.max():
                print(f"{int(_positions.argmax() / 3)} is far oustide");
            if _positions.min() < bbmin.min():
                print(f"{int(_positions.argmin() / 3)} is far outside");

        if log_positions and (k+2) % pos_log_size == 0:
            for b in range(batch_size):
                for g in range(n_components):
                    f = open(f"{gm_path}/pos-b{b}-g{g}.bin", "a+b")
                    pdata = pos_log[b,g,:,:].view(-1)
                    bin = struct.pack('<' + 'd'*len(pdata),*pdata) #little endian!
                    f.write(bin)
                    f.close()

        # Use Visualizer
        #if True:
        if k % 250 == 0:
            _amplitudes, _positions, _covariances = rescale_gmm_to_gm(pi_normalized, positions,
                                                                           covariances, determinants, scale, scale2)

            _mixture = gm.pack_mixture(_amplitudes, _positions, _covariances)
            vis3d.set_pointclouds(pointclouds)
            vis3d.set_gaussian_mixtures(_mixture.detach().cpu(), isgmm=False)
            res = vis3d.render(k)
            for i in range(res.shape[0]):
                tensor_board_writer.add_image(f"GM {i}, Ellipsoids", res[i, 0, :, :, :], k, dataformats="HWC")
                tensor_board_writer.add_image(f"GM {i}, Positions", res[i, 1, :, :, :], k, dataformats="HWC")
                tensor_board_writer.add_image(f"GM {i}, Density", res[i, 2, :, :, :], k, dataformats="HWC")
                tensor_board_writer.flush()
                #gm.write_gm_to_ply(_amplitudes, _positions, _covariances, i, f"{gm_path}/pcgm-{i}.ply")
                gm.write_gm_to_ply(_amplitudes, _positions, _covariances, i, f"{gm_path}/pcgmm-{i}-" + str(k).zfill(5) + ".ply")
            gm.save(_mixture, f"{gm_path}/pcgm-{i}.gm")

            # _mixture = gm.pack_mixture(amplitudes.clone(), positions.clone(), covariances.clone())
            # vis3d.set_pointclouds(target.clone().cpu())
            # vis3d.set_gaussian_mixtures(_mixture.detach().cpu(), isgmm=False)
            # res = vis3d.render(k)
            # for i in range(res.shape[0]):
            #     tensor_board_writer.add_image(f"GMX {i}, Ellipsoids", res[i, 0, :, :, :], k, dataformats="HWC")
            #     tensor_board_writer.add_image(f"GMX {i}, Positions", res[i, 1, :, :, :], k, dataformats="HWC")
            #     tensor_board_writer.add_image(f"GMX {i}, Density", res[i, 2, :, :, :], k, dataformats="HWC")
            #     tensor_board_writer.flush()
            #     gm.write_gm_to_ply(amplitudes, positions, covariances, i, f"{gm_path}/pcgmmX-{i}-" + str(k).zfill(5) + ".ply")

        del output
        gc.collect()
        torch.cuda.empty_cache()

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")

    # TODO: Save final GM, Old code vv
    # positions = positions.detach()
    # covariances = inversed_covariances.detach().inverse().transpose(-1,-2)
    # #scaling
    # positions -= 0.5
    # positions *= scale
    # covariances *= scale2
    # pi_sum = pi_relative.abs().sum(dim=2).view(batch_size, 1, 1)  # shape: (m,1) -> (m,1,1)
    # pi_normalized = pi_relative.abs() / pi_sum  # shape (m,1,n)
    # amplitudes = pi_normalized / covariances.det()
    # _mixture = gm.pack_mixture(amplitudes, positions, covariances)
    # for i in range(batch_size):
    #     gm.write_gm_to_ply(amplitudes, positions, covariances, i, f"{gm_path}/pcgm-{i}.ply")
    # gm.save(_mixture, f"{gm_path}/pcgm.gm")
    return _mixture

# This function has suboptimal usability
def rescale_igmm_to_gm(pi_normalized: Tensor, positions: Tensor, inversed_covariances: Tensor, determinants: Tensor, scale: Tensor, scale2: Tensor):
    _positions = positions.detach().clone()
    _positions -= 0.5
    _positions *= scale

    _covariances = inversed_covariances.detach().inverse().transpose(-1, -2).clone()
    # Scaling of covariance by f@s@f', where f is the diagonal matrix of scalings
    # if all diag entries of f are the same, then this just results in times x^2, where x is the element of f
    _covariances *= scale2
    _determinants = determinants * torch.pow(scale2, 3)

    _amplitudes = pi_normalized / (_determinants.sqrt() * 15.74960995)

    return _amplitudes, _positions, _covariances

def rescale_gmm_to_gm(pi_normalized: Tensor, positions: Tensor, covariances: Tensor, determinants: Tensor, scale: Tensor, scale2: Tensor):
    _positions = positions.detach().clone()
    _positions -= 0.5
    _positions *= scale

    _covariances = covariances.detach().clone()
    # Scaling of covariance by f@s@f', where f is the diagonal matrix of scalings
    # if all diag entries of f are the same, then this just results in times x^2, where x is the element of f
    _covariances *= scale2
    _determinants = (determinants * torch.pow(scale2, 3)).view(1, 1, -1)

    _amplitudes = pi_normalized / (_determinants.sqrt() * 15.74960995)

    return _amplitudes, _positions, _covariances

def calculate_covariance_matrices(cov_train_mode: str, data: Tensor, epsilon: float):
    # Reconstruct Covariance Matrix
    if cov_train_mode == 'cholesky':
        cov_shape = data.shape
        cov_factor_mat_rec = torch.zeros((cov_shape[0], cov_shape[1], cov_shape[2], 3, 3)).to(data.device)
        cov_factor_mat_rec[:, :, :, 0, 0] = torch.abs(data[:, :, :, 0]) + epsilon
        cov_factor_mat_rec[:, :, :, 1, 1] = torch.abs(data[:, :, :, 1]) + epsilon
        cov_factor_mat_rec[:, :, :, 2, 2] = torch.abs(data[:, :, :, 2]) + epsilon
        cov_factor_mat_rec[:, :, :, 1, 0] = data[:, :, :, 3]
        cov_factor_mat_rec[:, :, :, 2, 0] = data[:, :, :, 4]
        cov_factor_mat_rec[:, :, :, 2, 1] = data[:, :, :, 5]
        covariances = cov_factor_mat_rec @ cov_factor_mat_rec.transpose(-2, -1)
        cov_factor_mat_rec_inv = cov_factor_mat_rec.inverse()
        inversed_covariances = cov_factor_mat_rec_inv.transpose(-2, -1) @ cov_factor_mat_rec_inv
        #numerically better way of calculating the determinants
        determinants = torch.pow(cov_factor_mat_rec[:, :, :, 0,0],2)*\
                       torch.pow(cov_factor_mat_rec[:, :, :, 1,1],2)*\
                       torch.pow(cov_factor_mat_rec[:, :, :, 2,2],2)
    else:
        inversed_covariances = data @ data.transpose(-2, -1) + torch.eye(3, 3, device=data.device) * epsilon
        covariances = inversed_covariances.inverse()
        determinants = covariances.det()
    if torch.isnan(inversed_covariances).any():
        print()
    assert not torch.isnan(inversed_covariances).any()
    assert not torch.isinf(inversed_covariances).any()
    assert (determinants > 0).all()
    return covariances, inversed_covariances, determinants

def test():
    torch.manual_seed(0)
    np.random.seed(0)

    pcs = pointcloud.load_pc_from_off(
        # "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/daav/face02.off")
        # "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/TEST3.off")
        # "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud/ModelNet10/chair/train/chair_0030.off")
        # "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-lores/ModelNet10/chair/train/chair_0030.off")
        # "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-lores-validation/ModelNet10/chair/train/chair_0030.off")
        "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-hires/ModelNet10/chair/train/chair_0030.off")
        # "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud/ModelNet10/toilet/train/toilet_0001.off")
    # validation = pointcloud.load_pc_from_off(
    #    "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-lores-validation/ModelNet10/chair/train/chair_0030.off")

    #gms = gm.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/TEST3-preiner2.ply", True)
    #gms = gm.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/TEST3-PREINER3/pcgmm-0-109750.ply", False).cuda()
    #gms = gm.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30HR.ply", True).cuda()
    #gms = gm.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/GDPREINER-onlyp-0.001-loged/pcgmm-0-initial.ply", False).cuda()
    #gms = gm.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/gdp-inout/pcgmm-0-initial.ply", False).cuda()
    #gms = gms[0, :, :, :].view(1, 1, 88, 13)
    # gms = gm.load(
    #     "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/exvs-lr-0.001/pcgm-.gm")[0]
    # gms = gms[0, :, :, :].view(1, 1, 100, 13)  # [m,1,n_components,13]

    name = input('Name for this training (or empty for auto): ')
    ad_algorithm(
        pointclouds=pcs,
        n_components=32684,
        n_iterations=1000000,
        device='cuda',
        name=name,
        #mixture=gms,
        #start_epoch=0,
        #validation_pc=validation,
        init_positions_from_pointcloud=True,
        init_weights_equal=True,
        penalize_long_gaussians=False,
        penalize_amplitude_differences=False,
        penalize_extends_differences=False,
        penalize_small_determinants=True,
        weight_softmax=False,
        constant_weights=False,
        log_positions=False,
        cov_train_mode='cholesky',
        learn_rate_pos=0.001,
        learn_rate_cov=0.0001,
        learn_rate_wei=0.0005
    )

test()
