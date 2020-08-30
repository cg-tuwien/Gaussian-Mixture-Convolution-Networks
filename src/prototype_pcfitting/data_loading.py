from typing import List

import torch
import os
import gmc.mixture as gm


def load_pc_from_off(path: str) -> torch.Tensor:
    # Loads a pointcloud from an off-file at the given path
    # Returns it as a 3d cpu-tensor with shape [1,n,3]
    file = open(path, "r")
    if 'OFF' != file.readline().strip():
        raise Exception("Not a valid OFF header!")
    n_points = int(file.readline().strip().split(" ")[0])
    points = [[[float(s) for s in file.readline().strip().split(" ") if s != ''] for pt in range(n_points)]]
    file.close()
    return torch.tensor(points, dtype=torch.float32)


def write_pc_to_off(path: str, pc: torch.Tensor):
    # Writes a single pointcloud to an off-file at the given path
    # The pointcloud is given as a [n, 3] or an [1, n, 3] tensor
    if len(pc.shape) == 3:
        pc = pc.view(-1, 3)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    file = open(path, "w+")
    file.write("OFF\n")
    file.write(f"{pc.shape[0]} 0 0\n")
    for point in pc:
        file.write(f"{point[0]} {point[1]} {point[2]}\n")
    file.close()


def sample(pcbatch: torch.Tensor, n_sample_points: int) -> torch.Tensor:
    # Samples points from the given pointclouds
    # The pcbatch is given as a [m, n, 3] tensor
    # n_sample_points random points are sampled for each point cloud
    # This returns a tensor of size [m, n_sample_points, 3]
    batch_size, point_count, _ = pcbatch.shape
    sample_point_idz = torch.randperm(point_count)[0:n_sample_points]   # Shape: (s)
    sample_points = pcbatch[:, sample_point_idz, :]                     # Shape: (m,s,3)
    return sample_points


def read_gm_from_ply(filename: str, ismodel: bool) -> torch.Tensor:
    return gm.read_gm_from_ply(filename, ismodel)


def write_gm_to_ply(weights: torch.Tensor, positions: torch.Tensor,
                    covariances: torch.Tensor, index: int, filename: str):
    gm.write_gm_to_ply(weights, positions, covariances, index, filename)


def save_gms(gmbatch: torch.Tensor, gmmbatch: torch.Tensor, basepath: str, names: List[str]):
    gmw = gm.weights(gmmbatch)
    gma = gm.weights(gmbatch)
    gmp = gm.positions(gmbatch)
    gmc = gm.covariances(gmbatch)
    for i in range(gmbatch.shape[0]):
        write_gm_to_ply(gmw, gmp, gmc, i, f"{basepath}/{names[i]}.gmm.ply")
        write_gm_to_ply(gma, gmp, gmc, i, f"{basepath}/{names[i]}.gma.ply")