import typing
import os.path

import numpy as np
import torch
from torch import Tensor

import gmc.mixture as gm
import gmc.config as config


def save(mixture: Tensor, file_name: str, meta_info=None) -> None:
    assert gm.is_valid_mixture(mixture)
    dictionary = {
        "type": "gm.Mixture",
        "version": 6,
        "data": mixture.detach().cpu(),
        "meta_info": meta_info
    }
    torch.save(dictionary, config.data_base_path / file_name)


def load(file_name: str) -> typing.Tuple[Tensor, typing.Any]:
    dictionary = torch.load(config.data_base_path / file_name)
    assert dictionary["type"] == "gm.Mixture"
    if dictionary["version"] == 3:
        weights = dictionary["weights"]
        n_components = weights.shape[1]
        positions = dictionary["positions"]
        n_dims = positions.shape[2]
        covariances = dictionary["covariances"]
        mixture = gm.pack_mixture(weights.view(-1, 1, n_components), positions.view(-1, 1, n_components, n_dims), covariances.view(-1, 1, n_components, n_dims, n_dims))
        mixture = gm.convert_amplitudes_to_priors(mixture)

    elif dictionary["version"] == 4:
        weights = dictionary["weights"]
        positions = dictionary["positions"]
        covariances = dictionary["covariances"]
        mixture = gm.pack_mixture(weights, positions, covariances)
        mixture = gm.convert_amplitudes_to_priors(mixture)
    elif dictionary["version"] == 5:
        mixture = dictionary["data"]
        mixture = gm.convert_amplitudes_to_priors(mixture)
    else:
        assert dictionary["version"] == 6
        mixture = dictionary["data"]

    assert gm.is_valid_mixture(mixture)
    return mixture, dictionary["meta_info"]


def write_gm_to_ply(m_weights: Tensor, m_positions: Tensor, m_covariances: Tensor, index: int, filename: str):
    # Writes a single Gaussian Mixture to a ply-file
    # The parameter "index" defines which element in the batch to use
    weight_shape = m_weights.shape  # should be (m,1,n)
    pos_shape = m_positions.shape  # should be (m,1,n,3)
    cov_shape = m_covariances.shape  # should be (m,1,n,3,3)
    assert len(weight_shape) == 3
    assert len(pos_shape) == 4
    assert len(cov_shape) == 5
    assert weight_shape[0] == pos_shape[0] == cov_shape[0]
    assert weight_shape[1] == pos_shape[1] == cov_shape[1] == 1
    assert weight_shape[2] == pos_shape[2] == cov_shape[2]
    assert pos_shape[3] == cov_shape[3] == 3
    assert cov_shape[4] == 3
    n = weight_shape[2]

    _weights = m_weights[index, 0, :].view(n)
    _positions = m_positions[index, 0, :, :].view(n, 3)
    _covs = m_covariances[index, 0, :, :, :].view(n, 3, 3)

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    file = open(filename, "w+")
    file.write("ply\nformat ascii 1.0\n")
    file.write(f"element component {n}\n")
    file.write("property float x\nproperty float y\nproperty float z\n")
    file.write("property float covxx\nproperty float covxy\nproperty float covxz\n")
    file.write("property float covyy\nproperty float covyz\nproperty float covzz\n")
    file.write("property float weight\nend_header\n")

    file.close()
    file = open(filename, "ab")  # open in append mode

    data = torch.zeros(n, 10)
    data[:, 0:3] = _positions
    data[:, 3] = _covs[:, 0, 0]
    data[:, 4] = _covs[:, 0, 1]
    data[:, 5] = _covs[:, 0, 2]
    data[:, 6] = _covs[:, 1, 1]
    data[:, 7] = _covs[:, 1, 2]
    data[:, 8] = _covs[:, 2, 2]
    data[:, 9] = _weights

    np.savetxt(file, data.detach().numpy(), delimiter="  ")

    file.close()


def write_gm_to_ply2(m: Tensor, path_without_ending: str):
    assert gm.n_dimensions(m) == 3
    for b in range(gm.n_batch(m)):
        for l in range(gm.n_layers(m)):
            t = m[b:b+1, l:l+1, :, :]
            write_gm_to_ply(gm.weights(t), gm.positions(t), gm.covariances(t), 0, f"{path_without_ending}_b{b}_l{l}.ply");


def read_gm_from_ply(filename: str, ismodel: bool = False, device='cuda') -> Tensor:
    # Reads a Gaussian Mixture from a ply-file
    # The parameter "ismodel" defines whether the weights in the file represent amplitudes (False) or priors (True)
    # The weights of the returned GM are amplitudes.
    fin = open(filename)
    header = True
    index = 0
    for line in fin:
        if header:
            if line.startswith("element component "):
                number = int(line[18:])
                gmpos = torch.zeros((1, 1, number, 3), device=device)
                gmcov = torch.zeros((1, 1, number, 3, 3), device=device)
                gmwei = torch.zeros((1, 1, number), device=device)
            elif line.startswith("end_header"):
                header = False
        else:
            elements = line.split("  ")
            gmpos[0, 0, index, :] = torch.tensor([float(e) for e in elements[0:3]])
            gmcov[0, 0, index, 0, 0] = float(elements[3])
            gmcov[0, 0, index, 0, 1] = gmcov[0, 0, index, 1, 0] = float(elements[4])
            gmcov[0, 0, index, 0, 2] = gmcov[0, 0, index, 2, 0] = float(elements[5])
            gmcov[0, 0, index, 1, 1] = float(elements[6])
            gmcov[0, 0, index, 1, 2] = gmcov[0, 0, index, 2, 1] = float(elements[7])
            gmcov[0, 0, index, 2, 2] = float(elements[8])
            gmwei[0, 0, index] = float(elements[9])
            index = index + 1
    fin.close()
    if ismodel:
        gmwei /= gmwei.sum()
        amplitudes = gmwei / (gmcov.det().sqrt() * 15.74960995)
        return gm.pack_mixture(amplitudes, gmpos, gmcov)
    else:
        return gm.pack_mixture(gmwei, gmpos, gmcov)
