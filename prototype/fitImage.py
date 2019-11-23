import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.optim as optim

import gm

from gm import Mixture
from torch import Tensor


def em_algorithm(image: Tensor, n_components: int, n_iterations: int, device: torch.device = 'cpu') -> Mixture:
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    fitting_start = time.time()

    mixture = gm.generate_random_mixtures(1, n_components, 2,
                                          0.5,
                                          5 * min(width, height) / math.sqrt(n_components),
                                          0, 1, device=device)
    mixture.positions += 0.5
    mixture.positions *= torch.tensor([[[width, height]]], dtype=torch.float, device=device).expand_as(mixture.positions)

    xv, yv = torch.meshgrid([torch.arange(0, width, 1, dtype=torch.float, device=device),
                             torch.arange(0, height, 1, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2)
    if device == 'cuda':
        values = image.view(-1).cuda()
    else:
        values = image.view(-1)

    values = torch.max(values, torch.ones(1, dtype=values.dtype, device=values.device))

    print("starting expectation maximisation")
    for k in range(n_iterations):
        print(f"classifying..")
        selected_components = mixture.max_component_many_xes(xes)
        print(f"updating..")
        new_mixture = gm.generate_null_mixture(1, n_components, 2, device=mixture.device())
        n_pixels = torch.zeros(n_components, device=new_mixture.device())
        for i in range(values.size()[0]):
            w = values[i]
            x = xes[0, i, :]
            c = selected_components[i]
            n_pixels[c] += 1
            new_mixture.weights[0, c] += w.float()
            dx = x - new_mixture.positions[0, c, :]
            new_mixture.positions[0, c, :] += w / new_mixture.weights[0, c] * dx
            new_mixture.covariances[0, c, :, :] += w * (1 - w / new_mixture.weights[0, c]) * (dx.view(-1, 1) @ dx.view(1, -1))

        for j in range(new_mixture.n_components()):
            if new_mixture.weights[0, j] > 1:
                new_mixture.covariances[0, j, :, :] /= new_mixture.weights[0, j] - 1
            if n_pixels[j] > 0:
                new_mixture.weights[0, j] /= n_pixels[j]

        # minimum variance
        new_mixture.covariances[0, :, :, :] += torch.eye(new_mixture.n_dimensions(), device=new_mixture.device()) * 0.1

        print(f"inverting..")
        new_mixture.inverted_covariances = torch.inverse(new_mixture.covariances)
        print(f"iterations {k} finished")

        # new_mixture.debug_show(0, 0, width, height, 1)
        mixture = new_mixture

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    return mixture


## https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=922055093#Online
## https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
## essentially, there are two weighting schemes. one of them adds up to 1, the other is int (repeats / frequency).
## i don't know whether it's possible to handle float weights that don't add up to 1 in an online algorithm
## this method returns the same as np.cov(xes, fweights=w)
# def _fit_gaussian(data: np.ndarray, weights: np.ndarray, dims: int):
# w_sum = 0
# mean = np.zeros(dims)
# cov = np.zeros((dims, dims))
# for i in range(data.shape[1]):
# w = weights[i]
# w_sum += w
# x = data[:, i]
# dx = x - mean
# mean += w/w_sum * dx
# dx.shape = (dims, 1)
## update scheme in wikipedia uses a different w_sum for x and y. the term (1-w/wsum) corrects that
# cov += w * (1 - w/w_sum) * dx @ dx.T
# cov /= (w_sum - 1)

# print(f"mean = {mean}, \ncov=\n{cov}")


# dims = 4
# xes = nprnd.rand(dims, 20)
# w = (nprnd.rand(20)*20).astype(int)
# print(f"numpy mean = {np.average(xes, axis=1, weights=w)}, \n numpy cov=\n{np.cov(xes, fweights=w)}")
# my_funs(xes, w, dims)

def ad_algorithm(image: Tensor, n_components: int, n_iterations: int = 8, device: torch.device = 'cpu') -> Mixture:
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    mixture = em_algorithm(image, n_components, 5, device='cpu')
    # mixture = gm.generate_random_mixtures(n_batch=1, n_components=n_components, n_dims=2,
    #                                       pos_radius=0.5, cov_radius=5 * min(width, height) / math.sqrt(n_components),
    #                                       factor_min=0, factor_max=1, device=device)
    # mixture.positions += 0.5
    # mixture.positions *= torch.tensor([[[width, height]]], dtype=torch.float, device=device)

    mixture.debug_show(0, 0, 0, width, height, min(width, height) / 100)
    if device == 'cuda':
        mixture = mixture.cuda()

    fitting_start = time.time()

    if device == 'cuda':
        target = image.view(-1).cuda() / 255.0
    else:
        target = image.view(-1) / 255.0

    mixture.weights /= 255.0

    mixture.weights.requires_grad = True;
    mixture.positions.requires_grad = True;

    mixture.debug_show(0, 0, 0, width, height, 1)
    (eigvals, eigvecs) = torch.symeig(mixture.inverted_covariances, eigenvectors=True)
    eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float, device=device))
    icov_factor = torch.matmul(eigvecs, eigvals.sqrt().diag_embed())
    icov_factor.requires_grad = True

    optimiser = optim.Adam([mixture.weights, mixture.positions, icov_factor], lr=0.01)

    print("starting gradient descent")
    for k in range(n_iterations):
        optimiser.zero_grad()
        xes = torch.rand(1, 50*50, 2, device=device, dtype=torch.float32) * torch.tensor([[[width, height]]], device=device, dtype=torch.float32)
        xes = xes.floor() + 0.5
        xes_indices = xes.type(torch.long).view(-1, 2)
        mixture.inverted_covariances = icov_factor @ icov_factor.transpose(-2, -1) + torch.eye(2, 2, device=mixture.device()) * 0.005
        output = mixture.evaluate_few_xes(xes)
        assert not torch.isnan(mixture.inverted_covariances).any()
        assert not torch.isinf(mixture.inverted_covariances).any()
        xes_indices = xes_indices[:, 0] * int(height) + xes_indices[:, 1]
        loss = torch.mean(torch.abs(output - target[xes_indices]))

        loss.backward()
        optimiser.step()

        if k % 100 == 0:
            print(f"iterations {k}: loss = {loss.item()}, min det = {torch.min(torch.det(mixture.inverted_covariances.detach()))}")#, regularisation_1 = {regularisation_1.item()}, "
                  # f"regularisation_2 = {regularisation_2.item()}, regularisation_3 = {regularisation_3.item()}")
            mixture.covariances = torch.inverse(mixture.inverted_covariances)
            md = mixture.detach().cpu()
            md.save("fire_small_mixture")
            mixture.debug_show(0, 0, 0, image.shape[1], image.shape[0], min(image.shape[0], image.shape[1]) / 100)

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    mixture = mixture.detach()
    mixture.covariances = torch.inverse(mixture.inverted_covariances)
    return mixture


def test():
    # image: np.ndarray = plt.imread("/home/madam/cloud/Photos/fire_small.jpg")
    image = plt.imread("/home/madam/Downloads/mnist_png/training/8/17.png")
    if len(image.shape) == 3:
        image = image.mean(axis=2)
    # m1 = em_algorithm(torch.tensor(image, dtype=torch.float32), n_components=2500, n_iterations=5, device='cpu')
    m1 = ad_algorithm(torch.tensor(image, dtype=torch.float32), n_components=50, n_iterations=1000, device='cuda')
    m1.save("mnist_8")

    m1 = m1.cpu()

    m1.debug_show(0, 0, 0, image.shape[1], image.shape[0], min(image.shape[0], image.shape[1]) / 100)


test()
