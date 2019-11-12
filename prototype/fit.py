import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn as nn
import torch.optim as optim
import numpy.random as nprnd

import mat_tools
import gm

from gm import Mixture
from torch import Tensor


def em_image(image: Tensor, n_components: int, n_iterations: int, device: torch.device = 'cpu') -> Mixture:
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    fitting_start = time.time()

    mixture = gm.generate_random_mixtures(n_components, 2,
                                          0.5,
                                          5 * min(width, height) / math.sqrt(n_components),
                                          0, 1, device=device)
    mixture.positions += 0.5
    mixture.positions *= torch.tensor([[width], [height]], dtype=torch.float, device=device).expand_as(mixture.positions)

    xv, yv = torch.meshgrid([torch.arange(0, width, 1, dtype=torch.float, device=device),
                             torch.arange(0, height, 1, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0)
    if device == 'cuda':
        values = image.view(-1).cuda()
    else:
        values = image.view(-1)

    print("starting expectation maximisation")
    for k in range(n_iterations):
        print(f"classifying..")
        selected_components = mixture.max_component_many_xes(xes)
        print(f"updating..")
        new_mixture = gm.generate_null_mixture(n_components, 2, device=mixture.device())
        n_pixels = torch.zeros(n_components, device=new_mixture.device())
        for i in range(values.size()[0]):
            w = values[i]
            x = xes[:, i]
            c = selected_components[i]
            n_pixels[c] += 1
            new_mixture.factors[c] += w.float()
            dx = x - new_mixture.positions[:, c]
            new_mixture.positions[:, c] += w / new_mixture.factors[c] * dx
            new_mixture.covariances[:, c] += w * (1 - w / new_mixture.factors[c]) * mat_tools.triangle_outer_product(dx)

        for j in range(new_mixture.number_of_components()):
            if new_mixture.factors[j] > 1:
                new_mixture.covariances[:, j] /= new_mixture.factors[j] - 1
            if n_pixels[j] > 0:
                new_mixture.factors[j] /= n_pixels[j]

            # minimum variance
            new_mixture.covariances[:, j] += mat_tools.gen_identity_triangle(1, new_mixture.dimensions, device=new_mixture.device()).view(-1) * 0.1

        print(f"inverting..")
        new_mixture.inverted_covariances = mat_tools.triangle_invert(new_mixture.covariances)
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

def ad_image(image: Tensor, n_components: int, n_iterations: int = 8, device: torch.device = 'cpu') -> Mixture:
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    mixture = em_image(image, n_components, 5, device='cpu')
    if device == 'cuda':
        mixture = mixture.cuda()

    fitting_start = time.time()

    xv, yv = torch.meshgrid([torch.arange(0, width, 1, dtype=torch.float, device=device),
                             torch.arange(0, height, 1, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0)

    if device == 'cuda':
        values = image.view(-1).cuda() / 255.0
    else:
        values = image.view(-1) / 255.0

    mixture.factors /= 255.0

    mixture = mixture
    mixture.factors.requires_grad = True;
    mixture.positions.requires_grad = True;
    mixture.inverted_covariances.requires_grad = True;

    # def trim_icov_gradient_function(grad: Tensor) -> Tensor:
    #     print(f"original icov gradient: \n{grad}")
    #     dets = (mat_tools.triangle_det(mixture.inverted_covariances - grad) - 0.5)
    #     grad_copy = grad.clone()
    #     correction = torch.cat((0.25 * dets.view(1, -1), -0.5 * dets.view(1, -1), 0.25 * dets.view(1, -1)), dim=0)
    #     correction[:, dets > 0] = torch.zeros(1, dtype=torch.float32, device=mixture.device())
    #     grad_copy += correction
    #
    #     print(f"new icov gradient: \n{grad_copy}\n\n")
    #     return grad_copy


    # trim_icov_gradient_hook = mixture.inverted_covariances.register_hook(trim_icov_gradient_function)
    # print(mixture.inverted_covariances)
    # print(mat_tools.triangle_det(mixture.inverted_covariances))
    # mixture.inverted_covariances.backward(-mixture.inverted_covariances.clone())
    # mixture.inverted_covariances.data -= mixture.inverted_covariances.grad
    # print(mixture.inverted_covariances)
    # print(mat_tools.triangle_det(mixture.inverted_covariances))
    optimiser = optim.Adam([mixture.factors, mixture.positions, mixture.inverted_covariances], lr=0.005)

    print("starting gradient descent")
    # mixture.debug_show(0, 0, image.shape[1], image.shape[0], 1)
    for k in range(n_iterations):
        optimiser.zero_grad()
        output = mixture.evaluate_many_xes(xes)
        # assert not torch.isnan(mixture.inverted_covariances).any()
        # assert not torch.isinf(mixture.inverted_covariances).any()
        loss = torch.mean(torch.abs(output - values))

        # regularisation_1 = 0.1 * torch.mean(torch.max(torch.ones(1, dtype=torch.float, device=device) * (-math.log(0.01)),
        #                                               -torch.log((mat_tools.triangle_det(mixture.inverted_covariances)).abs())))
        #
        # regularisation_1 = 0.1 * torch.mean(torch.max(torch.ones(1, dtype=torch.float, device=device),
        #                                               (torch.ones(1, dtype=torch.float, device=device) * 0.05)/mat_tools.triangle_det(mixture.inverted_covariances).abs()))
        #
        # regularisation_2 = 0.005 * mixture.inverted_covariances.mean()
        #
        # regularisation_3 = 0.005 * torch.mean(torch.max(torch.zeros(1, dtype=torch.float, device=device),
        #                                                -torch.log(mixture.inverted_covariances.abs().mean(dim=0)[0]))**2)
        # (loss + regularisation_1 + regularisation_2 + regularisation_3).backward()
        # (loss + regularisation_1).backward()
        loss.backward()
        optimiser.step()

        # dets = mat_tools.triangle_det(mixture.inverted_covariances.detach())
        # dets -= 0.01
        # dets[dets > 0] = 0
        # dets = -dets / 4
        # sq_sign = torch.sign(mixture.inverted_covariances.detach()[0] * mixture.inverted_covariances.detach()[2])
        # mixture.inverted_covariances.detach()[0] += dets * sq_sign
        # mixture.inverted_covariances.detach()[1] -= 2 * dets * torch.sign(mixture.inverted_covariances.detach()[1])
        # mixture.inverted_covariances.detach()[2] += dets * sq_sign

        # print(mixture.inverted_covariances.detach())
        icovs = mat_tools.triangle_to_normal(mixture.inverted_covariances.detach()).transpose(0, 2)
        # print(icovs)
        (eigvals, eigvecs) = torch.symeig(icovs, eigenvectors=True)
        eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float, device=device))
        icovs = torch.matmul(eigvecs, torch.matmul(eigvals.diag_embed(), eigvecs)) # V Lambda V, no need for a transpose because of symmetry
        # print(eigvals)
        # print(eigvecs)
        #
        # print(icovs)
        mixture.inverted_covariances.detach()[:, :] = mat_tools.normal_to_triangle(icovs.transpose(0, 2))
        # print(mixture.inverted_covariances.detach())

        if k % 50 == 0:
            print(f"iterations {k}: loss = {loss.item()}, min det = {torch.min(mat_tools.triangle_det(mixture.inverted_covariances.detach()))}")#, regularisation_1 = {regularisation_1.item()}, "
                  # f"regularisation_2 = {regularisation_2.item()}, regularisation_3 = {regularisation_3.item()}")
            mixture.debug_show(0, 0, image.shape[1], image.shape[0], 1)

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    mixture.detach()
    mixture.covariances = mat_tools.triangle_invert(mixture.inverted_covariances)
    return mixture


image: np.ndarray = plt.imread("/home/madam/cloud/Photos/fire_small.jpg")
image = image.mean(axis=2)
m1 = em_image(torch.tensor(image, dtype=torch.float32), n_components=2500, n_iterations=5, device='cuda')
# m1 = ad_image(torch.tensor(image, dtype=torch.float32), n_components=2500, n_iterations=1500, device='cuda')

m1 = m1.cpu()

m1.debug_show(0, 0, image.shape[1], image.shape[0], 1)

k1 = gm.generate_null_mixture(9, 2, device=m1.device())
k1.factors[0] = -1
k1.factors[1] = 1
k1.positions[:, 0] = torch.tensor([0, -5], dtype=torch.float32, device=m1.device())
k1.positions[:, 1] = torch.tensor([0, 5], dtype=torch.float32, device=m1.device())
k1.covariances[:, 0] = torch.tensor([5, 0, 5], dtype=torch.float32, device=m1.device())
k1.covariances[:, 1] = torch.tensor([5, 0, 5], dtype=torch.float32, device=m1.device())
k1.debug_show(-128, -128, 128, 128, 1)

k2 = gm.generate_random_mixtures(9, 2, device=m1.device())
k2.debug_show(-128, -128, 128, 128, 1)

k3 = gm.generate_random_mixtures(9, 2, device=m1.device())
k3.debug_show(-128, -128, 128, 128, 1)

conv_start = time.time()
conved1 = gm.convolve(m1, k1)
conved2 = gm.convolve(m1, k2)
conved3 = gm.convolve(m1, k3)
conv_end = time.time()
print(f"convolution time: {conv_end - conv_start}")

conved1.debug_show(0, 0, image.shape[1], image.shape[0], 1)
conved1.show_after_activation(0, 0, image.shape[1], image.shape[0], 1)
conved2.debug_show(0, 0, image.shape[1], image.shape[0], 1)
conved2.show_after_activation(0, 0, image.shape[1], image.shape[0], 1)
conved3.debug_show(0, 0, image.shape[1], image.shape[0], 1)
conved3.show_after_activation(0, 0, image.shape[1], image.shape[0], 1)

