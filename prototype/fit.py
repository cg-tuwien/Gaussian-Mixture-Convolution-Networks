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


def em_image(image: Tensor, n_components: int, n_iterations: int) -> Mixture:
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    mixture = gm.generate_random_mixtures(n_components, 2,
                                          0.5,
                                          5 * min(width, height) / math.sqrt(n_components),
                                          0, 1)
    mixture.positions += 0.5
    mixture.positions *= torch.tensor([[width], [height]], dtype=torch.float).expand_as(mixture.positions)
    
    xv, yv = torch.meshgrid([torch.arange(0, width, 1, dtype=torch.float), torch.arange(0, height, 1, dtype=torch.float)])
    xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0)
    values = image.view(-1)
    
    for k in range(n_iterations):
        selected_components = mixture.max_component_many_xes(xes)
        new_mixture = gm.generate_null_mixture(n_components, 2)
        n_pixels = torch.zeros(n_components)
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
            new_mixture.covariances[:, j] += mat_tools.gen_identity_triangle(1, new_mixture.dimensions).view(-1) * 0.1
        new_mixture.inverted_covariances = mat_tools.triangle_invert(new_mixture.covariances)

        new_mixture.debug_show(0, 0, width, height, 1)
        mixture = new_mixture

    return mixture


## https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=922055093#Online
## https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
## essentially, there are two weighting schemes. one of them adds up to 1, the other is int (repeats / frequency).
## i don't know whether it's possible to handle float weights that don't add up to 1 in an online algorithm
## this method returns the same as np.cov(xes, fweights=w)
#def _fit_gaussian(data: np.ndarray, weights: np.ndarray, dims: int):
    #w_sum = 0
    #mean = np.zeros(dims)
    #cov = np.zeros((dims, dims))
    #for i in range(data.shape[1]):
        #w = weights[i]
        #w_sum += w
        #x = data[:, i]
        #dx = x - mean
        #mean += w/w_sum * dx
        #dx.shape = (dims, 1)
        ## update scheme in wikipedia uses a different w_sum for x and y. the term (1-w/wsum) corrects that
        #cov += w * (1 - w/w_sum) * dx @ dx.T
    #cov /= (w_sum - 1)
    
    #print(f"mean = {mean}, \ncov=\n{cov}")


#dims = 4
#xes = nprnd.rand(dims, 20)
#w = (nprnd.rand(20)*20).astype(int)
#print(f"numpy mean = {np.average(xes, axis=1, weights=w)}, \n numpy cov=\n{np.cov(xes, fweights=w)}")
#my_funs(xes, w, dims)

def ad_image(image: Tensor, n_components: int, n_iterations: int = 8) -> Mixture:
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    mixture = em_image(image, n_components, 8)

    xv, yv = torch.meshgrid([torch.arange(0, width, 1, dtype=torch.float), torch.arange(0, height, 1, dtype=torch.float)])
    xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0).cuda()
    values = image.view(-1).cuda() / 255.0

    mixture.factors /= 255.0

    mixture = mixture.cuda()
    mixture.factors.requires_grad = True;
    mixture.positions.requires_grad = True;
    mixture.inverted_covariances.requires_grad = True;

    optimiser = optim.Adam([mixture.factors, mixture.positions, mixture.inverted_covariances], lr=0.0001)

    print("starting gradient descent")
    for k in range(n_iterations):
        if k % 50 == 0:
            mixture.debug_show(0, 0, image.shape[1], image.shape[0], 0.5)
        optimiser.zero_grad()
        output = mixture.evaluate_many_xes(xes)
        loss = torch.mean(torch.abs(output - values))
        regularisation = torch.sqrt(torch.sum(1.0 / mat_tools.triangle_det(mixture.inverted_covariances)))
        (loss+regularisation).backward()
        optimiser.step()

    mixture.covariances = mat_tools.triangle_invert(mixture.inverted_covariances)
    return mixture


image: np.ndarray = plt.imread("/home/madam/cloud/Photos/fire_small.jpg")
image = image.mean(axis = 2)
fitting_start = time.time()
m1 = ad_image(torch.tensor(image, dtype=torch.float32), 10000, 5000)
fitting_end = time.time()
print(f"fitting time: {fitting_end - fitting_start}")

m1.debug_show(0, 0, image.shape[1], image.shape[0], 1)

k1 = gm.generate_random_mixtures(9, 2).cuda()
# k1.factors[0] = -1
# k1.factors[1] = 1
# k1.positions[:, 0] = torch.tensor([0, -5], dtype=torch.float32)
# k1.positions[:, 1] = torch.tensor([0, 5], dtype=torch.float32)
# k1.covariances[:, 0] = torch.tensor([5, 0, 5], dtype=torch.float32)
# k1.covariances[:, 1] = torch.tensor([5, 0, 5], dtype=torch.float32)
k1.debug_show(-128, -128, 128, 128, 1)

conv_start = time.time()
m2 = gm.convolve(m1, k1)
conv_end = time.time()
print(f"convolution time: {conv_end - conv_start}")
m2.debug_show(0, 0, image.shape[1], image.shape[0], 1)
m2.show_after_activation(0, 0, image.shape[1], image.shape[0], 1)

a = 3
