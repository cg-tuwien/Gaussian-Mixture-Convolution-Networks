import math
import time
import typing

import matplotlib.pyplot as plt
import numpy as np
import pomegranate
import sklearn.mixture
import torch
from torch import Tensor
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms
from torch.distributions.categorical import Categorical as DiscreteDistribution

import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import gmc.inout as inout
import gmc.render as render


def debug_render(mixture: Tensor, orig_size: typing.Tuple[int, int] = (28, 28), image_size: typing.Tuple[int, int] = (200, 200), clamp: typing.Tuple[float, float] = (0, 1.5)):
    mixture = mixture.view(1, gm.n_batch(mixture), gm.n_components(mixture), -1)
    images = render.render(mixture, torch.zeros(1, 1), x_low=0, x_high=orig_size[0], y_low=0, y_high=orig_size[1], width=image_size[0], height=image_size[1])
    images = render.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
    return images[:, :, :3]


m1 = gm.pack_mixture(torch.tensor((0.3, 0.2, 0.5, 0.8)).view(1, 1, -1),
                     torch.tensor(((5, 5), (10, 10), (15, 15), (20, 20))).view(1, 1, -1, 2),
                     torch.tensor(((1, -0.7, -0.7, 1), (1, -0.7, -0.7, 1), (1, -0.7, -0.7, 1), (1, -0.7, -0.7, 1))).view(1, 1, -1, 2, 2),)

rendering = debug_render(m1, orig_size=(25, 25), clamp=(0, 0.5))
plt.imshow(rendering)
plt.show()

w1 = gm.weights(m1)
p1 = gm.positions(m1)
c1 = gm.covariances(m1)

wprime = w1.unsqueeze(-1) / w1.sum()
w2 = w1.mean(2, keepdim=True)
p2 = (wprime * p1).mean(2, keepdim=True)
c2 = (((p1-p2) * wprime).unsqueeze(-1) @ ((p1-p2) * wprime).unsqueeze(-1).transpose(-1, -2)).mean(2, keepdim=True)  # maybe move wprime out. add cov mean on top. seems like em isn't aware of the direction either

m2 = gm.pack_mixture(w2, p2, c2)

rendering = debug_render(m2, orig_size=(25, 25), clamp=(0, 0.5))
plt.imshow(rendering)
plt.show()