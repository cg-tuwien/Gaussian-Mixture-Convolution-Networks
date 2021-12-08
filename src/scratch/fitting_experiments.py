import math
import time
import typing

import matplotlib.image
import matplotlib.pyplot as plt
import scipy.misc
import torch
from torch import Tensor
import torch.optim as optim
import torch.utils.data
import torchvision.datasets
import torchvision.transforms
import torch.cuda
from torch.distributions.categorical import Categorical as DiscreteDistribution

from gmc.cpp.extensions.convolution_fitting.binding import apply as cpp_conv
import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import gmc.inout as inout
import gmc.render as render
import gmc.fitting as fitting


def my_render(mixture: Tensor):
    constant = torch.zeros(1, 1, device=mixture.device)
    assert gm.is_valid_mixture_and_constant(mixture, constant)
    n_dims = gm.n_dimensions(mixture)
    positions = gm.positions(mixture).view(-1, 2)
    return render.render(mixture, constant,
                         x_low=positions.min(dim=0)[0][0].item(), y_low=positions.min(dim=0)[0][1].item(), x_high=positions.max(dim=0)[0][0].item(), y_high=positions.max(dim=0)[0][1].item(),
                         width=200, height=200)


# c = torch.jit.script(Container(conv_input))
data = torch.jit.load(f"/home/madam/Documents/work/tuw/gmc_net/data/mnist_intermediate_data/conv_inputs_0.pt")

# n_target_g = gm.n_layers(data.conv_layer_1_data) * gm.n_components(data.conv_layer_1_data) * gm.n_components(data.conv_layer_1_kernels)
# gt = cpp_conv(data.conv_layer_1_data, data.conv_layer_1_kernels, n_target_g)
m = cpp_conv(data.conv_layer_2_data.cuda(), data.conv_layer_2_kernels.cuda(), 256)
constant = torch.zeros(1, 1, device=m.device)
# m = m[:5, :, :, :]
# m = torch.cat((m[0:4, :], m[22:23, :]), dim=0).contiguous()
# m = torch.cat((m[:, 0:2], m[:, 6:7], m[:, 9:11]), dim=1).contiguous()
gt = my_render(m)


# matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/input.png', render.imshow(m, torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/gt.png', render.imshow_with_ReLU(m, torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/solver_1.png', render.imshow(fitting.solver(m, constant, -1, solver_n_samples=-1)[0], torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/solver_2.png', render.imshow(fitting.solver(m, constant, -1, solver_n_samples=-2)[0], torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/solver_centres.png', render.imshow(fitting.solver(m, constant, -1, solver_n_samples=0)[0], torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/solver_centres_and_1.png', render.imshow(fitting.solver(m, constant, -1, solver_n_samples=1)[0], torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# # matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/solver_centres_and_2.png', render.imshow(fitting.solver(m, constant, -1, solver_n_samples=2)[0], torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# # matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/solver_centres_and_3.png', render.imshow(fitting.solver(m, constant, -1, solver_n_samples=3)[0], torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# # matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/solver_centres_and_9.png', render.imshow(fitting.solver(m, constant, -1, solver_n_samples=9)[0], torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))
# matplotlib.image.imsave('/home/madam/Documents/work/tuw/gmcn_paper/figures/solver_vs_heuristic/heuristic.png', render.imshow(fitting.fixed_point_only(m, constant, -1)[0], torch.zeros(1, 1, device=m.device), width=200, height=200, clamp=(-32, 32)))

for n in (-1, -2, -4):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fitting.solver(m, constant, -1, solver_n_samples=n)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"solver with {n} took {t1-t0}")

torch.cuda.synchronize()
t0 = time.perf_counter()
fitting.fixed_point_only(m, constant, -1)
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f"heuristic took {t1-t0}")

print("d")