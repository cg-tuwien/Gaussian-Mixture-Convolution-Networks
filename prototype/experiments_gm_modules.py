import random

import torch
import matplotlib.pyplot as plt
import numpy as np

import gm_modules
import gm
import gm_fitting

n_kernel_components = 8

m = gm.generate_random_mixtures(n_layers=3, n_components=4, n_dims=2, pos_radius=10, cov_radius=2.5, weight_min=0)

# todo: fitting and learning fitting doesn't work with the bias and relu layer.
# todo: - maybe the recursion is bad. tried reducing n_kernel_components, but it doesn't run any more..
gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=5, n_kernel_components=n_kernel_components).cuda()
relu1 = gm_modules.GmBiasAndRelu(n_layers=5, n_output_gaussians=10).cuda()
gmc2 = gm_modules.GmConvolution(n_layers_in=5, n_layers_out=3, n_kernel_components=n_kernel_components).cuda()
relu2 = gm_modules.GmBiasAndRelu(n_layers=3, n_output_gaussians=10).cuda()
relu2.net = relu1.net

relu1.train_fitting(True)
relu2.train_fitting(True)
trainer1 = gm_fitting.Trainer(relu1)
trainer2 = gm_fitting.Trainer(relu1)
epoch = 0
for j in range(1000):
    for i in range(599):
        gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=5, n_kernel_components=random.randint(1, n_kernel_components), position_range=4, covariance_range=1).cuda()
        x, l = gm.load(f"mnist/train_{i}")
        x = x.to('cuda')
        x = gmc1(x)
        trainer1.train_on(x, torch.rand_like(relu1.bias) * 0.3, epoch)
        epoch += 1

        # x = x.detach()
        # x = relu1(x)
        # x = gmc2(x)
        #
        # trainer2.train_on(x, torch.rand_like(relu2.bias) * 0.3, epoch)
        # epoch += 1

        if epoch % 100 == 0:
            trainer1.save_weights()
    trainer1.save_weights()
relu1.train_fitting(False)
relu2.train_fitting(False)

m, l = gm.load("mnist/test_0")
m = m[:10]
m = m.to('cuda')

def debug_show(m: torch.Tensor):
    low = -5
    high = 33
    size = 64
    spacing = (high - low) / size

    n_cols = gm.n_layers(m)
    n_rows = gm.n_batch(m)
    canvas = np.zeros((n_rows * size, n_cols * size))
    for r in range(n_rows):
        for c in range(n_cols):
            i = gm.debug_show(m, batch_i=r, layer_i=c, x_low=low, y_low=low, x_high=high, y_high=high, step=spacing, imshow=False)
            canvas[r*size:(r+1)*size, c*size:(c+1)*size] = i
    plt.imshow(canvas,  extent=[low, low + n_cols * (high - low), low, low + n_rows * (high - low)])
    plt.colorbar()
    plt.show()

x = m
print(f"n_layers = {gm.n_layers(x)}, n_components = {gm.n_components(x)}")
debug_show(x)

x = gmc1(x)
print(f"n_layers = {gm.n_layers(x)}, n_components = {gm.n_components(x)}")
debug_show(x)

x = relu1(x)
print(f"n_layers = {gm.n_layers(x)}, n_components = {gm.n_components(x)}")
debug_show(x)

x = gmc2(x)
print(f"n_layers = {gm.n_layers(x)}, n_components = {gm.n_components(x)}")
debug_show(x)

x = relu2(x)
print(f"n_layers = {gm.n_layers(x)}, n_components = {gm.n_components(x)}")
debug_show(x)

