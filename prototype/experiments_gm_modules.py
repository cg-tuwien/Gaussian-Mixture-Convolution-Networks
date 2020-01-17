import torch
import matplotlib.pyplot as plt
import numpy as np

import gm_modules
import gm
import gm_fitting

n_kernel_components = 5
n_layers_1 = 5
n_layers_2 = 6

m = gm.generate_random_mixtures(n_layers=3, n_components=4, n_dims=2, pos_radius=10, cov_radius=2.5, weight_min=0)

# todo: fitting and learning fitting doesn't work with the bias and relu layer.
# todo: - maybe the recursion is bad. tried reducing n_kernel_components, but it doesn't run any more..
# gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=5, n_kernel_components=n_kernel_components).cuda()
relu1 = gm_modules.GmBiasAndRelu(n_layers=n_layers_1, n_output_gaussians=25).cuda()
# gmc2 = gm_modules.GmConvolution(n_layers_in=5, n_layers_out=3, n_kernel_components=n_kernel_components).cuda()
relu2 = gm_modules.GmBiasAndRelu(n_layers=n_layers_2, n_output_gaussians=12).cuda()
relu3 = gm_modules.GmBiasAndRelu(n_layers=10, n_output_gaussians=5).cuda()
# relu2.net = relu1.net
# relu3.net = relu1.net

relu1.train_fitting(True)
relu2.train_fitting(True)
relu3.train_fitting(True)
# trainer1 = gm_fitting.Trainer(relu1, n_training_samples=400)
# trainer2 = gm_fitting.Trainer(relu2, n_training_samples=400)
trainer3 = gm_fitting.Trainer(relu3, n_training_samples=400)
epoch = 0
for j in range(1000):
    for i in range(599):
        x_all, l = gm.load(f"train_{i}")
        x_all = x_all.to('cuda')
        for k in range(4):
            x = x_all[k*25:(k+1)*25]

            # gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=n_layers_1, n_kernel_components=n_kernel_components, position_range=4, covariance_range=1).cuda()
            gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=n_layers_1, n_kernel_components=n_kernel_components,
                                            position_range=2, covariance_range=0.5,
                                            learn_positions=False, learn_covariances=False,
                                            weight_sd=0.00015).cuda()
            x = gmc1(x)
            # x = gm.generate_random_mixtures(n_batch=100, n_layers=n_layers_1, n_components=25, n_dims=2, pos_radius=14, cov_radius=5, weight_min=-0.5, weight_max=1).cuda()
            # trainer1.train_on(x, torch.rand_like(relu1.bias) * 0.00015, epoch)
            # epoch += 1

            gmc1 = None
            x = relu1(x)
            x = x.detach()
            #
            #
            gmc2 = gm_modules.GmConvolution(n_layers_in=n_layers_1, n_layers_out=n_layers_2, n_kernel_components=n_kernel_components,
                                            position_range=4, covariance_range=2,
                                            learn_positions=False, learn_covariances=False,
                                            weight_sd=0.002).cuda()
            x = gmc2(x)
            #
            # # x = gm.generate_random_mixtures(n_batch=100, n_layers=n_layers_2, n_components=12, n_dims=2, pos_radius=14, cov_radius=5, weight_min=-0.5, weight_max=1).cuda()
            # trainer2.train_on(x, torch.rand_like(relu2.bias) * 0.000, epoch)
            # gmc2 = None
            x = relu2(x)
            x = x.detach()
            #
            #
            gmc3 = gm_modules.GmConvolution(n_layers_in=n_layers_2, n_layers_out=10, n_kernel_components=n_kernel_components,
                                                 position_range=8, covariance_range=4,
                                                 learn_positions=False, learn_covariances=False,
                                                 weight_sd=0.001).cuda()
            x = gmc3(x)
            x = x.detach()

            # x = gm.generate_random_mixtures(n_batch=100, n_layers=10, n_components=5, n_dims=2, pos_radius=14, cov_radius=5, weight_min=-0.5, weight_max=1).cuda()
            trainer3.train_on(x, torch.rand_like(relu3.bias) * 0.00, epoch)
            gmc3 = None
#
            if epoch % 10 == 0:
                # trainer1.save_weights()
                # trainer2.save_weights()
                trainer3.save_weights()

            epoch += 1
    # trainer1.save_weights()
    # trainer2.save_weights()
    trainer3.save_weights()
relu1.train_fitting(False)
relu2.train_fitting(False)
relu3.train_fitting(False)

gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=5, n_kernel_components=n_kernel_components).cuda()
gmc2 = gm_modules.GmConvolution(n_layers_in=5, n_layers_out=3, n_kernel_components=n_kernel_components).cuda()

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

