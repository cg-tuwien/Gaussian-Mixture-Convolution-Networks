import matplotlib.pyplot as plt
import numpy as np

import gm_modules
import gm

m = gm.generate_random_mixtures(n_layers=3, n_components=4, n_dims=2, pos_radius=10, cov_radius=2.5, weight_min=0)
m, l = gm.Mixture.load("mnist/train_0")
m = m.to('cuda')

gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=5, n_kernel_components=6).cuda()
relu1 = gm_modules.GmBiasAndRelu(n_layers=5, n_output_gaussians=10).cuda()
gmc2 = gm_modules.GmConvolution(n_layers_in=5, n_layers_out=3, n_kernel_components=6).cuda()
relu2 = gm_modules.GmBiasAndRelu(n_layers=3, n_output_gaussians=10).cuda()


def debug_show(m: gm.Mixture):
    low = -5
    high = 33
    size = 64
    spacing = (high - low) / size

    n_cols = m.n_layers()
    n_rows = m.n_batch()
    canvas = np.zeros((n_rows * size, n_cols * size))
    for r in range(n_rows):
        for c in range(n_cols):
            i = m.debug_show(batch_i=r, layer_i=c, x_low=low, y_low=low, x_high=high, y_high=high, step=spacing, imshow=False)
            canvas[r*size:(r+1)*size, c*size:(c+1)*size] = i
    plt.imshow(canvas,  extent=[low, low + n_cols * (high - low), low, low + n_rows * (high - low)],origin='lower')
    plt.colorbar()
    plt.show()

x = m
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
# debug_show(x)

x = gmc1(x)
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
# debug_show(x)

x = relu1(x)
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
# debug_show(x)

x = gmc2(x)
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
# debug_show(x)

x = relu2(x)
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
# debug_show(x)

