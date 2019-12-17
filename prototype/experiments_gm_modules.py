import matplotlib.pyplot as plt

import gm_modules
import gm

m = gm.generate_random_mixtures(n_layers=3, n_components=4, n_dims=2, pos_radius=10, cov_radius=2.5, weight_min=0)
m, l = gm.Mixture.load("mnist/train_0")

m = gm.cat((m.batch(0), m.batch(1), m.batch(2)), dim=0)

gmc1 = gm_modules.GmConvolution(3, 5, n_kernel_components=6)
relu1 = gm_modules.GmBiasAndRelu(5, 10)
gmc2 = gm_modules.GmConvolution(5, 3, n_kernel_components=6)
relu2 = gm_modules.GmBiasAndRelu(3, 10)

x = m
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
fig, sbs = plt.subplots(1, x.n_layers())
for i, sb in enumerate(sbs.flat):
    im = x.debug_show(i, -5, -5, 33, 33, 0.2, imshow=False)
    im = sb.imshow(im)
fig.colorbar(im, ax=sbs.ravel().tolist())
fig.show()

x = gmc1(x)
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
fig, sbs = plt.subplots(1, x.n_layers())
for i, sb in enumerate(sbs.flat):
    im = x.debug_show(i, -5, -5, 33, 33, 0.2, imshow=False)
    im = sb.imshow(im)
fig.colorbar(im, ax=sbs.ravel().tolist())
fig.show()

x = relu1(x)
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
fig, sbs = plt.subplots(1, x.n_layers())
for i, sb in enumerate(sbs.flat):
    im = x.debug_show(i, -5, -5, 33, 33, 0.2, imshow=False)
    im = sb.imshow(im)
fig.colorbar(im, ax=sbs.ravel().tolist())
fig.show()

x = gmc2(x)
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
fig, sbs = plt.subplots(1, x.n_layers())
for i, sb in enumerate(sbs.flat):
    im = x.debug_show(i, -5, -5, 33, 33, 0.2, imshow=False)
    im = sb.imshow(im)
fig.colorbar(im, ax=sbs.ravel().tolist())
fig.show()

x = relu2(x)
print(f"n_layers = {x.n_layers()}, n_components = {x.n_components()}")
fig, sbs = plt.subplots(1, x.n_layers())
for i, sb in enumerate(sbs.flat):
    im = x.debug_show(i, -5, -5, 33, 33, 0.2, imshow=False)
    im = sb.imshow(im)
fig.colorbar(im, ax=sbs.ravel().tolist())
fig.show()

