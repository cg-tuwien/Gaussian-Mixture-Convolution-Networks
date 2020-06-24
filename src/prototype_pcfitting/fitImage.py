import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms
import torch.utils.data
import typing

import gmc.mixture as gm
import gmc.mat_tools as mat_tools
import gmc.image_tools as madam_imagetools

import config

from torch import Tensor


def debug_render(mixture: Tensor, orig_size: typing.Tuple[int, int] = (28, 28), image_size: typing.Tuple[int, int] = (200, 200), clamp: typing.Tuple[float, float] = (0, 1.5)):
    mixture = mixture.view(1, gm.n_batch(mixture), gm.n_components(mixture), -1)
    images = gm.render(mixture, x_low=0, x_high=orig_size[0], y_low=0, y_high=orig_size[1], width=image_size[0], height=image_size[1])
    images = madam_imagetools.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
    return images[:, :, :3]

def em_algorithm(image: Tensor, n_components: int, n_iterations: int, device: torch.device = 'cpu') -> Tensor:
    # todo test (after moving from Mixture class to Tensor data
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
        selected_components = mixture.max_component(xes)
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

def ad_algorithm(image: Tensor, n_components: int, n_iterations: int = 8, device: torch.device = 'cpu') -> Tensor:
    assert len(image.shape) == 3
    assert n_components > 0
    batch_size = image.size()[0]
    width = image.size()[2]
    height = image.size()[1]

    target = image.to(device)
    target = target / torch.max(target)

    if width == 28 and height == 28:
        # mnist
        yv, xv = torch.meshgrid([torch.arange(0, height, 1, dtype=torch.float32, device=device),
                                 torch.arange(0, width, 1, dtype=torch.float32, device=device)])
        xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, -1, 2).repeat((batch_size, 1, 1))
        xes += 0.5
        xes += torch.rand_like(xes) - 0.5
        randperm = torch.randperm(width * height, device=device)  # torch.sort might be stable, but we want to be random. I don't think it matters that all images are permuted the same way.
        xes = xes[:, randperm, :]
        values = target.view(batch_size, -1)[:, randperm]
        # quantise with reduced resolution
        values = (values + 0.5).int()
        indices = torch.argsort(values, dim=1, descending=True)
        indices = indices[:, 0:n_components]

        weights = torch.rand(batch_size, 1, n_components, dtype=torch.float32, device=device) * 0.1 + 0.9
        positions = mat_tools.batched_index_select(xes, 1, indices).view(batch_size, 1, -1, 2)
        covariances = (torch.tensor([[32 / height, 0], [0, 32 / width]], dtype=torch.float32, device=device)).view(1, 1, 1, 2, 2).expand((batch_size, 1, n_components, 2, 2))
        mixture = gm.pack_mixture(weights, positions, covariances)
    else:
        # the mnist initialisation might not be ideal for larger or more complex images.
        # todo: implement batching for em
        # assert False
        # mixture = em_algorithm(image * 255, n_components, 5, device='cpu')
        # if device == 'cuda':
        #     mixture = mixture.cuda()
        # mixture.weights /= 255.0
        mixture = gm.generate_random_mixtures(n_layers=batch_size, n_components=n_components, n_dims=2,
                                              pos_radius=0.5, cov_radius=5 * min(width, height) / math.sqrt(n_components),
                                              weight_min=0, weight_max=1, device=device)
        positions = gm.positions(mixture)
        positions += 0.5
        positions *= torch.tensor([[[width, height]]], dtype=torch.float, device=device)

    # gm.debug_show(mixture, 0, 0, 0, 0, width, height, min(width, height) / 100)

    fitting_start = time.time()

    weights = gm.weights(mixture)
    positions = gm.positions(mixture)
    weights.requires_grad = True;
    positions.requires_grad = True;

    inversed_covariances = gm.covariances(mixture).inverse()
    (eigvals, eigvecs) = torch.symeig(inversed_covariances, eigenvectors=True)
    eigvals = torch.max(eigvals, torch.tensor([0.01], dtype=torch.float, device=device))
    icov_factor = torch.matmul(eigvecs, eigvals.sqrt().diag_embed())
    icov_factor.requires_grad = True

    optimiser = optim.Adam([weights, positions, icov_factor], lr=0.005)

    # print("starting gradient descent")
    for k in range(n_iterations):
        if k == n_iterations / 2:
            optimiser = optim.Adam([weights, positions, icov_factor], lr=0.0005)
        optimiser.zero_grad()

        xes = torch.rand(batch_size, 1, config.eval_img_n_sample_points, 2, device=device, dtype=torch.float32) * torch.tensor([[[width, height]]], device=device, dtype=torch.float32)
        # indices are row * column, while position is x * y, therefore we flip
        xes_indices = xes.type(torch.long).view(batch_size, -1, 2).flip(dims=[-1])
        xes_indices_b = xes_indices
        xes_indices = mat_tools.flatten_index(xes_indices, target.shape[-2:])

        inversed_covariances = icov_factor @ icov_factor.transpose(-2, -1) + torch.eye(2, 2, device=mixture.device) * 0.001
        assert not torch.isnan(inversed_covariances).any()
        assert not torch.isinf(inversed_covariances).any()
        mixture_with_inversed_cov = gm.pack_mixture(weights, positions, inversed_covariances)
        output = gm.evaluate_inversed(mixture_with_inversed_cov, xes).view(batch_size, -1)
        target_samples = mat_tools.batched_index_select(target.view(batch_size, -1), dim=1, index=xes_indices)
        loss = torch.mean(torch.abs(output - target_samples))

        loss.backward()
        optimiser.step()

        if k % 40 == 0:
            print(f"iterations {k}: loss = {loss.item()}, min det = {torch.min(torch.det(inversed_covariances.detach()))}")
            _weights = weights.detach()
            _positions = positions.detach()
            # torch inverse returns a transposed matrix (v 1.3.1). our matrix is symmetric however, and we want to take a view, so the transpose avoids a copy.
            _covariances = inversed_covariances.detach().inverse().transpose(-1, -2)
            mixture = gm.pack_mixture(_weights, _positions, _covariances)
            # gm.debug_show(mixture, x_low=0, y_low=0, x_high=width, y_high=height, step=min(width, height) / 256)

            # rendering = debug_render(mixture)
            # plt.imsave("/home/madam/temp/prototype/rendering.png", rendering)
            # input("Press enter to continue!")

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    weights = weights.detach()
    positions = positions.detach()
    # torch inverse returns a transposed matrix (v 1.3.1). our matrix is symmetric however, and we want to take a view, so the transpose avoids a copy.
    covariances = inversed_covariances.detach().inverse().transpose(-1, -2)
    final_mixture = gm.pack_mixture(weights, positions, covariances)
    # rendering = debug_render(final_mixture)
    # plt.imsave("/home/madam/temp/prototype/rendering.png", rendering)
    return final_mixture


def test():
    image = plt.imread("/home/madam/cloud/celarek/Photos/auto.jpg")
    # image = plt.imread("/home/madam/Downloads/mnist_png/training/3/7.png") * 255
    if len(image.shape) == 3:
        image = image.mean(axis=2)
    image_width = image.shape[1]
    image_height = image.shape[0]
    image = torch.tensor(image, dtype=torch.float32).view(1, image_height, image_width)
    # m1 = em_algorithm(torch.tensor(image, dtype=torch.float32), n_components=2500, n_iterations=5, device='cpu')
    m1 = ad_algorithm(image, n_components=2000, n_iterations=400000, device='cuda')
    gm.save(m1, "auto")

    m1 = m1.cpu()

    rendering = debug_render(m1, (image_width, image_height), (1920, 1080))
    plt.imsave("/home/madam/cloud/celarek/Photos/auto_rendering.png", rendering)

# todo test_mnist() doesn't work but test() does. WHY?
def test_mnist():
    batch_size = 50
    width = 28
    height = 28
    n_components = 25
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('../data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),

    mnist_test = torchvision.datasets.MNIST("/home/madam/temp/mnist/", train=False, transform=torchvision.transforms.ToTensor(),
                                            target_transform=None, download=True)
    data_generator = torch.utils.data.DataLoader(mnist_test,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=0)
    for i, (local_batch, local_labels) in enumerate(data_generator):
        assert local_batch.shape[1] == 1
        images = local_batch.view(batch_size, height, width)
        gms = ad_algorithm(images, n_components=n_components, n_iterations=121, device='cuda')
        # gm.debug_show(gms, x_low=0, y_low=0, x_high=28, y_high=28, step=28 / 200)
        for j in range(0, batch_size):
            gm.save(gms[j].view(1, 1, n_components, -1), f"mnist/test_{i*batch_size + j}", local_labels[j])
        print(f"mnist/test_{i}")

    mnist_training = torchvision.datasets.MNIST("/home/madam/temp/mnist/", train=True, transform=torchvision.transforms.ToTensor(),
                                                target_transform=None, download=True)
    data_generator = torch.utils.data.DataLoader(mnist_training,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=16)
    for i, (local_batch, local_labels) in enumerate(data_generator):
        assert local_batch.size()[1] == 1
        images = local_batch.view(batch_size, height, width)
        gms = ad_algorithm(images, n_components=n_components, n_iterations=121, device='cuda')
        for j in range(0, batch_size):
            gm.save(gms[j].view(1, 1, n_components, -1), f"mnist/train_{i*batch_size + j}", local_labels[j])
        print(f"mnist/train_{i}")

        # for i in range(20):
        #     fit = gms.debug_show(i, 0, 0, width, height, width / 200, imshow=False)
        #     target = local_batch[i, 0, :, :]
        #     fig, (ax1, ax2) = plt.subplots(1, 2)
        #     fig.suptitle(f"{local_labels[i]}")
        #     ax1.imshow(fit)
        #     ax2.imshow(target)
        #     plt.show()


test_mnist()
