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


def em_algorithm(image: Tensor, n_components: int, n_iterations: int, device: torch.device = 'cpu') -> Tensor:
    # todo test (after moving from Mixture class to Tensor data
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    xv, yv = torch.meshgrid([torch.arange(0, width, 1, dtype=torch.float, device=device),
                             torch.arange(0, height, 1, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(-1, 2)

    image_distribution = DiscreteDistribution(image.view(-1))
    point_cloud = xes[image_distribution.sample((10000, )), :]
    point_cloud += torch.rand_like(point_cloud)
    # plt.scatter(point_cloud[:, 0], point_cloud[:, 1], marker='.')
    # plt.show()

    fitting_start = time.time()
    model = sklearn.mixture.GaussianMixture(n_components=n_components,
                                            tol=0.00005,
                                            reg_covar=6.0e-2,
                                            max_iter=n_iterations,
                                            init_params='kmeans',  # 'kmeans' or 'random'
                                            verbose=2,
                                            )
    model.fit(point_cloud.numpy())
    positions = torch.tensor(model.means_, dtype=torch.float32).view(1, 1, n_components, 2)
    covariances = torch.tensor(model.covariances_, dtype=torch.float32).view(1, 1, n_components, 2, 2)
    weights = torch.tensor(model.weights_, dtype=torch.float32).view(1, 1, n_components) * gm.normal_amplitudes(covariances)
    return gm.pack_mixture(weights, positions, covariances)

    # model = pomegranate.gmm.GeneralMixtureModel.from_samples(
    #     pomegranate.MultivariateGaussianDistribution,  # Either single function, or list of functions
    #     n_components=n_components,  # Required if single function passed as first arg
    #     X=point_cloud.numpy(),  # data format: each row is a point-coordinate, each column is a dimension
    #     stop_threshold=.001,  # Lower this value to get better fit but take longer.
    #     # n_init=10,
    #     max_iterations=n_iterations,
    #     # init='kmeans++',
    #     inertia=0.9
    # )
    # model_dict = model.to_dict()
    #
    # weights = torch.tensor(model_dict['weights'])
    # positions = list()
    # covariances = list()
    # for distribution in model_dict['distributions']:
    #     positions.append(torch.tensor(distribution['parameters'][0]).view(1, 1, 1, 2))
    #     covariances.append(torch.tensor(distribution['parameters'][1]).view(1, 1, 1, 2, 2))
    #
    # positions = torch.cat(positions, dim=2)
    # covariances = torch.cat(covariances, dim=2)
    # weights = weights.view(1, 1, n_components) * gm.normal_amplitudes(covariances)
    # return gm.pack_mixture(weights, positions, covariances)




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


def run_on_one():
    image = plt.imread("/home/madam/Downloads/mnist_png/training/5/35.png")
    # image = plt.imread("/home/madam/Downloads/mnist_png/training/3/7.png") * 255
    if len(image.shape) == 3:
        image = image.mean(axis=2)
    image_width = image.shape[1]
    image_height = image.shape[0]
    image = torch.tensor(image, dtype=torch.float32).transpose(0, 1).contiguous()
    m1 = em_algorithm(image, n_components=64, n_iterations=100, device='cpu')
    inout.save(m1, "auto")

    m1 = m1.cpu()

    rendering = debug_render(m1, orig_size=(image_width, image_height), clamp=(0, 0.02))
    plt.imshow(rendering)
    plt.show()
    # plt.imsave("/home/madam/cloud/celarek/Photos/em_test.png", rendering)

run_on_one()

def run_on_mnist():
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
        gms = em_algorithm(images, n_components=n_components, n_iterations=121, device='cuda')
        # gm.debug_show(gms, x_low=0, y_low=0, x_high=28, y_high=28, step=28 / 200)
        for j in range(0, batch_size):
            inout.save(gms[j].view(1, 1, n_components, -1), f"mnist_em/test_{i*batch_size + j}", local_labels[j])
        print(f"mnist_em/test_{i}")

    mnist_training = torchvision.datasets.MNIST("/home/madam/temp/mnist/", train=True, transform=torchvision.transforms.ToTensor(),
                                                target_transform=None, download=True)
    data_generator = torch.utils.data.DataLoader(mnist_training,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=16)
    for i, (local_batch, local_labels) in enumerate(data_generator):
        assert local_batch.size()[1] == 1
        images = local_batch.view(batch_size, height, width)
        gms = em_algorithm(images, n_components=n_components, n_iterations=121, device='cuda')
        # gms = ad_algorithm(images, n_components=n_components, n_iterations=121, device='cuda')
        for j in range(0, batch_size):
            inout.save(gms[j].view(1, 1, n_components, -1), f"mnist_em/train_{i*batch_size + j}", local_labels[j])
        print(f"mnist_em/train_{i}")

        # for i in range(20):
        #     fit = gms.debug_show(i, 0, 0, width, height, width / 200, imshow=False)
        #     target = local_batch[i, 0, :, :]
        #     fig, (ax1, ax2) = plt.subplots(1, 2)
        #     fig.suptitle(f"{local_labels[i]}")
        #     ax1.imshow(fit)
        #     ax2.imshow(target)
        #     plt.show()


# test_mnist()
