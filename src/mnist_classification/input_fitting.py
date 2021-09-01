import typing

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import sklearn.mixture
from sklearn.utils._testing import ignore_warnings
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical as DiscreteDistribution
import torch.utils.data
from torchvision import datasets, transforms

import gmc.mixture as gm
import gmc.inout as inout
import gmc.render as render
from gmc.modules import BatchNorm

from mnist_classification.config import Config


def debug_render(mixture: Tensor, orig_size: typing.Tuple[int, int] = (28, 28), image_size: typing.Tuple[int, int] = (200, 200), clamp: typing.Tuple[float, float] = (0, 1.5)):
    mixture = mixture.view(1, gm.n_batch(mixture), gm.n_components(mixture), -1)
    images = render.render(mixture, torch.zeros(1, 1), x_low=0, x_high=orig_size[0], y_low=0, y_high=orig_size[1], width=image_size[0], height=image_size[1])
    images = render.colour_mapped(images.cpu().numpy(), clamp[0], clamp[1])
    return images[:, :, :3]


def em_algorithm(image: Tensor, n_components: int, n_iterations: int) -> Tensor:
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    xv, yv = torch.meshgrid([torch.arange(0, width, 1, dtype=torch.float),
                             torch.arange(0, height, 1, dtype=torch.float)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(-1, 2)

    image_distribution = DiscreteDistribution(image.view(-1))
    point_cloud = xes[image_distribution.sample((4000, )), :]
    point_cloud += torch.rand_like(point_cloud)
    # plt.scatter(point_cloud[:, 0], point_cloud[:, 1], marker='.')
    # plt.show()

    model = sklearn.mixture.GaussianMixture(n_components=n_components,
                                            tol=0.00005,
                                            reg_covar=0.1,
                                            max_iter=n_iterations,
                                            init_params='kmeans',  # 'kmeans' or 'random'
                                            verbose=0,
                                            )
    model.fit(point_cloud.numpy())
    positions = torch.tensor(model.means_, dtype=torch.float32).view(1, 1, n_components, 2)
    covariances = torch.tensor(model.covariances_, dtype=torch.float32).view(1, 1, n_components, 2, 2)
    weights = torch.tensor(model.weights_, dtype=torch.float32).view(1, 1, n_components) * gm.normal_amplitudes(covariances)
    return gm.pack_mixture(weights, positions, covariances)


@ignore_warnings(category=ConvergenceWarning)
def run_and_store(config: Config, data_loader: torch.utils.data.DataLoader, dataset_name: str, start: int, end: int):
    norm = BatchNorm(n_layers=1, batch_norm=False, learn_scaling=False)
    for batch_idx, (data, target) in enumerate(data_loader):
        if batch_idx < start or batch_idx >= end:
            continue
        filename = f"{config.produce_input_description()}/{dataset_name}_{batch_idx}"
        try:
            inout.load(filename)
            # print(f"{filename} already processed")
        except OSError as e:
            mixture = em_algorithm(data.view(28, 28).transpose(0, 1).contiguous(), n_components=config.input_fitting_components, n_iterations=config.input_fitting_iterations)
            mixture, _ = norm((mixture, None))
            inout.save(mixture.view(1, 1, config.input_fitting_components, -1), filename, target[0])
            if batch_idx < 20:
                png_file = config.data_base_path / config.produce_input_description() / "png" / f"{dataset_name}_{batch_idx}.png"
                rendering = debug_render(mixture, orig_size=(28, 28), clamp=(0, 5.0))
                plt.imsave(png_file, rendering)
            # print(filename)


def fit(config: Config):
    if config.input_fitting_iterations == -1:
        return
    (config.data_base_path / config.produce_input_description()).mkdir(parents=True, exist_ok=True)
    (config.data_base_path / config.produce_input_description() / 'png').mkdir(parents=True, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        config.dataset_class(config.data_base_path / f"{config.dataset_name}_raw", train=True, download=True, transform=transforms.ToTensor()),
        batch_size=1)
    test_loader = torch.utils.data.DataLoader(
        config.dataset_class(config.data_base_path / f"{config.dataset_name}_raw", train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1)

    run_and_store(config, train_loader, "train", start=config.training_set_start, end=config.training_set_end)
    run_and_store(config, test_loader, "test", start=config.test_set_start, end=config.test_set_end)

