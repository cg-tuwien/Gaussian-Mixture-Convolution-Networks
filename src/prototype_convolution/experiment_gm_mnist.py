from __future__ import print_function
import argparse
import datetime
import random
import time
import math
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import typing
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import prototype_convolution.config as config
import gmc.mixture as gm
# import fitting_net
import prototype_convolution.gm_modules as gm_modules
import prototype_convolution.experiment_gm_mnist_model as experiment_gm_mnist_model

# based on https://github.com/pytorch/examples/blob/master/mnist/main.py
import gmc.image_tools as madam_imagetools

n_kernel_components = 5


# torch.autograd.set_detect_anomaly(True)


def sum_losses(losses: typing.List[torch.Tensor]):
    loss_sum = None
    for loss in losses:
        if loss_sum is None:
            loss_sum = loss;
        else:
            loss_sum = loss_sum + loss
    return loss_sum


class GmMnistDataSet(torch.utils.data.Dataset):
    def __init__(self, prefix: str, begin: int, end: int):
        self.prefix = prefix
        self.begin = begin
        self.end = end

    def __len__(self):
        return self.end - self.begin

    def __getitem__(self, index):
        mixture, meta = gm.load(f"{self.prefix}{index}")
        return mixture[0], meta


def render_debug_images_to_tensorboard(model, epoch, tensor_board_writer):
    tensor_board_writer.add_image("mnist conv 1", model.gmc1.debug_render(clamp=[-0.80, 0.80]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist conv 2", model.gmc2.debug_render(clamp=[-0.32, 0.32]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist conv 3", model.gmc3.debug_render(clamp=[-0.20, 0.20]), epoch, dataformats='HWC')

    tensor_board_writer.add_image("mnist relu 1", model.relus[0].debug_render(position_range=[-14, -14, 42, 42], clamp=[-4 / (28 ** 2), 16.0 / (28 ** 2)]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist relu 2", model.relus[1].debug_render(position_range=[-14, -14, 42, 42], clamp=[-20 / (28 ** 2), 80.0 / (28 ** 2)]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist relu 3", model.relus[2].debug_render(position_range=[-14, -14, 42, 42], clamp=[-6 / (28 ** 2), 24.0 / (28 ** 2)]), epoch, dataformats='HWC')


def train(args, model: experiment_gm_mnist_model.Net, device: torch.device, train_loader: torch.utils.data.DataLoader,
          kernel_optimiser: optim.Optimizer, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter = None):

    model.train()
    start_time = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        step = epoch * len(train_loader.dataset) + batch_idx * len(data)

        batch_start_time = time.perf_counter()
        output = model(data)
        loss = F.nll_loss(output, target)
        # regularisation_loss = model.regularisation_loss() * len(data)
        training_loss = loss  # (loss + regularisation_loss)

        kernel_optimiser.zero_grad()
        training_loss.backward()
        kernel_optimiser.step()
        batch_end_time = time.perf_counter()

        if step % args.log_interval == 0:
            pred = output.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            tensor_board_writer.add_scalar("00. mnist training loss", training_loss.item(), step)
            tensor_board_writer.add_scalar("01. mnist training accuracy", 100 * correct / len(data), step)
            tensor_board_writer.add_scalar("02. mnist kernel loss", loss.item(), step)
            tensor_board_writer.add_scalar("03. duration per sample", (batch_end_time - batch_start_time) / len(data), step)
            # tensor_board_writer.add_scalar("04. mnist training regularisation loss", regularisation_loss.item(), step)
            tensor_board_writer.add_scalar("05. model layer 1 max(bias)", torch.nn.functional.softplus(model.relus[0].bias, beta=20).max().item(), step)
            tensor_board_writer.add_scalar("05. model layer 2 max(bias)", torch.nn.functional.softplus(model.relus[1].bias, beta=20).max().item(), step)
            tensor_board_writer.add_scalar("05. model layer 3 max(bias)", torch.nn.functional.softplus(model.relus[2].bias, beta=20).max().item(), step)
            render_debug_images_to_tensorboard(model, step, tensor_board_writer)

            print(f'Training kernels: {epoch}/{step} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tClassification loss: {loss.item():.6f} (accuracy: {100 * correct / len(data)})')

            if args.save_model:
                model.save_model()
            # print(f"experiment_gm_mnist.tain: saving optimiser state to {model.storage_path}.optimiser")
            # torch.save(kernel_optimiser.state_dict(), f"{model.storage_path}.optimiser")
    end_time = time.perf_counter()

    tensor_board_writer.add_scalar("10. batch_duration", end_time - start_time, step)

def test(args, model: experiment_gm_mnist_model.Net, device: torch.device, test_loader: torch.utils.data.DataLoader, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter):
    model.eval()
    test_loss = 0
    correct = 0
    fitting_loss_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            fitting_inputs = list()
            data, target = data.to(device), target.to(device)
            output = model(data, fitting_inputs=fitting_inputs)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # fitting_losses = model.run_fitting_sampling(fitting_inputs, train=False, epoch=epoch, tensor_board_writer=tensor_board_writer, tensor_board_prefix="test_")
            # fitting_loss_sum += sum_losses(fitting_losses).item()

    test_loss /= len(test_loader.dataset)
    # fitting_loss_sum /= len(test_loader.dataset)
    tensor_board_writer.add_scalar("99. mnist test loss", test_loss, epoch)
    tensor_board_writer.add_scalar("98. mnist test accuracy", 100. * correct / len(test_loader.dataset), epoch)
    # tensor_board_writer.add_scalar("97. mnist test fitting loss", fitting_loss_sum, epoch)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def experiment(device: str = 'cuda', n_epochs: int = 20, kernel_learning_rate: float = 0.001, log_interval: int = 100,
                           learn_positions_after: int = 0,
                           learn_covariances_after: int = 0,
                           desc_string: str = ""):
    # Training settings
    torch.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/train_', begin=0, end=60000), batch_size=config.batch_size, num_workers=config.num_dataloader_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/test_', begin=0, end=10000), batch_size=100, num_workers=config.num_dataloader_workers)

    model = experiment_gm_mnist_model.Net(name=desc_string,
                                          learn_positions=learn_positions_after == 0,
                                          learn_covariances=learn_covariances_after == 0,
                                          n_kernel_components=n_kernel_components)
    model.load()
    model = model.to(device)

    class Args:
        pass

    args = Args()
    args.save_model = True
    args.log_interval = log_interval
    args.save_model = False

    kernel_optimiser = optim.SGD(model.parameters(), lr=kernel_learning_rate)

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'sgd_b100_{desc_string}_{datetime.datetime.now().strftime("%m%d_%H%M")}')
    # scheduler = StepLR(kernel_optimiser, step_size=1, gamma=args.gamma)


    for epoch in range(n_epochs):
        model.set_position_learning(epoch >= learn_positions_after)
        model.set_covariance_learning(epoch >= learn_covariances_after)
        train(args, model, device, train_loader, kernel_optimiser=kernel_optimiser, epoch=epoch, tensor_board_writer=tensor_board_writer)

        test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model:
            model.save_model()
            model.save_fitting_optimiser_state()


def main():
    default_learning_rate = 0.01
    default_epochs = 6 * 10
    default_log_interval = 20
    train_fitting_layers = True
    train_mnist = False
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=default_epochs, metavar='N',
                        help=f'number of epochs to train (default: {default_epochs})')
    parser.add_argument('--lr', type=float, default=default_learning_rate, metavar='LR',
                        help=f'learning rate (default: {default_learning_rate})')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=default_log_interval, metavar='N',
                        help=f'how many batches to wait before logging training status (default: {default_log_interval}')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/train_', begin=0, end=600), batch_size=None, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/test_', begin=0, end=100), batch_size=None, collate_fn=lambda x: x)

    model = experiment_gm_mnist_model.Net(n_kernel_components=n_kernel_components)
    model.load()
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'gm_mnist_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    # scheduler = StepLR(kernel_optimiser, step_size=1, gamma=args.gamma)
    for epoch in range(0, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch, tensor_board_writer=tensor_board_writer)
        test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model and train_mnist:
            model.save_model()

        if args.save_model and train_fitting_layers is not None:
            model.save_fitting_parameters()
            model.save_fitting_optimiser_state()


if __name__ == '__main__':
    main()
