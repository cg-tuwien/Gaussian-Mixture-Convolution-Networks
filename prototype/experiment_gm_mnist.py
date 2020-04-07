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

import config
import gm
import gm_fitting
import gm_modules
import experiment_gm_mnist_model

# based on https://github.com/pytorch/examples/blob/master/mnist/main.py
import madam_imagetools

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
          kernel_optimiser: optim.Optimizer, fitting_optimiser: optim.Optimizer,
          epoch: int, train_kernels: bool = None, train_fitting_layers: bool = None, probDta=None,
          tensor_board_writer: torch.utils.tensorboard.SummaryWriter = None):
    if probDta is not None:
        assert train_fitting_layers is None
        assert train_kernels is None

    cummulative_fitting_losses = list()
    for i in range(3):
        cummulative_fitting_losses.append(0)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        step = epoch * len(train_loader.dataset) + batch_idx * len(data)
        tensorboard_writer_option = None
        if step % args.log_interval == 0:
            tensorboard_writer_option = tensor_board_writer

        if probDta is not None:
            kernel_training_probability = 1
            for i, averaged_fitting_loss in enumerate(probDta.averaged_fitting_losses):
                best_fitting_loss = probDta.best_fitting_losses[i]
                assert best_fitting_loss <= averaged_fitting_loss
                probability = best_fitting_loss / averaged_fitting_loss
                kernel_training_probability = min(kernel_training_probability, probability)
                tensor_board_writer.add_scalar(f"6.2. probDta.best_fitting_losses[{i}]", probDta.best_fitting_losses[i], step)
                tensor_board_writer.add_scalar(f"6.2. probDta.averaged_fitting_losses[{i}]", probDta.averaged_fitting_losses[i], step)

            kernel_training_probability = min(0.9, max(0.1, kernel_training_probability))
            train_kernels = random.uniform(0, 1) < kernel_training_probability
            tensor_board_writer.add_scalar("6.1. kernel_train_probability", kernel_training_probability, step)
            tensor_board_writer.add_scalar("6.2. learning_kernel", train_kernels, step)

            model.set_fitting_training(not train_kernels)
            train_fitting_layers = not train_kernels

        if train_fitting_layers:  # save some computatinos and memory
            fitting_inputs = list()
            model.forward(data, fitting_inputs=fitting_inputs)
            fitting_losses = model.run_fitting_sampling(fitting_inputs, train=True, epoch=step, tensor_board_writer=tensorboard_writer_option, tensor_board_prefix="train_")
            # fitting_optimiser.zero_grad()
            # backward_start_time = time.perf_counter()
            # fitting_loss.backward()
            # backward_time = time.perf_counter() - backward_start_time
            # fitting_optimiser.step()
            for i, loss in enumerate(fitting_losses):
                cummulative_fitting_losses[i] += loss.item()

            if probDta is not None:
                for i, loss in enumerate(fitting_losses):
                    probDta.averaged_fitting_losses[i] += (loss.item() - probDta.averaged_fitting_losses[i]) * 0.0005 * len(data)
                    probDta.best_fitting_losses[i] = min(probDta.best_fitting_losses[i], probDta.averaged_fitting_losses[i])

            if step % args.log_interval == 0:
                fitting_loss_sum = sum_losses(fitting_losses).item()
                tensor_board_writer.add_scalar("4. mnist fitting loss sum", fitting_loss_sum, step)
                # tensor_board_writer.add_scalar(f"train_fitting 3. backward_time_all", backward_time, epoch)

                print(f'Training fitting: {epoch}/{step} [{batch_idx}/{len(train_loader)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tMSE loss: {fitting_loss_sum:.6f}')

                if args.save_model:
                    model.save_fitting_parameters()
                    model.save_fitting_optimiser_state()

        if train_kernels:
            output = model(data)
            loss = F.nll_loss(output, target)
            regularisation_loss = model.regularisation_loss()
            training_loss = (loss + regularisation_loss)

            kernel_optimiser.zero_grad()
            training_loss.backward()
            kernel_optimiser.step()

            if step % args.log_interval == 0:
                pred = output.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                tensor_board_writer.add_scalar("0. mnist training loss", training_loss.item(), step)
                tensor_board_writer.add_scalar("1. mnist training accuracy", 100 * correct / len(data), step)
                tensor_board_writer.add_scalar("2. mnist kernel loss", loss.item(), step)
                tensor_board_writer.add_scalar("3. mnist training regularisation loss", regularisation_loss.item(), step)
                tensor_board_writer.add_scalar("5. model layer 1 avg(abs(bias))", model.relus[0].bias.abs().mean().item(), step)
                tensor_board_writer.add_scalar("5. model layer 2 avg(abs(bias))", model.relus[1].bias.abs().mean().item(), step)
                tensor_board_writer.add_scalar("5. model layer 3 avg(abs(bias))", model.relus[2].bias.abs().mean().item(), step)
                render_debug_images_to_tensorboard(model, step, tensor_board_writer)

                print(f'Training kernels: {epoch}/{step} [{batch_idx}/{len(train_loader)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tClassification loss: {loss.item():.6f} (accuracy: {100 * correct / len(data)})')

                if args.save_model:
                    model.save_model()
                # print(f"experiment_gm_mnist.tain: saving optimiser state to {model.storage_path}.optimiser")
                # torch.save(kernel_optimiser.state_dict(), f"{model.storage_path}.optimiser")

    if probDta is None:
        for loss in cummulative_fitting_losses:
            loss = loss / (len(train_loader.dataset))
    return cummulative_fitting_losses


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
            fitting_losses = model.run_fitting_sampling(fitting_inputs, train=False, epoch=epoch, tensor_board_writer=tensor_board_writer, tensor_board_prefix="test_")
            fitting_loss_sum += sum_losses(fitting_losses).item()

    test_loss /= len(test_loader.dataset)
    fitting_loss_sum /= len(test_loader.dataset)
    tensor_board_writer.add_scalar("99. mnist test loss", test_loss, epoch)
    tensor_board_writer.add_scalar("98. mnist test accuracy", 100. * correct / len(test_loader.dataset), epoch)
    tensor_board_writer.add_scalar("97. mnist test fitting loss", fitting_loss_sum, epoch)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def experiment_alternating(device: str = 'cuda', n_epochs: int = 20, kernel_learning_rate: float = 0.001, fitting_learning_rate: float = 0.001, log_interval: int = 100,
                           layer1_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                           layer2_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                           layer3_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                           learn_positions_after: int = 0,
                           learn_covariances_after: int = 0,
                           desc_string: str = ""):
    # Training settings
    torch.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/train_', begin=0, end=60000), batch_size=config.batch_size, num_workers=config.num_dataloader_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/test_', begin=0, end=10000), batch_size=100, num_workers=config.num_dataloader_workers)

    model = experiment_gm_mnist_model.Net(name=desc_string,
                                          layer1_m2m_fitting=layer1_m2m_fitting,
                                          layer2_m2m_fitting=layer2_m2m_fitting,
                                          layer3_m2m_fitting=layer3_m2m_fitting,
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

    kernel_optimiser = optim.Adam(model.parameters(), lr=kernel_learning_rate)
    fitting_optimiser = optim.Adam(model.fitting_parameters(), lr=fitting_learning_rate)

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'altr_{desc_string}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    # scheduler = StepLR(kernel_optimiser, step_size=1, gamma=args.gamma)

    # do not train kernels during initial phase.
    model.set_fitting_training(True)
    train(args, model, device, train_loader, kernel_optimiser=kernel_optimiser, fitting_optimiser=fitting_optimiser,
          epoch=0, train_kernels=False, train_fitting_layers=True, tensor_board_writer=tensor_board_writer)

    for epoch in range(1, n_epochs):
        model.set_position_learning(epoch >= learn_positions_after)
        model.set_covariance_learning(epoch >= learn_covariances_after)
        train_kernels = epoch % 2 == 0  # starts with epoch 1 / fitting
        model.set_fitting_training(not train_kernels)
        train(args, model, device, train_loader, kernel_optimiser=kernel_optimiser, fitting_optimiser=fitting_optimiser,
              epoch=epoch, train_kernels=train_kernels, train_fitting_layers=not train_kernels, tensor_board_writer=tensor_board_writer)

        if train_kernels:
            test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model:
            model.save_model()
            model.save_fitting_parameters()
            model.save_fitting_optimiser_state()


def experiment_probabalistic(device: str = 'cuda', n_epochs: int = 20, n_epochs_fitting_training: int = 10, kernel_learning_rate: float = 0.001, fitting_learning_rate: float = 0.001, log_interval: int = 100,
                             layer1_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                             layer2_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                             layer3_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                             learn_positions_after: int = 0,
                             learn_covariances_after: int = 0,
                             desc_string: str = "", ):
    # Training settings
    torch.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/train_', begin=0, end=60000), batch_size=config.batch_size, num_workers=config.num_dataloader_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/test_', begin=0, end=10000), batch_size=100, num_workers=config.num_dataloader_workers)

    model = experiment_gm_mnist_model.Net(name=desc_string,
                                          layer1_m2m_fitting=layer1_m2m_fitting,
                                          layer2_m2m_fitting=layer2_m2m_fitting,
                                          layer3_m2m_fitting=layer3_m2m_fitting,
                                          learn_positions=learn_positions_after == 0,
                                          learn_covariances=learn_covariances_after == 0,
                                          n_kernel_components=n_kernel_components)
    model.load()
    model = model.to(device)

    class Args:
        pass

    args = Args()
    args.log_interval = log_interval
    args.save_model = False

    kernel_optimiser = optim.Adam(model.parameters(), lr=kernel_learning_rate)
    fitting_optimiser = optim.Adam(model.fitting_parameters(), lr=fitting_learning_rate)

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'prb_{desc_string}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    # scheduler = StepLR(kernel_optimiser, step_size=1, gamma=args.gamma)

    # do not train kernels during initial phase.
    model.set_fitting_training(True)
    probDta = Args
    assert n_epochs_fitting_training > 0
    for epoch in range(0, n_epochs_fitting_training):
        probDta.averaged_fitting_losses = train(args, model, device, train_loader, kernel_optimiser=kernel_optimiser, fitting_optimiser=fitting_optimiser,
                                              epoch=epoch, train_kernels=False, train_fitting_layers=True, tensor_board_writer=tensor_board_writer)
    probDta.best_fitting_losses = probDta.averaged_fitting_losses

    for epoch in range(n_epochs_fitting_training, n_epochs):
        model.set_position_learning(epoch >= learn_positions_after)
        model.set_covariance_learning(epoch >= learn_covariances_after)
        train(args, model, device, train_loader, kernel_optimiser=kernel_optimiser, fitting_optimiser=fitting_optimiser,
              epoch=epoch, probDta=probDta, tensor_board_writer=tensor_board_writer)

        test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model:
            model.save_model()
            model.save_fitting_parameters()
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
        train(args, model, device, train_loader, optimizer, epoch, only_simulate=not train_mnist, train_fitting_layers=train_fitting_layers, tensor_board_writer=tensor_board_writer)
        test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model and train_mnist:
            model.save_model()

        if args.save_model and train_fitting_layers is not None:
            model.save_fitting_parameters()
            model.save_fitting_optimiser_state()


if __name__ == '__main__':
    main()
