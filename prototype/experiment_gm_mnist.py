from __future__ import print_function
import argparse
import datetime
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


class GmMnistDataSet(torch.utils.data.Dataset):
    def __init__(self, prefix: str, begin: int, end: int):
        self.prefix = prefix
        self.begin = begin
        self.end = end

    def __len__(self):
        return self.end - self.begin

    def __getitem__(self, index):
        return gm.load(f"{self.prefix}{index}")


def render_debug_images_to_tensorboard(model, epoch, tensor_board_writer):
    tensor_board_writer.add_image("mnist conv 1", model.gmc1.debug_render(clamp=[-0.80, 0.80]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist conv 2", model.gmc2.debug_render(clamp=[-0.32, 0.32]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist conv 3", model.gmc3.debug_render(clamp=[-0.20, 0.20]), epoch, dataformats='HWC')

    tensor_board_writer.add_image("mnist relu 1", model.relu1.debug_render(position_range=[-14, -14, 42, 42], clamp=[-4 / (28 ** 2), 16.0 / (28 ** 2)]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist relu 2", model.relu2.debug_render(position_range=[-14, -14, 42, 42], clamp=[-20 / (28 ** 2), 80.0 / (28 ** 2)]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist relu 3", model.relu3.debug_render(position_range=[-14, -14, 42, 42], clamp=[-6 / (28 ** 2), 24.0 / (28 ** 2)]), epoch, dataformats='HWC')


def train(args, model: experiment_gm_mnist_model.Net, device, train_loader, optimizer, epoch, only_simulate, train_fitting_layers, combined_fitting_layer_training, tensor_board_writer):
    model.train()
    for batch_idx, (data_all, target_all) in enumerate(train_loader):

        data_all, target_all = data_all.to(device), target_all.to(device)

        batch_divisor = 4

        for k in range(batch_divisor):
            i = epoch * len(train_loader.dataset) * batch_divisor + batch_idx * batch_divisor + k
            tensorboard_writer_option = None
            if i % args.log_interval == 0:
                tensorboard_writer_option = tensor_board_writer
            divided_batch_length = 25
            data = data_all[k * divided_batch_length:(k + 1) * divided_batch_length]
            target = target_all[k * divided_batch_length:(k + 1) * divided_batch_length]

            if train_fitting_layers is not None:
                # model.set_fitting_training(True)
                model.run_fitting_sampling(data, sampling_layers=train_fitting_layers, train=True, epoch=i, tensor_board_writer=tensorboard_writer_option)
                # model.set_fitting_training(False)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            regularisation_loss = model.regularisation_loss()
            training_loss = (loss + regularisation_loss)

            if not only_simulate and combined_fitting_layer_training is None:
                training_loss.backward()
                optimizer.step()
            elif not only_simulate:
                fitting_loss = 100 * model.run_fitting_sampling(data, sampling_layers=combined_fitting_layer_training, train=False, epoch=i, tensor_board_writer=tensorboard_writer_option)
                training_loss = (loss + regularisation_loss + fitting_loss)
                training_loss.backward()
                optimizer.step()

            if i % args.log_interval == 0:
                pred = output.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                tensor_board_writer.add_scalar("0. mnist training loss", training_loss.item(), i)
                tensor_board_writer.add_scalar("1. mnist training accuracy", 100 * correct / len(data), i)
                tensor_board_writer.add_scalar("2. mnist kernel loss", loss.item(), i)
                tensor_board_writer.add_scalar("3. mnist training regularisation loss", regularisation_loss.item(), i)
                if combined_fitting_layer_training is not None:
                    tensor_board_writer.add_scalar("4. mnist fitting loss", fitting_loss.item(), i)
                render_debug_images_to_tensorboard(model, i, tensor_board_writer)

                print(f'Train Epoch: {epoch} [{(batch_idx * batch_divisor + k) * len(data)}/{len(train_loader.dataset) * len(data_all)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} (accuracy: {100 * correct / len(data)})')

        if args.save_model and batch_idx % args.log_interval == 0 and not only_simulate:
            model.save_model()
            # print(f"experiment_gm_mnist.tain: saving optimiser state to {model.storage_path}.optimiser")
            # torch.save(optimizer.state_dict(), f"{model.storage_path}.optimiser")

        if args.save_model and batch_idx % args.log_interval == 0 and train_fitting_layers is not None:
            model.save_fitting_parameters()
            model.save_fitting_optimiser_state()

        if args.save_model and batch_idx % args.log_interval == 0 and combined_fitting_layer_training is not None:
            model.save_fitting_parameters()


def test(args, model, device, test_loader, epoch, tensor_board_writer):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    tensor_board_writer.add_scalar("99. mnist test loss", test_loss, epoch)
    tensor_board_writer.add_scalar("98. mnist test accuracy", 100. * correct / (len(test_loader.dataset) * len(data)), epoch)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset) * len(data)} ({100. * correct / (len(test_loader.dataset) * len(data)):.0f}%)\n')


def experiment_alternating(device: str = 'cuda', n_epochs: int = 20, learning_rate: float = 0.01, log_interval: int = 100,
                           layer1_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                           layer2_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                           layer3_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                           learn_positions: bool = False,
                           learn_covariances: bool = False,
                           desc_string: str = ""):
    # Training settings
    torch.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/train_', begin=0, end=600), batch_size=None, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/test_', begin=0, end=100), batch_size=None, collate_fn=lambda x: x)

    model = experiment_gm_mnist_model.Net(name=desc_string,
                                          layer1_m2m_fitting=layer1_m2m_fitting,
                                          layer2_m2m_fitting=layer2_m2m_fitting,
                                          layer3_m2m_fitting=layer3_m2m_fitting,
                                          learn_positions=learn_positions,
                                          learn_covariances=learn_covariances,
                                          n_kernel_components=n_kernel_components)
    model.load()
    model = model.to(device)

    class Args:
        pass

    args = Args()
    args.save_model = True
    args.log_interval = log_interval

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'gm_mnist_alternate_{desc_string}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(0, n_epochs):
        train(args, model, device, train_loader, optimizer, epoch * 2, only_simulate=True, train_fitting_layers={1, 2, 3}, combined_fitting_layer_training=None, tensor_board_writer=tensor_board_writer)
        train(args, model, device, train_loader, optimizer, epoch * 2 + 1, only_simulate=False, train_fitting_layers=None, combined_fitting_layer_training=None, tensor_board_writer=tensor_board_writer)
        test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model:
            model.save_model()

        if args.save_model:
            model.save_fitting_parameters()
            model.save_fitting_optimiser_state()


def experiment_combined_loss(device: str = 'cuda', n_epochs: int = 20, learning_rate: float = 0.01, log_interval: int = 100,
                             layer1_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                             layer2_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                             layer3_m2m_fitting: typing.Callable = gm_modules.generate_default_fitting_module,
                             learn_positions: bool = False,
                             learn_covariances: bool = False,
                             desc_string: str = ""):
    # Training settings
    torch.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/train_', begin=0, end=600), batch_size=None, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/test_', begin=0, end=100), batch_size=None, collate_fn=lambda x: x)

    model = experiment_gm_mnist_model.Net(name=desc_string,
                                          layer1_m2m_fitting=layer1_m2m_fitting,
                                          layer2_m2m_fitting=layer2_m2m_fitting,
                                          layer3_m2m_fitting=layer3_m2m_fitting,
                                          learn_positions=learn_positions,
                                          learn_covariances=learn_covariances,
                                          n_kernel_components=n_kernel_components)
    model.load()
    model = model.to(device)

    class Args:
        pass

    args = Args()
    args.save_model = True
    args.log_interval = log_interval

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'gm_mnist_combined_{desc_string}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    train(args, model, device, train_loader, optimizer, 0, only_simulate=True, train_fitting_layers={1, 2, 3}, combined_fitting_layer_training=None, tensor_board_writer=tensor_board_writer)
    train(args, model, device, train_loader, optimizer, 1, only_simulate=True, train_fitting_layers={1, 2, 3}, combined_fitting_layer_training=None, tensor_board_writer=tensor_board_writer)
    for epoch in range(2, n_epochs):
        train(args, model, device, train_loader, optimizer, epoch, only_simulate=False, train_fitting_layers=None, combined_fitting_layer_training={1, 2, 3}, tensor_board_writer=tensor_board_writer)
        test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model:
            model.save_model()
            model.save_fitting_parameters()


def main():
    default_learning_rate = 0.01
    default_epochs = 6 * 10
    default_log_interval = 20
    train_fitting_layers = {1, 2, 3}
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
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
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
