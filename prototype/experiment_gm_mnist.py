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


class Net(nn.Module):
    def __init__(self, train_fitting_layers: typing.Set[int] = None):
        super(Net, self).__init__()
        n_layers_1 = 5
        n_layers_2 = 6
        # sqrt(2) / (n_kernel_components * n_input_gauss * math.sqrt(2 * math.pi * det(cov))
        self.gmc1 = gm_modules.GmConvolution(n_layers_in=1, n_layers_out=n_layers_1, n_kernel_components=n_kernel_components,
                                             position_range=2, covariance_range=0.5,
                                             learn_positions=False, learn_covariances=False,
                                             weight_sd=0.4).cuda()
        self.relu1 = gm_modules.GmBiasAndRelu(n_layers=n_layers_1, n_output_gaussians=25, max_bias=0.0).cuda()
        # self.maxPool1 = gm_modules.MaxPooling(10)
        self.gmc2 = gm_modules.GmConvolution(n_layers_in=n_layers_1, n_layers_out=n_layers_2, n_kernel_components=n_kernel_components,
                                             position_range=4, covariance_range=2,
                                             learn_positions=False, learn_covariances=False,
                                             weight_sd=0.04).cuda()
        self.relu2 = gm_modules.GmBiasAndRelu(n_layers=n_layers_2, n_output_gaussians=12, max_bias=0.0).cuda()
        # self.maxPool2 = gm_modules.MaxPooling(10)
        self.gmc3 = gm_modules.GmConvolution(n_layers_in=n_layers_2, n_layers_out=10, n_kernel_components=n_kernel_components,
                                             position_range=8, covariance_range=4,
                                             learn_positions=False, learn_covariances=False,
                                             weight_sd=0.025).cuda()
        self.relu3 = gm_modules.GmBiasAndRelu(n_layers=10, n_output_gaussians=5, max_bias=0.0).cuda()
        # self.maxPool3 = gm_modules.MaxPooling(2)

        # to do: all the relus must use the same net for now, because all of them save it to the same location on disc.
        # for now testing seperate nets
        # self.relu2.net = self.relu1.net
        # self.relu3.net = self.relu1.net

        self.bn0 = gm_modules.BatchNorm(per_gaussian_norm=True)
        self.bn = gm_modules.BatchNorm(per_gaussian_norm=False)

        self.train_fitting_layers = train_fitting_layers
        if self.train_fitting_layers is not None:
            self.train_fitting_epoch = 0

            if 1 in self.train_fitting_layers:
                self.trainer1 = gm_fitting.Trainer(self.relu1, n_training_samples=400)
            if 2 in self.train_fitting_layers:
                self.trainer2 = gm_fitting.Trainer(self.relu2, n_training_samples=400)
            if 3 in self.train_fitting_layers:
                self.trainer3 = gm_fitting.Trainer(self.relu3, n_training_samples=400)

    def forward(self, in_x: torch.Tensor):
        in_x_norm = self.bn0(in_x)

        if self.train_fitting_layers is not None and self.training:
            x = self.gmc1(in_x_norm.detach())
            if 1 in self.train_fitting_layers:
                self.relu1.train_fitting(True)
                self.trainer1.train_on(x.detach(), self.relu1.bias.detach(), self.train_fitting_epoch)
                self.relu1.train_fitting(False)
                self.trainer1.save_weights()

            x = self.relu1(x)
            x = self.bn(x)
            # x = self.maxPool1(x)
            x = self.gmc2(x)

            if 2 in self.train_fitting_layers:
                self.relu2.train_fitting(True)
                self.trainer2.train_on(x.detach(), self.relu2.bias.detach(), self.train_fitting_epoch)
                self.relu2.train_fitting(False)
                self.trainer2.save_weights()

            x = self.relu2(x)
            x = self.bn(x)
            # x = self.maxPool2(x)
            x = self.gmc3(x)

            if 3 in self.train_fitting_layers:
                self.relu3.train_fitting(True)
                self.trainer3.train_on(x.detach(), self.relu3.bias.detach(), self.train_fitting_epoch)
                self.relu3.train_fitting(False)
                self.trainer3.save_weights()

            self.train_fitting_epoch = self.train_fitting_epoch + 1

        x = self.gmc1(in_x_norm)
        x = self.relu1(x)
        x = self.bn(x)
        # x = self.maxPool1(x)

        x = self.gmc2(x)
        x = self.relu2(x)
        x = self.bn(x)
        # x = self.maxPool2(x)

        x = self.gmc3(x)
        x = self.relu3(x)
        x = self.bn(x)
        # x = self.maxPool3(x)

        x = gm.integrate(x)
        x = F.log_softmax(x, dim=1)
        return x.view(-1, 10)


tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'gm_mnist_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')


def train(args, model, device, train_loader, optimizer, epoch, only_simulate):
    model.train()
    for batch_idx, (data_all, target_all) in enumerate(train_loader):

        data_all, target_all = data_all.to(device), target_all.to(device)

        batch_divisor = 4

        for k in range(batch_divisor):
            divided_batch_length = 25
            data = data_all[k * divided_batch_length:(k + 1) * divided_batch_length]
            target = target_all[k * divided_batch_length:(k + 1) * divided_batch_length]

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            if not only_simulate:
                loss.backward()
                optimizer.step()
            i = (epoch - 1) * len(train_loader.dataset) * batch_divisor + batch_idx * batch_divisor + k
            if i % args.log_interval == 0:
                pred = output.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                tensor_board_writer.add_scalar("0. loss", loss.item(), i)
                tensor_board_writer.add_scalar("1. accuracy", 100 * correct / len(data), i)
                tensor_board_writer.add_image("conv 1", model.gmc1.debug_render(clamp=[-0.8, 0.8]), i, dataformats='HWC')
                tensor_board_writer.add_image("conv 2", model.gmc2.debug_render(clamp=[-0.08, 0.08]), i, dataformats='HWC')
                tensor_board_writer.add_image("conv 3", model.gmc3.debug_render(clamp=[-0.05, 0.05]), i, dataformats='HWC')

                tensor_board_writer.add_image("relu 1", model.relu1.debug_render(position_range=[-14, -14, 42, 42], clamp=[-4/(28**2), 16.0/(28**2)]), i, dataformats='HWC')
                tensor_board_writer.add_image("relu 2", model.relu2.debug_render(position_range=[-14, -14, 42, 42], clamp=[-4/(28**2), 16.0/(28**2)]), i, dataformats='HWC')
                tensor_board_writer.add_image("relu 3", model.relu3.debug_render(position_range=[-14, -14, 42, 42], clamp=[-4/(28**2), 16.0/(28**2)]), i, dataformats='HWC')

                print(f'Train Epoch: {epoch} [{(batch_idx * batch_divisor + k) * len(data)}/{len(train_loader.dataset) * len(data_all)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} (accuracy: {100 * correct / len(data)})')


def test(args, model, device, test_loader):
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

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset) * len(data)} ({100. * correct / (len(test_loader.dataset) * len(data)):.0f}%)\n')


def main():
    default_learning_rate = 0.01
    default_epochs = 6 * 10
    default_log_interval = 10
    train_fitting_layers = {2}
    train_mnist = False
    model_storage_path = config.data_base_path / "mnist_gmcnet.pt"
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

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('train_', begin=0, end=600), batch_size=None, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('test_', begin=0, end=100), batch_size=None, collate_fn=lambda x: x)

    model = Net(train_fitting_layers)
    print(f"experiment_gm_mnist.Net: trying to load {model_storage_path}")
    if pathlib.Path(model_storage_path).is_file():
        state_dict = torch.load(model_storage_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        print(f"experiment_gm_mnist.Net: loaded (missing: {missing_keys}, unexpected: {unexpected_keys}")
    else:
        print("experiment_gm_mnist.Net: not found")
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, not train_mnist)
        test(args, model, device, test_loader)
        # scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), model_storage_path)


if __name__ == '__main__':
    main()
