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
    def __init__(self, train_fitting=False):
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

        # todo: all the relus must use the same net for now, because all of them save it to the same location on disc.
        self.relu2.net = self.relu1.net
        self.relu3.net = self.relu1.net

        self.bn0 = gm_modules.BatchNorm(per_gaussian_norm=True)
        self.bn = gm_modules.BatchNorm(per_gaussian_norm=False)

        self.train_fitting = train_fitting
        if self.train_fitting:
            self.train_fitting_epoch = 0
            self.trainer1 = gm_fitting.Trainer(self.relu1, n_training_samples=400)
            self.trainer2 = gm_fitting.Trainer(self.relu2, n_training_samples=400)
            self.trainer3 = gm_fitting.Trainer(self.relu3, n_training_samples=400)

    def forward(self, in_x: torch.Tensor, activation_out: typing.List[torch.Tensor] = None):
        in_x_norm = self.bn0(in_x)

        if self.train_fitting and self.training:
            x = self.gmc1(in_x_norm.detach())
            self.relu1.train_fitting(True)
            self.trainer1.train_on(x, self.relu1.bias, self.train_fitting_epoch)
            self.relu1.train_fitting(False)

            x = self.relu1(x)
            # x = self.maxPool1(x)
            x = self.gmc2(x)

            self.relu2.train_fitting(True)
            self.trainer2.train_on(x, self.relu2.bias, self.train_fitting_epoch)
            self.relu2.train_fitting(False)

            x = self.relu2(x)
            # x = self.maxPool2(x)
            x = self.gmc3(x)

            self.relu3.train_fitting(True)
            self.trainer3.train_on(x, self.relu3.bias, self.train_fitting_epoch)
            self.relu3.train_fitting(False)

            self.train_fitting_epoch = self.train_fitting_epoch + 1
            self.trainer3.save_weights()

        if activation_out is not None:
            activation_out.append(in_x_norm.detach())
        x = self.gmc1(in_x_norm)
        # integral11 = gm.integrate(x).mean()
        if activation_out is not None:
            activation_out.append(x.detach())
        x = self.relu1(x)
        x = self.bn(x)
        if activation_out is not None:
            activation_out.append(x.detach())
        integral12 = gm.integrate(x)
        # x = self.maxPool1(x)

        x = self.gmc2(x)
        # integral21 = gm.integrate(x)
        if activation_out is not None:
            activation_out.append(x.detach())
        x = self.relu2(x)
        x = self.bn(x)
        if activation_out is not None:
            activation_out.append(x.detach())
        # integral22 = gm.integrate(x)
        # x = self.maxPool2(x)

        x = self.gmc3(x)
        # integral31 = gm.integrate(x)
        if activation_out is not None:
            activation_out.append(x.detach())
        x = self.relu3(x)
        x = self.bn(x)
        if activation_out is not None:
            activation_out.append(x.detach())
        # integral32 = gm.integrate(x)
        # x = self.maxPool3(x)

        x = gm.integrate(x)
        # integral4 = x
        x = F.log_softmax(x, dim=1)
        # print(integral11.mean())
        # print(integral12.mean())
        #
        # print(integral21.mean())
        # print(integral22.mean())
        #
        # print(integral31.mean())
        # print(integral32.mean())
        #
        # print(integral4.mean())
        return x.view(-1, 10)


tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'gm_mnist_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data_all, target_all) in enumerate(train_loader):

        data_all, target_all = data_all.to(device), target_all.to(device)
        for k in range(4):
            data = data_all[k * 25:(k + 1) * 25]
            target = target_all[k * 25:(k + 1) * 25]

            optimizer.zero_grad()
            activation_out = list()
            output = model(data, activation_out=activation_out)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            i = (epoch - 1) * len(train_loader.dataset) * 4 + batch_idx * 4 + k
            if i % args.log_interval == 0:
                pred = output.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                tensor_board_writer.add_scalar("0. loss", loss.item(), i)
                tensor_board_writer.add_scalar("1. accuracy", 100 * correct / len(data), i)
                tensor_board_writer.add_image("conv layer 1", model.gmc1.debug_render(clamp=[-0.8, 0.8]), i, dataformats='HWC')
                tensor_board_writer.add_image("conv layer 2", model.gmc2.debug_render(clamp=[-0.08, 0.08]), i, dataformats='HWC')
                tensor_board_writer.add_image("conv layer 3", model.gmc3.debug_render(clamp=[-0.05, 0.05]), i, dataformats='HWC')
                for activation_index, act in enumerate(activation_out):
                    rendering = gm.render(act, batches=[0, 1], layers=[0, None], x_low=-2, x_high=30, y_low=-2, y_high=30, width=80, height=80)
                    rendering = madam_imagetools.colour_mapped(rendering.cpu().numpy(), -0.5/(28**2), 2.0/(28**2))
                    tensor_board_writer.add_image(f"activation {activation_index}", rendering, i, dataformats='HWC')

                print(
                    f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset) * len(data)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f} (accuracy: {100 * correct / len(data)})')


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
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('train_', begin=0, end=600), batch_size=None, collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('test_', begin=0, end=100), batch_size=None, collate_fn=lambda x: x)

    model = Net()
    print(f"experiment_gm_mnist.Net: trying to load {model_storage_path}")
    if pathlib.Path(model_storage_path).is_file():
        state_dict = torch.load(model_storage_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=True)
        print(f"experiment_gm_mnist.Net: loaded (missing: {missing_keys}, unexpected: {unexpected_keys}")
    else:
        print("experiment_gm_mnist.Net: not found")
    model = model.to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        # scheduler.step()

        if args.save_model:
            torch.save(model.state_dict(), model_storage_path)


if __name__ == '__main__':
    main()
