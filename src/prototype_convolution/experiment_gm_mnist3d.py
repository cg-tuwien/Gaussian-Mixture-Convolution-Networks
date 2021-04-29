from __future__ import print_function
import datetime
import time

import torch
import torch.jit
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.optimizer as Optimizer
import torch.utils.data
import torch.utils.tensorboard
import typing

import gmc.mixture as gm
import prototype_convolution.experiment_gm_mnist3d_model as experiment_model
import gmc.fitting
import prototype_convolution.config

# based on https://github.com/pytorch/examples/blob/master/mnist/main.py

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

        n_batch = gm.n_batch(mixture)
        n_layers = gm.n_layers(mixture)
        n_components = gm.n_components(mixture)

        weights = gm.weights(mixture)

        positions = gm.positions(mixture)
        positions3d = torch.zeros([n_batch, n_layers, n_components, 3])
        positions3d[..., 0:2] = positions

        covariances = gm.covariances(mixture)
        covariances3d = torch.zeros([n_batch, n_layers, n_components, 3, 3])
        covariances3d[..., 0:2, 0:2] = covariances
        covariances3d[..., 2, 2] = 1

        return gm.pack_mixture(weights, positions3d, covariances3d)[0], meta


def render_debug_images_to_tensorboard(model, epoch, tensor_board_writer):
    tensor_board_writer.add_image("mnist conv 1", model.gmc1.debug_render3d(clamp=[-0.80, 0.80]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist conv 2", model.gmc2.debug_render3d(clamp=[-0.32, 0.32]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist conv 3", model.gmc3.debug_render3d(clamp=[-0.20, 0.20]), epoch, dataformats='HWC')

    # tensor_board_writer.add_image("mnist relu 1", model.relus[0].debug_render(position_range=[-14, -14, 42, 42], clamp=[-4 / (28 ** 2), 16.0 / (28 ** 2)]), epoch, dataformats='HWC')
    # tensor_board_writer.add_image("mnist relu 2", model.relus[1].debug_render(position_range=[-14, -14, 42, 42], clamp=[-20 / (28 ** 2), 80.0 / (28 ** 2)]), epoch, dataformats='HWC')
    # tensor_board_writer.add_image("mnist relu 3", model.relus[2].debug_render(position_range=[-14, -14, 42, 42], clamp=[-6 / (28 ** 2), 24.0 / (28 ** 2)]), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist relu 1", model.relus[0].debug_render3d(), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist relu 2", model.relus[1].debug_render3d(), epoch, dataformats='HWC')
    tensor_board_writer.add_image("mnist relu 3", model.relus[2].debug_render3d(), epoch, dataformats='HWC')


def train(args, model: experiment_model.Net, device: torch.device, train_loader: torch.utils.data.DataLoader,
          kernel_optimiser: Optimizer, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter, config: prototype_convolution.config):
    model.train()
    start_time = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        step = epoch * len(train_loader.dataset) + batch_idx * len(data)

        batch_start_time = time.perf_counter()
        tx = time.perf_counter()

        temp_tb = None
        # if step % args.log_interval == 0:
        #     temp_tb = tensor_board_writer
        output = model(data, temp_tb)
        loss = F.nll_loss(output, target)
        ty = time.perf_counter()
        # regularisation_loss = model.regularisation_loss() * len(data)
        training_loss = loss  # (loss + regularisation_loss)

        kernel_optimiser.zero_grad()
        tz = time.perf_counter()
        training_loss.backward()
        tw = time.perf_counter()
        kernel_optimiser.step()
        batch_end_time = time.perf_counter()

        if step % args.log_interval == 0:
            pred = output.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            tensor_board_writer.add_scalar("00. mnist training loss", training_loss.item(), step)
            tensor_board_writer.add_scalar("01. mnist training accuracy", 100 * correct / len(data), step)
            tensor_board_writer.add_scalar("02. mnist kernel loss", loss.item(), step)
            tensor_board_writer.add_scalar("03.1 total duration per sample", (batch_end_time - batch_start_time) / len(data), step)
            tensor_board_writer.add_scalar("03.2 forward time per batch", (ty - tx), step)
            tensor_board_writer.add_scalar("03.3 backward time per batch", (tw - tz), step)

            # tensor_board_writer.add_scalar("07.1 model layer 1 max(bias)", model.biases[0].max().item(), step)
            # tensor_board_writer.add_scalar("07.2 model layer 2 max(bias)", model.biases[1].max().item(), step)
            # tensor_board_writer.add_scalar("07.3 model layer 3 max(bias)", model.biases[2].max().item(), step)
            # tensor_board_writer.add_scalar("07.1 model layer 1 min(bias)", model.biases[0].min().item(), step)
            # tensor_board_writer.add_scalar("07.2 model layer 2 min(bias)", model.biases[1].min().item(), step)
            # tensor_board_writer.add_scalar("07.3 model layer 3 min(bias)", model.biases[2].min().item(), step)

            # tensor_board_writer.add_scalar("04. mnist training regularisation loss", regularisation_loss.item(), step)

            for i, relu in enumerate(model.relus):
                mse = gmc.fitting.mse(*relu.last_in, *relu.last_out)
                tensor_board_writer.add_scalar(f"05.1 model layer {i} relu mse", mse, step)

            # for name, timing in model.timings.items():
            #     tensor_board_writer.add_scalar(f"06. {name} time", timing, step)

            render_debug_images_to_tensorboard(model, step, tensor_board_writer)

            print(f'Training kernels: {epoch}/{step} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tClassification loss: {loss.item():.6f} (accuracy: {100 * correct / len(data)}), '
                  f'Cuda max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024}G, '
                  f'batch time: {batch_end_time - batch_start_time}')
            tensor_board_writer.add_scalar("11. CUDA max memory allocated [GiB]", torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, step)

            if args.save_model:
                model.save_model()
            # print(f"experiment_gm_mnist.tain: saving optimiser state to {model.storage_path}.optimiser")
            # torch.save(kernel_optimiser.state_dict(), f"{model.storage_path}.optimiser")

        if epoch == config.fitting_test_data_store_at_epoch and batch_idx < config.fitting_test_data_store_n_batches:
            full_input = dict()
            after_fixed_point = dict()
            for i, relu in enumerate(model.relus):
                full_input[f"{i}"] = relu.last_in[0].detach().cpu()
                full_input[f"{i}_bias"] = relu.last_in[1].detach().cpu()

                fp_fitting, fp_const, _ = gmc.fitting.fixed_point_and_tree_hem(relu.last_in[0].detach(), relu.last_in[1].detach(), n_components=-1)
                after_fixed_point[f"{i}"] = fp_fitting.cpu()
                after_fixed_point[f"{i}_constant"] = fp_const.cpu()
                # gm.save(relu.last_in[0], f"fitting_input/fitting_input_batch{batch_idx}_netlayer{i}", relu.last_in[1].detach().cpu())

            class Container(torch.nn.Module):
                def __init__(self, my_values):
                    super().__init__()
                    for key in my_values:
                        setattr(self, key, my_values[key])

            c = torch.jit.script(Container(full_input))
            c.save(f"{config.fitting_test_data_store_path}3d/full_input_batch{batch_idx}.pt")

            c = torch.jit.script(Container(after_fixed_point))
            c.save(f"{config.fitting_test_data_store_path}3d/after_fixed_point_batch{batch_idx}.pt")

    end_time = time.perf_counter()

    tensor_board_writer.add_scalar("10. batch_duration", end_time - start_time, step)


def test(args, model: experiment_model.Net, device: torch.device, test_loader: torch.utils.data.DataLoader, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter):
    model.eval()
    test_loss = 0
    correct = 0
    fitting_loss_sum = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    tensor_board_writer.add_scalar("99. mnist test loss", test_loss, epoch)
    tensor_board_writer.add_scalar("98. mnist test accuracy", 100. * correct / len(test_loader.dataset), epoch)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def experiment(device: str = 'cuda', n_epochs: int = 20, kernel_learning_rate: float = 0.001, log_interval: int = 100,
               learn_positions_after: int = 0, learn_covariances_after: int = 0, desc_string: str = "", gmcn_config = None):
    # Training settings
    torch.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/train_', begin=0, end=60000), batch_size=gmcn_config.batch_size, num_workers=gmcn_config.num_dataloader_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet('mnist/test_', begin=0, end=10000), batch_size=gmcn_config.batch_size, num_workers=gmcn_config.num_dataloader_workers)

    model = experiment_model.Net(name=desc_string,
                                 learn_positions=learn_positions_after == 0,
                                 learn_covariances=learn_covariances_after == 0,
                                 gmcn_config=gmcn_config)
    model.load()
    model = model.to(device)

    class Args:
        pass

    args = Args()
    args.save_model = True
    args.log_interval = log_interval
    args.save_model = False

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)
    #
    # for parameter in model.parameters():
    #     print(parameter)

    kernel_optimiser = optim.Adam(model.parameters(), lr=kernel_learning_rate)
    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(gmcn_config.data_base_path / 'tensorboard' / f'{desc_string}_{datetime.datetime.now().strftime("%m%d_%H%M")}')

    # scheduler = StepLR(kernel_optimiser, step_size=1, gamma=args.gamma)

    for epoch in range(n_epochs):
        model.set_position_learning(epoch >= learn_positions_after)
        model.set_covariance_learning(epoch >= learn_covariances_after)
        train(args, model, device, train_loader, kernel_optimiser=kernel_optimiser, epoch=epoch, tensor_board_writer=tensor_board_writer, config=gmcn_config)
        test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model:
            model.save_model()
            model.save_fitting_optimiser_state()
