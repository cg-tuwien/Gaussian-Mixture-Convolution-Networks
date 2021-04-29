from __future__ import print_function
import datetime
import math
import time
import pathlib

import torch
import torch.jit
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.optimizer as Optimizer
import torch.utils.data
import torch.utils.tensorboard
import typing

import gmc.mixture as gm
import qm9.model
import gmc.fitting
import qm9.config as Config
from qm9.data_set import DataSet

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


def render_debug_images_to_tensorboard(model, epoch, tensor_board_writer, config: Config):
    for i, gmc in enumerate(model.gmcs):
        rendering = gmc.debug_render3d(clamp=(-0.80, 0.80), camera={'positions': (2.91864, 3.45269, 2.76324), 'lookat': (0.0, 0.0, 0.0), 'up': (0.0, 1.0, 0.0)})
        tensor_board_writer.add_image(f"conv {i}", rendering, epoch, dataformats='HWC')
        gmc.debug_save3d(f"{config.data_base_path}/debug_out/kernels/conv{i}")

    clamps = ((-0.005, 0.005), (-0.025, 0.025), (-0.1, 0.1), (-0.2, 0.2), (-0.3, 0.3))
    for i, relu in enumerate(model.relus):
        rendering = relu.debug_render3d(clamp=clamps[i], camera={'positions': (22.0372, 30.9668, 23.3432), 'lookat': (0.0, 0.0, 0.0), 'up': (0.0, 1.0, 0.0)})
        tensor_board_writer.add_image(f"relu {i+1}", rendering, epoch, dataformats='HWC')
        relu.debug_save3d(f"{config.data_base_path}/debug_out/activations/relu{i+1}")


def train(model: qm9.model.Net, device: str, train_loader: torch.utils.data.DataLoader,
          kernel_optimiser: Optimizer, weight_decay_optimiser: Optimizer, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter, config: Config):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        step = epoch * (len(train_loader.dataset) - len(train_loader.dataset) % len(data)) + batch_idx * len(data)  # modulo, because we are dropping non-full batches

        batch_start_time = time.perf_counter()
        tx = time.perf_counter()

        temp_tb = None
        # if step % args.log_interval == 0:
        #     temp_tb = tensor_board_writer
        output = model(data, temp_tb)
        loss = F.mse_loss(config.target_range * output, target)

        ty = time.perf_counter()
        # regularisation_loss = model.regularisation_loss() * len(data)
        training_loss = loss  # (loss + regularisation_loss)

        kernel_optimiser.zero_grad()
        tz = time.perf_counter()
        training_loss.backward()
        tw = time.perf_counter()
        kernel_optimiser.step()

        if weight_decay_optimiser is not None:
            weight_decay_optimiser.zero_grad()
            model.weight_decay_loss().backward()
            weight_decay_optimiser.step()

        batch_end_time = time.perf_counter()

        if step % config.log_interval == 0:
            mae = F.l1_loss(config.target_range * output, target)
            tensor_board_writer.add_scalar("00. RMSE training loss", training_loss.sqrt().item(), step)
            tensor_board_writer.add_scalar("00.2 training MAE", mae.item(), step)
            tensor_board_writer.add_scalar("01.1 Mean Output", output.mean().item(), step)
            tensor_board_writer.add_scalar("01.2 Mean Abs Output", output.abs().mean().item(), step)
            tensor_board_writer.add_scalar("02. kernel loss", loss.item(), step)
            tensor_board_writer.add_scalar("03.1 total duration per sample", (batch_end_time - batch_start_time) / len(data), step)
            tensor_board_writer.add_scalar("03.2 forward time per sample", (ty - tx) / len(data), step)
            tensor_board_writer.add_scalar("03.3 backward time per sample", (tw - tz) / len(data), step)

            # tensor_board_writer.add_scalar("07.1 model layer 1 max(bias)", model.biases[0].max().item(), step)
            # tensor_board_writer.add_scalar("07.2 model layer 2 max(bias)", model.biases[1].max().item(), step)
            # tensor_board_writer.add_scalar("07.3 model layer 3 max(bias)", model.biases[2].max().item(), step)
            # tensor_board_writer.add_scalar("07.1 model layer 1 min(bias)", model.biases[0].min().item(), step)
            # tensor_board_writer.add_scalar("07.2 model layer 2 min(bias)", model.biases[1].min().item(), step)
            # tensor_board_writer.add_scalar("07.3 model layer 3 min(bias)", model.biases[2].min().item(), step)

            # tensor_board_writer.add_scalar("04. mnist training regularisation loss", regularisation_loss.item(), step)

            for i, relu in enumerate(model.relus):
                mse = gmc.fitting.mse(*relu.last_in, *relu.last_out)
                tensor_board_writer.add_scalar(f"05.1 convolution layer {i} relu mse", mse, step)

            # for name, timing in model.timings.items():
            #     tensor_board_writer.add_scalar(f"06. {name} time", timing, step)

            if config.log_tensorboard_renderings:
                render_debug_images_to_tensorboard(model, step, tensor_board_writer, config)

            print(f'Training kernels: {epoch}/{step} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tRMSE: {training_loss.sqrt().item():.6f}, MAE: {mae.item():.6f}, '
                  f'Cuda max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024}G, '
                  f'batch time: {batch_end_time - batch_start_time}')
            tensor_board_writer.add_scalar("11. CUDA max memory allocated [GiB]", torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, step)

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
            c.save(f"{config.fitting_test_data_store_path}/full_input_batch{batch_idx}.pt")

            c = torch.jit.script(Container(after_fixed_point))
            c.save(f"{config.fitting_test_data_store_path}/after_fixed_point_batch{batch_idx}.pt")


def test(model: qm9.model.Net, device: str, test_loader: torch.utils.data.DataLoader, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter, config: Config):
    model.eval()
    test_loss = 0
    test_mae = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.mse_loss(config.target_range * output, target, reduction='sum').item()  # sum up batch loss
            test_mae += F.l1_loss(config.target_range * output, target, reduction='sum').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    test_loss = math.sqrt(test_loss)
    test_mae /= len(test_loader.dataset)
    tensor_board_writer.add_scalar("99. test RMSE loss", test_loss, epoch)
    tensor_board_writer.add_scalar("99. test MAE", test_mae, epoch)
    print(f'\nTest set: Average RMSE loss: {test_loss:.4f}, MAE: {test_mae:.4f})\n')


def experiment(device: str = 'cuda', desc_string: str = "", config: Config = None):
    # Training settings
    torch.manual_seed(0)

    model = qm9.model.Net(name=desc_string,
                          learn_positions=config.learn_positions_after == 0,
                          learn_covariances=config.learn_covariances_after == 0,
                          config=config)
    model.load()
    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(DataSet(config, config.training_start_index, config.training_end_index, model.learnable_atom_weights, model.learnable_atom_radii),
                                               batch_size=config.batch_size, num_workers=config.num_dataloader_workers, shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(DataSet(config, config.validation_start_index, config.validation_end_index, model.learnable_atom_weights, model.learnable_atom_radii),
                                              batch_size=config.batch_size, num_workers=config.num_dataloader_workers)

    kernel_optimiser = optim.Adam(model.parameters(), lr=config.kernel_learning_rate)
    # kernel_optimiser = optim.SGD(model.parameters(), lr=config.kernel_learning_rate)
    weight_decay_optimiser = None  # optim.SGD(model.parameters(), lr=(config.weight_decay_rate * config.kernel_learning_rate * 0))
    # todo: load optimiser state
    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'{desc_string}_{datetime.datetime.now().strftime("%m%d_%H%M")}')

    # scheduler = StepLR(kernel_optimiser, step_size=1, gamma=args.gamma)

    for epoch in range(config.n_epochs):
        model.set_position_learning(epoch >= config.learn_positions_after)
        model.set_covariance_learning(epoch >= config.learn_covariances_after)
        train(model, device, train_loader, kernel_optimiser=kernel_optimiser, weight_decay_optimiser=weight_decay_optimiser, epoch=epoch, tensor_board_writer=tensor_board_writer, config=config)
        test(model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer, config=config)
        # scheduler.step()

        if config.save_model:
            model.save_model()
            torch.save(kernel_optimiser.state_dict(), f"{model.storage_path}.optimiser")
            torch.save(weight_decay_optimiser.state_dict(), f"{model.storage_path}.optimiser")
