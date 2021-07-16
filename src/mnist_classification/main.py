from __future__ import print_function
import datetime
import time
import typing

import torch
import torch.jit
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.optimizer as Optimizer
import torch.utils.data
import torch.utils.tensorboard

import gmc.fitting
import gmc.inout
import gmc.model
from mnist_classification.config import Config
import mnist_classification.input_fitting as input_fitting

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
        mixture, meta = gmc.inout.load(f"{self.prefix}{index + self.begin}")
        if len(meta.shape) == 1:
            return mixture[0], meta[0]
        return mixture[0], meta


def render_debug_images_to_tensorboard(model, epoch, tensor_board_writer):
    for i, gmc in enumerate(model.gmcs):
        tensor_board_writer.add_image(f"mnist conv {i}", gmc.debug_render(clamp=[-2.2, 2.2]), epoch, dataformats='HWC')
    for i, relu in enumerate(model.relus):
        tensor_board_writer.add_image(f"mnist relu {i}", relu.debug_render(clamp=[-5, 5], image_size=200), epoch, dataformats='HWC')


def train(model: gmc.model.Net, device: str, train_loader: torch.utils.data.DataLoader,
          kernel_optimiser: Optimizer, weight_decay_optimiser: Optimizer, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter, config: Config):
    model.train()
    start_time = time.perf_counter()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        step = epoch * (len(train_loader.dataset) - len(train_loader.dataset) % len(data)) + batch_idx * len(data)  # modulo, because we are dropping non-full batches

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

        weight_decay_optimiser.zero_grad()
        model.weight_decay_loss().backward()
        weight_decay_optimiser.step()

        batch_end_time = time.perf_counter()

        if step % config.log_interval == 0:
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
                tensor_board_writer.add_scalar(f"05.1 convolution layer {i} relu mse", mse, step)

            # for name, timing in model.timings.items():
            #     tensor_board_writer.add_scalar(f"06. {name} time", timing, step)

            if config.log_tensorboard_renderings:
                render_debug_images_to_tensorboard(model, step, tensor_board_writer)

            print(f'Training kernels: {epoch}/{step} [{batch_idx}/{len(train_loader)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tClassification loss: {loss.item():.6f} (accuracy: {100 * correct / len(data)}), '
                  f'Cuda max memory allocated: {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024}G, '
                  f'batch time: {batch_end_time - batch_start_time}')

            tensor_board_writer.add_scalar("11. CUDA max memory allocated [GiB]", torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, step)

        if epoch == config.fitting_test_data_store_at_epoch and batch_idx < config.fitting_test_data_store_n_batches:
            conv_input = dict()
            for i, conv in enumerate(model.gmcs):
                conv_input[f"conv_layer_{i}_data"] = conv.last_in[0].cpu()
                conv_input[f"conv_layer_{i}_kernels"] = conv.kernels().detach().cpu()
            # full_input = dict()
            # after_fixed_point = dict()
            # for i, relu in enumerate(model.relus):
            #     full_input[f"{i}"] = relu.last_in[0].detach().cpu()
            #     full_input[f"{i}_bias"] = relu.last_in[1].detach().cpu()
            #
            #     fp_fitting, fp_const, _ = gmc.fitting.fixed_point_and_tree_hem(relu.last_in[0].detach(), relu.last_in[1].detach(), n_components=-1)
            #     after_fixed_point[f"{i}"] = fp_fitting.cpu()
            #     after_fixed_point[f"{i}_constant"] = fp_const.cpu()
                # gm.save(relu.last_in[0], f"fitting_input/fitting_input_batch{batch_idx}_netlayer{i}", relu.last_in[1].detach().cpu())

            class Container(torch.nn.Module):
                def __init__(self, my_values):
                    super().__init__()
                    for key in my_values:
                        setattr(self, key, my_values[key])

            c = torch.jit.script(Container(conv_input))
            c.save(f"{config.fitting_test_data_store_path}/conv_inputs_{batch_idx}.pt")

            # c = torch.jit.script(Container(full_input))
            # c.save(f"{config.fitting_test_data_store_path}/full_input_batch{batch_idx}.pt")
            #
            # c = torch.jit.script(Container(after_fixed_point))
            # c.save(f"{config.fitting_test_data_store_path}/after_fixed_point_batch{batch_idx}.pt")

    end_time = time.perf_counter()

    tensor_board_writer.add_scalar("10. batch_duration", end_time - start_time, epoch)


def test(model: gmc.model.Net, device: str, test_loader: torch.utils.data.DataLoader, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            step = epoch * len(test_loader) + (batch_id + 1)
            output = model(data, (tensor_board_writer, step))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    tensor_board_writer.add_scalar("99. mnist test loss", test_loss, epoch)
    tensor_board_writer.add_scalar("98. mnist test accuracy", 100. * correct / len(test_loader.dataset), epoch)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
    return correct / len(test_loader.dataset)


def experiment(device: str = 'cuda', desc_string: str = "", config: typing.Optional[Config] = None, ablation_name: str = ""):
    print(f"starting {desc_string}")
    # Training settings
    torch.manual_seed(0)
    # input_fitting.fit(config)

    train_loader = torch.utils.data.DataLoader(GmMnistDataSet(f'{config.produce_input_description()}/train_', begin=config.training_set_start, end=config.training_set_end), batch_size=config.batch_size,
                                               num_workers=config.num_dataloader_workers, shuffle=True)
    test_loader = torch.utils.data.DataLoader(GmMnistDataSet(f'{config.produce_input_description()}/test_', begin=config.test_set_start, end=config.test_set_end), batch_size=config.batch_size,
                                              num_workers=config.num_dataloader_workers)

    model = gmc.model.Net(learn_positions=config.learn_positions_after == 0,
                          learn_covariances=config.learn_covariances_after == 0,
                          config=config.model)
    model = model.to(device)

    kernel_optimiser = optim.Adam(model.parameters(), lr=config.kernel_learning_rate)
    weight_decay_optimiser = optim.SGD(model.parameters(), lr=(config.weight_decay_rate * config.kernel_learning_rate))
    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / f'tensorboard_{ablation_name}' / f'{desc_string}_{datetime.datetime.now().strftime("%m%d_%H%M")}')

    kernel_scheduler = optim.lr_scheduler.ReduceLROnPlateau(kernel_optimiser, mode="max", threshold=0.0002, factor=0.1, patience=5, cooldown=8, verbose=True, eps=1e-8)
    weight_decay_scheduler = optim.lr_scheduler.ReduceLROnPlateau(kernel_optimiser, mode="max", threshold=0.0002, factor=0.1, patience=5, cooldown=8, verbose=True, eps=1e-9)

    for epoch in range(config.n_epochs):
        model.set_position_learning(epoch >= config.learn_positions_after)
        model.set_covariance_learning(epoch >= config.learn_covariances_after)
        train(model, device, train_loader, kernel_optimiser=kernel_optimiser, weight_decay_optimiser=weight_decay_optimiser, epoch=epoch, tensor_board_writer=tensor_board_writer, config=config)
        test_loss = test(model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        kernel_scheduler.step(test_loss)
        weight_decay_scheduler.step(test_loss)
