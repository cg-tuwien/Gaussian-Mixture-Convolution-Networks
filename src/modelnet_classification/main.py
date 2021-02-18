from __future__ import print_function
import datetime
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
import modelnet_classification.model
import gmc.fitting
import modelnet_classification.config as Config

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


class ModelNetDataSet(torch.utils.data.Dataset):
    def __init__(self, data_base_path: pathlib.Path, category_list_file: pathlib.Path, sample_names_file: pathlib.Path):
        self.data_base_path = data_base_path
        category_names = []
        with open(category_list_file) as inFile:
            for line in inFile:
                category_names.append(line.replace('\n', ''))

        self.sample_names = []
        self.sample_labels = []
        with open(sample_names_file) as inFile:
            for line in inFile:
                line = line.replace('\n', '')
                category = line[:-5]
                self.sample_names.append(category + '/' + line)
                if category not in category_names:
                    raise ValueError('Unknown category ' + category)
                self.sample_labels.append(category_names.index(category))

    def __len__(self):
        return len(self.sample_names)

    def __getitem__(self, index):
        file_path = f"{self.data_base_path}/{self.sample_names[index]}.torch"
        mixture = torch.load(file_path)

        assert len(mixture.shape) == 4

        assert gmc.mixture.n_batch(mixture) == 1
        assert gmc.mixture.n_layers(mixture) == 1
        assert gmc.mixture.n_components(mixture) > 0
        assert gmc.mixture.n_dimensions(mixture) == 3

        return mixture[0], torch.tensor(self.sample_labels[index])


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


def train(args, model: modelnet_classification.model.Net, device: torch.device, train_loader: torch.utils.data.DataLoader,
          kernel_optimiser: Optimizer, weight_decay_optimiser: Optimizer, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter, config: Config):
    model.train()
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

        if step % args.log_interval == 0:
            pred = output.detach().argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            tensor_board_writer.add_scalar("00. training loss", training_loss.item(), step)
            tensor_board_writer.add_scalar("01. training accuracy", 100 * correct / len(data), step)
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

            render_debug_images_to_tensorboard(model, step, tensor_board_writer, config)

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

                fp_fitting, fp_const, _ = gmc.fitting.fixed_point_and_bvh_mhem(relu.last_in[0].detach(), relu.last_in[1].detach(), n_components=-1)
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


def test(args, model: modelnet_classification.model.Net, device: torch.device, test_loader: torch.utils.data.DataLoader, epoch: int, tensor_board_writer: torch.utils.tensorboard.SummaryWriter):
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
    tensor_board_writer.add_scalar("98. mnist test accuracy", 100. * correct / len(test_loader.dataset), epoch)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')


def experiment(device: str = 'cuda', n_epochs: int = 20, kernel_learning_rate: float = 0.001, log_interval: int = 100,
               learn_positions_after: int = 0, learn_covariances_after: int = 0, desc_string: str = "", config: Config = None):
    # Training settings
    torch.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(ModelNetDataSet(config.modelnet_data_path, config.modelnet_category_list_file, config.modelnet_training_sample_names_file), batch_size=config.batch_size, num_workers=config.num_dataloader_workers, shuffle=True, drop_last=True)
    test_loader  = torch.utils.data.DataLoader(ModelNetDataSet(config.modelnet_data_path, config.modelnet_category_list_file, config.modelnet_test_sample_names_file),     batch_size=config.batch_size, num_workers=config.num_dataloader_workers)

    model = modelnet_classification.model.Net(name=desc_string,
                                              learn_positions=learn_positions_after == 0,
                                              learn_covariances=learn_covariances_after == 0,
                                              config=config)
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
    weight_decay_optimiser = optim.SGD(model.parameters(), lr=(0.05*kernel_learning_rate))
    tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'{desc_string}_{datetime.datetime.now().strftime("%m%d_%H%M")}')

    # scheduler = StepLR(kernel_optimiser, step_size=1, gamma=args.gamma)

    for epoch in range(n_epochs):
        model.set_position_learning(epoch >= learn_positions_after)
        model.set_covariance_learning(epoch >= learn_covariances_after)
        train(args, model, device, train_loader, kernel_optimiser=kernel_optimiser, weight_decay_optimiser=weight_decay_optimiser, epoch=epoch, tensor_board_writer=tensor_board_writer, config=config)
        test(args, model, device, test_loader, epoch, tensor_board_writer=tensor_board_writer)
        # scheduler.step()

        if args.save_model:
            model.save_model()
            model.save_fitting_optimiser_state()
