import time
import math
import datetime
import typing

import torch
import torch.utils.tensorboard
from torch import Tensor

# import update_syspath
import prototype_convolution.config as config
import gmc.mixture as gm
import prototype_convolution.fitting_net as fitting_net
import prototype_convolution.fitting as fitting


def log(target: Tensor, target_bias, fitting_steps: typing.List[Tensor], fitting: Tensor, fitting_bias: Tensor, label: str, tensor_board_writer):
    device = target.device
    image_size = 80
    xv, yv = torch.meshgrid([torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device),
                             torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)

    input_image = gm.evaluate(target.detach(), xes).view(-1, image_size, image_size) + target_bias.view(-1).unsqueeze(-1).unsqueeze(-1)
    image_target = gm.evaluate_with_activation_fun(target.detach(), target_bias, xes).view(-1, image_size, image_size)
    rendered_steps = list()
    for m in fitting_steps:
        rendering = gm.evaluate(m.detach(), xes).view(-1, image_size, image_size) + fitting_bias.view(-1).unsqueeze(-1).unsqueeze(-1)
        rendered_steps.append(rendering.transpose(0, 1).reshape(image_size, -1))
    fitting = gm.evaluate(fitting.detach(), xes).view(-1, image_size, image_size) + fitting_bias.view(-1).unsqueeze(-1).unsqueeze(-1)
    fitting_net.Sampler.log_images(tensor_board_writer,
                                   f"{label} input target prediction",
                                   [input_image.transpose(0, 1).reshape(image_size, -1),
                                    image_target.transpose(0, 1).reshape(image_size, -1),
                                    *rendered_steps,
                                    fitting.transpose(0, 1).reshape(image_size, -1)],
                                   None, [-0.5, 2])


N_FITTED_GAUSSIANS = 20

tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'fitting{N_FITTED_GAUSSIANS}_{datetime.datetime.now().strftime("%d_%H-%M-%S")}')

torch.zeros(1).cuda()

measurements = dict()
measurements_n = dict()


def add_measurement(name: str, value: float):
    if name not in measurements:
        measurements[name] = value
        measurements_n[name] = 1
    else:
        measurements[name] += value
        measurements_n[name] += 1


def generate_random_sampling(m: Tensor, n: int) -> Tensor:
    covariance_adjustment = torch.sqrt(torch.diagonal(gm.covariances(m.detach()), dim1=-2, dim2=-1))
    position_max, _ = torch.max(gm.positions(m.detach()) + covariance_adjustment, dim=2, keepdim=True)
    position_min, _ = torch.min(gm.positions(m.detach()) - covariance_adjustment, dim=2, keepdim=True)
    sampling = torch.rand(gm.n_batch(m), gm.n_layers(m), n, gm.n_dimensions(m), device=m.device)
    sampling *= position_max - position_min
    sampling += position_min
    return sampling


for batch_idx in range(0, 1):  # max 10
    start_time = time.perf_counter()
    for layer_id in range(3):
        # for bias in [-1, -0.5, -0.1, -0.005, 0, 0.005, 0.1, 0.5, 1]:
        for bias in [-0.5, 0.0, 0.5]:
            m = gm.load(f"fitting_input/fitting_input_batch{batch_idx}_netlayer{layer_id}")[0]
            m = m[0:5, :, :, :]
            # m = torch.tensor([[[[1, -0.8, -0.8, 0.25, 0.04, 0.04, 0.05], [1, 0.8, 0.8, 0.05, -0.04, -0.04, 0.25]]]])
            m = m.cuda()
            # m.requires_grad = True
            device = m.device
            n_batch = gm.n_batch(m)
            n_layers = gm.n_layers(m)
            n_components = gm.n_components(m)
            m, _, _ = gm.normalise(m, torch.zeros([1, n_layers], device=device))
            # m.requires_grad = True
            # sorted_indices = torch.argsort(gm.weights(m.detach()))
            # sorted_m = mat_tools.my_index_select(m, sorted_indices)
            # n_negative_m = (gm.weights(m).detach() <= 0).sum(-1)
            # negative_m = sorted_m[:, :, :, :n_negative_m]
            # positive_m = sorted_m[:, :, :, n_negative_m:]

            bias_tensor = torch.zeros([n_batch, n_layers], device=device, requires_grad=False) + bias

            eval_xes = generate_random_sampling(m, 5000)
            eval_gt = gm.evaluate_with_activation_fun(m, bias_tensor, eval_xes)

            # m.requires_grad = True
            start = time.perf_counter()
            fitted_m, new_bias, fitting_steps = fitting.fixed_point_and_mhem(m, bias_tensor, N_FITTED_GAUSSIANS)
            add_measurement(f"time[layer{layer_id}]", time.perf_counter() - start)

            eval_fit = gm.evaluate(fitted_m, eval_xes) + new_bias.unsqueeze(-1)
            add_measurement(f"mse [layer{layer_id}]", ((eval_fit - eval_gt)**2).mean().item())
            add_measurement(f"mse [bias{bias}]", ((eval_fit - eval_gt)**2).mean().item())

            if batch_idx == 0 and bias in [-0.5, 0.0, 0.5]:
                log(m, bias_tensor, fitting_steps, fitted_m, new_bias, f"l{layer_id}_b{bias},", tensor_board_writer)
        print(f"batch {batch_idx} / layer {layer_id}")

tensor_board_writer.flush()
print("============")

printing_times = False
for name, measurement_sum in sorted(measurements.items()):
    if name.startswith("time") and printing_times is False:
        print("------------")
        printing_times = True
    if printing_times is False:
        print(f"r{name}: {math.sqrt(measurement_sum / measurements_n[name])}")
    else:
        print(f"{name}: {measurement_sum/measurements_n[name]}")
