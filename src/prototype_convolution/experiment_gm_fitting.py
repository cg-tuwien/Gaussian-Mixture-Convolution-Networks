import time
import torch
import datetime
import torch.utils.tensorboard
from torch import Tensor

# import update_syspath
import prototype_convolution.config as config
import gmc.mixture as gm
import prototype_convolution.fitting_net as fitting_net
import prototype_convolution.fitting_em as fitting_em


def log(target: Tensor, target_bias, fitting: Tensor, fitting_bias: Tensor, label: str, tensor_board_writer):
    device = target.device
    image_size = 80
    xv, yv = torch.meshgrid([torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device),
                             torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)

    input_image = gm.evaluate(target.detach(), xes).view(-1, image_size, image_size) + target_bias.view(-1).unsqueeze(-1).unsqueeze(-1)
    image_target = gm.evaluate_with_activation_fun(target.detach(), target_bias, xes).view(-1, image_size, image_size)
    output_image = gm.evaluate(fitting.detach(), xes).view(-1, image_size, image_size) + fitting_bias.view(-1).unsqueeze(-1).unsqueeze(-1)
    fitting_net.Sampler.log_images(tensor_board_writer,
                                   f"{label} input target prediction",
                                   [input_image.transpose(0, 1).reshape(image_size, -1),
                                    image_target.transpose(0, 1).reshape(image_size, -1),
                                    output_image.transpose(0, 1).reshape(image_size, -1)],
                                   None, [-0.5, 2])


tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'fitting_{datetime.datetime.now().strftime("%d_%H-%M-%S")}')

torch.zeros(1).cuda()

for batch_idx in range(9, 10): # was 10
    start_time = time.perf_counter()
    for layer_id in range(3): # was 3
        m = gm.load(f"fitting_input/fitting_input_netlayer{layer_id}_batch{batch_idx}")[0]
        # m = m[0:1, 1:2, 0:20]
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

        # m.requires_grad = True
        bias_g = torch.zeros([1, gm.n_layers(m)], device=device, requires_grad=True)
        bias = bias_g #- .015
        fitting_g, new_bias = fitting_em.relu(m, bias)
        # fitting_g.requires_grad = True
        (gm.integrate(fitting_g)).sum().backward()
        print(bias_g.grad)
        fitting = fitting_em.mhem_algorithm(fitting_g, n_fitting_components=15, n_iterations=1)
        log(m, bias, fitting, new_bias, f"l{layer_id}.", tensor_board_writer)
        # log(m, bias, fitting, new_bias, f"l{layer_id},", tensor_board_writer)
        print(f"{batch_idx}/{layer_id}")

    end_time = time.perf_counter()
    tensor_board_writer.flush()
    print(f"time for 3 layers: {end_time - start_time}")
    print("============")
