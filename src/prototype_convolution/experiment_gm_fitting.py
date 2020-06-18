import time
import torch
import datetime
import torch.utils.tensorboard
from torch import Tensor
import config
import gmc.mixture as gm
import gm_fitting

def log(input: Tensor, output: Tensor, tensor_board_writer):
    device = input.device
    image_size = 80
    xv, yv = torch.meshgrid([torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device),
                             torch.arange(-1.2, 1.2, 2.4 / image_size, dtype=torch.float, device=device)])
    xes = torch.cat((xv.reshape(-1, 1), yv.reshape(-1, 1)), 1).view(1, 1, -1, 2)

    input_image = gm.evaluate(input.detach(), xes).view(-1, image_size, image_size)
    image_target = gm.evaluate_with_activation_fun(input.detach(), torch.zeros([1, gm.n_layers(input)], device=device), xes).view(-1, image_size, image_size)
    output_image = gm.evaluate(output.detach(), xes).view(-1, image_size, image_size)
    gm_fitting.Sampler.log_images(tensor_board_writer,
                                  f"input target prediction",
                                  [input_image.transpose(0, 1).reshape(image_size, -1),
                                   image_target.transpose(0, 1).reshape(image_size, -1),
                                   output_image.transpose(0, 1).reshape(image_size, -1)],
                                  None, [-0.5, 2])


tensor_board_writer = torch.utils.tensorboard.SummaryWriter(config.data_base_path / 'tensorboard' / f'fitting_{datetime.datetime.now().strftime("%d_%H-%M-%S")}')

for batch_idx in range(10):
    start_time = time.perf_counter()
    for layer_id in range(3):
        m = gm.load(f"fitting_input/fitting_input_netlayer{layer_id}_batch{batch_idx}")[0]
        m = m.cuda()
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

        weights = gm.weights(m)
        positions = gm.positions(m)
        covariances = gm.covariances(m)
        negative_weights = weights.where(weights <= 0, torch.zeros(1, device=device))
        positive_weights = weights.where(weights > 0, torch.zeros(1, device=device))
        negative_m = gm.pack_mixture(negative_weights, positions, covariances)
        positive_m = gm.pack_mixture(positive_weights, positions, covariances)
        negative_eval = gm.evaluate(negative_m, positions)
        positive_eval = gm.evaluate(positive_m, positions)
        new_weights_factor = torch.max(torch.zeros(1, device=device),
                                       torch.ones(1, device=device) + (negative_eval - 0.0001) / (positive_eval + 0.0001))
        new_weights = new_weights_factor * positive_weights

        m_relu = gm.pack_mixture(new_weights, positions, covariances)
        # log(m, m_relu, tensor_board_writer)
        print(f"{batch_idx}/{layer_id}")

    end_time = time.perf_counter()
    print(f"time for 3 layers: {end_time - start_time}")
    print("============")
