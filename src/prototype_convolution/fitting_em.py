import time

import torch
import torch.Tensor as Tensor

import gmc.mixture as gm


def relu(mixture: Tensor) -> Tensor:
    weights = gm.weights(mixture)
    positions = gm.positions(mixture)
    covariances = gm.covariances(mixture)
    device = mixture.device

    negative_weights = weights.where(weights <= 0, torch.zeros(1, device=device))
    positive_weights = weights.where(weights > 0, torch.zeros(1, device=device))
    negative_m = gm.pack_mixture(negative_weights, positions, covariances)
    positive_m = gm.pack_mixture(positive_weights, positions, covariances)
    negative_eval = gm.evaluate(negative_m, positions)
    positive_eval = gm.evaluate(positive_m, positions)
    new_weights_factor = torch.max(torch.zeros(1, device=device),
                                   torch.ones(1, device=device) + (negative_eval - 0.0001) / (positive_eval + 0.0001))
    new_weights = new_weights_factor * positive_weights

    return gm.pack_mixture(new_weights, positions, covariances)

def max_component(fitting: Tensor, target: Tensor) -> Tensor:
    

def em_algorithm(mixture: Tensor, n_components: int, n_iterations: int) -> Tensor:
    # todo test (after moving from Mixture class to Tensor data
    assert gm.is_valid_mixture(mixture)
    assert n_components > 0
    n_batch = gm.n_batch(mixture)
    n_layers = gm.n_layers(mixture)
    n_dims = gm.n_dimensions(mixture)
    device = mixture.device

    target = relu(mixture)

    mixture = gm.generate_random_mixtures(n_batch, n_layers, n_components, n_dims,
                                          device=device)


    print("starting expectation maximisation")
    for k in range(n_iterations):
        print(f"classifying..")
        selected_components = mixture.max_component(xes)
        print(f"updating..")
        new_mixture = gm.generate_null_mixture(1, n_components, 2, device=mixture.device())
        n_pixels = torch.zeros(n_components, device=new_mixture.device())
        for i in range(values.size()[0]):
            w = values[i]
            x = xes[0, i, :]
            c = selected_components[i]
            n_pixels[c] += 1
            new_mixture.weights[0, c] += w.float()
            dx = x - new_mixture.positions[0, c, :]
            new_mixture.positions[0, c, :] += w / new_mixture.weights[0, c] * dx
            new_mixture.covariances[0, c, :, :] += w * (1 - w / new_mixture.weights[0, c]) * (dx.view(-1, 1) @ dx.view(1, -1))

        for j in range(new_mixture.n_components()):
            if new_mixture.weights[0, j] > 1:
                new_mixture.covariances[0, j, :, :] /= new_mixture.weights[0, j] - 1
            if n_pixels[j] > 0:
                new_mixture.weights[0, j] /= n_pixels[j]

        # minimum variance
        new_mixture.covariances[0, :, :, :] += torch.eye(new_mixture.n_dimensions(), device=new_mixture.device()) * 0.1
        print(f"iterations {k} finished")

        # new_mixture.debug_show(0, 0, width, height, 1)
        mixture = new_mixture

    fitting_end = time.time()
    print(f"fitting time: {fitting_end - fitting_start}")
    return mixture