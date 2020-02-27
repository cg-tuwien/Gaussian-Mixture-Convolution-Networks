from experiment_gm_fitting import test_dl_fitting
import gm_fitting
import experiment_gm_mnist


def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 64, 128, 128, n_output_gaussians * 10, n_output_gaussians * 10, n_output_gaussians * 20, n_output_gaussians * 20, n_output_gaussians * 25],
                                               [128, 128, 128, 64, 64, 64],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)

experiment_gm_mnist.experiment_alternating(device='cuda:0', n_epochs=100, desc_string="64-64-128-128-10n-10n-20n-20n-25n-a1-128-128-128-64-64-64",
                                           layer1_m2m_fitting=generate_fitting_module,
                                           layer2_m2m_fitting=generate_fitting_module,
                                           layer3_m2m_fitting=generate_fitting_module)
