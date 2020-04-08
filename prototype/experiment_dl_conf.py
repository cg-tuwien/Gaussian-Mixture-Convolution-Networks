import gm_fitting
import experiment_gm_mnist
import sys

learning_rate_kernels = 0.0005
learning_rate_fitting = 0.001
log_interval = 1000


def generate_fitting_module_XS(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 128, n_output_gaussians * 25],
                                               [128, 64],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)


def generate_fitting_module_S(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 128, 256, n_output_gaussians * 25],
                                               [256, 128, 64],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)


def generate_fitting_module_M(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([32, 64, 128, 256, n_output_gaussians * 32],
                                               [256, 128, 64, 32],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)


def generate_subdivide_fitting_module(n_gaussians_per_fitting_module, fitting_module):
    def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
        assert n_output_gaussians > 0
        return gm_fitting.SpaceSubdivider(fitting_module,
                                          n_input_gaussians=n_input_gaussians,
                                          n_fitting_module_out_gaussians=n_gaussians_per_fitting_module,
                                          n_output_gaussians=n_output_gaussians)

    return generate_fitting_module
