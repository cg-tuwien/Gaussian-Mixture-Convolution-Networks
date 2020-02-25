from experiment_gm_fitting import test_dl_fitting
import gm_fitting


def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 64, n_output_gaussians * 5, n_output_gaussians * 10, n_output_gaussians * 25],
                                               [50, 40, 30, 20, 10],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)
test_dl_fitting(generate_fitting_module, device='cpu')

def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 64, n_output_gaussians * 5, n_output_gaussians * 10, n_output_gaussians * 25],
                                               [100, 80, 60, 40, 20],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)
test_dl_fitting(generate_fitting_module, device='cpu')

def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 64, n_output_gaussians * 5, n_output_gaussians * 10, n_output_gaussians * 25],
                                               [200, 100, 40],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)
test_dl_fitting(generate_fitting_module, device='cpu')
