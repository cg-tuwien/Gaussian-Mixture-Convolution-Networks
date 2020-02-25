from experiment_gm_fitting import test_dl_fitting
import gm_fitting

def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 64, 64, 512, 512, 512, n_output_gaussians * 25],
                                               [100, 100, 50, 50, 20],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)
test_dl_fitting(generate_fitting_module, device='cuda')


def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 64, 64, n_output_gaussians * 20, n_output_gaussians * 20, n_output_gaussians * 20, n_output_gaussians * 25],
                                               [100, 100, 50, 50, 20],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)
test_dl_fitting(generate_fitting_module, device='cuda')


def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 64, 64, 512, 512, 512, n_output_gaussians * 25],
                                               [200, 200, 100, 100, 40],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)
test_dl_fitting(generate_fitting_module, device='cuda')







