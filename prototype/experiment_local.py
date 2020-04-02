from experiment_gm_fitting import test_dl_fitting
import gm_fitting
import experiment_gm_mnist


def generate_fitting_module_inner(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, n_output_gaussians * 25],
                                               [128],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)


def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    return gm_fitting.SpaceSubdivider(generate_fitting_module_inner,
                                      n_input_gaussians=n_input_gaussians,
                                      n_fitting_module_out_gaussians=6,
                                      n_output_gaussians=n_output_gaussians)


# experiment_gm_mnist.experiment_alternating(device='cuda:0', n_epochs=15, desc_string="learn_all", learning_rate=0.001,
#                                            learn_covariances=False, learn_positions=False)

experiment_gm_mnist.experiment_probabalistic(device='cuda:0', n_epochs=70, desc_string="test", kernel_learning_rate=0.0005, fitting_learning_rate=0.001,
                                           layer1_m2m_fitting=generate_fitting_module,
                                           layer2_m2m_fitting=generate_fitting_module,
                                           layer3_m2m_fitting=generate_fitting_module,
                                           learn_covariances_after=15, learn_positions_after=15)
