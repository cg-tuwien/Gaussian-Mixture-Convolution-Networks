from experiment_gm_fitting import test_dl_fitting
import gm_fitting
import experiment_gm_mnist
import gm_modules


def generate_fitting_module_inner(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    n_dimensions = 2
    return gm_fitting.PointNetWithParallelMLPs([64, 128, 256, 512, 512, n_output_gaussians * 25],
                                               [256, 256, 256, 256, 256, 128],
                                               n_output_gaussians=n_output_gaussians,
                                               n_dims=n_dimensions,
                                               aggregations=1, batch_norm=True)


def generate_fitting_module(n_input_gaussians: int, n_output_gaussians: int) -> gm_fitting.Net:
    assert n_output_gaussians > 0
    return gm_fitting.SpaceSubdivider(generate_fitting_module_inner,
                                      n_input_gaussians=n_input_gaussians,
                                      n_fitting_module_out_gaussians=6,
                                      n_output_gaussians=n_output_gaussians)


experiment_gm_mnist.experiment_combined_loss(device='cuda:0', n_epochs=100, desc_string="cmbnd_subdiv6p_LRe-3_learnPos",
                                             learning_rate=0.001,
                                             layer1_m2m_fitting=generate_fitting_module,
                                             layer2_m2m_fitting=generate_fitting_module,
                                             layer3_m2m_fitting=generate_fitting_module,
                                             log_interval=50)
