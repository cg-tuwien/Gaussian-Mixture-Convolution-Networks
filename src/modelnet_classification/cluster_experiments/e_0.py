import sys
import modelnet_classification.main as main
from modelnet_classification.config import Config
from gmc.model import Layer, Config as ModelConfig
import gmc.fitting as fitting

from pcfitting import MaxIterationTerminationCriterion, RelChangeTerminationCriterion
from pcfitting.generators import EMGenerator, PreinerGenerator, EckartGeneratorSP
import cluster_experiments.pc_fit as pcfit

# device = list(sys.argv)[1]
device = "cuda"

# tmp_gmm_base_path = "/scratch/acelarek/gmms/"
tmp_gmm_base_path = None


def fitting_config(n_g, layer):
    conv_fit_n = int(n_g * 3 // 2**layer)
    fit_n = int(n_g * 4 // 2**layer)
    if fit_n <= 16:
        conv_fit_n = fit_n
        fit_n = -1
    return conv_fit_n, fit_n


# for n_classes in (10, 40):
#     for n_g in (64, 128, 256):
n_classes = 10
n_g = 128
#running
for kernel_radius in (1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75):
    fitting_name = f"fpsm_n{n_g}_eps10"
    # pcfit.run(fitting_name,
    #           EMGenerator(n_gaussians=n_g, initialization_method="fpsmax", termination_criterion=MaxIterationTerminationCriterion(0), em_step_points_subbatchsize=10000, eps=1e-5),
    #           gengmm_path=tmp_gmm_base_path,
    #           batch_size=25)

    c: Config = Config(gmms_fitting=fitting_name, gengmm_path=tmp_gmm_base_path, n_classes=n_classes)
    c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
    c.model.relu_config.fitting_method = fitting.splitter_and_fixed_point
    c.log_tensorboard_renderings = False
    c.model.dropout = 0.0
    c.log_interval = 200
    c.batch_size = 50
    c.model.layers = [Layer(8, kernel_radius, *fitting_config(n_g, 1)),
                      Layer(16, kernel_radius, *fitting_config(n_g, 2)),
                      Layer(32, kernel_radius, *fitting_config(n_g, 3)),
                      Layer(64, kernel_radius, *fitting_config(n_g, 4)),
                      Layer(128, kernel_radius, *fitting_config(n_g, 5)),
                      Layer(n_classes, kernel_radius, *fitting_config(n_g, 6))]
    main.experiment(device=device, desc_string=f"bs{c.batch_size}_{fitting_name}_{c.produce_description()}", config=c, ablation_name=f"modelnet{n_classes}")


# #todo
# kernel_radius = select best run
# n_g = select best (128 or 256)
# for dropout in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
#     fitting_name = f"fpsm_n{n_g}_eps10"
#     # pcfit.run(fitting_name,
#     #           EMGenerator(n_gaussians=n_g, initialization_method="fpsmax", termination_criterion=MaxIterationTerminationCriterion(0), em_step_points_subbatchsize=10000, eps=1e-5),
#     #           gengmm_path=tmp_gmm_base_path,
#     #           batch_size=25)
#
#     c: Config = Config(gmms_fitting=fitting_name, gengmm_path=tmp_gmm_base_path, n_classes=n_classes)
#     c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
#     c.model.relu_config.fitting_method = fitting.splitter_and_fixed_point
#     c.log_tensorboard_renderings = False
#     c.model.dropout = dropout
#     c.log_interval = 200
#     c.batch_size = 50
#     c.model.layers = [Layer(8, kernel_radius, *fitting_config(n_g, 1)),
#                       Layer(16, kernel_radius, *fitting_config(n_g, 2)),
#                       Layer(32, kernel_radius, *fitting_config(n_g, 3)),
#                       Layer(64, kernel_radius, *fitting_config(n_g, 4)),
#                       Layer(128, kernel_radius, *fitting_config(n_g, 5)),
#                       Layer(n_classes, kernel_radius, *fitting_config(n_g, 6))]
#     main.experiment(device=device, desc_string=f"bs{c.batch_size}_{fitting_name}_{c.produce_description()}", config=c, ablation_name=f"modelnet{n_classes}")