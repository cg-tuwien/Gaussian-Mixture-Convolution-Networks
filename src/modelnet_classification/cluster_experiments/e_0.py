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

n_classes = 10
n_g = 128
fitting_name = f"fpsm_n{n_g}_eps10"
# pcfit.run(fitting_name,
#           EMGenerator(n_gaussians=n_g, initialization_method="fpsmax", termination_criterion=MaxIterationTerminationCriterion(0), em_step_points_subbatchsize=10000, eps=1e-5),
#           gengmm_path=tmp_gmm_base_path,
#           batch_size=25)
c: Config = Config(gmms_fitting=fitting_name, gengmm_path=tmp_gmm_base_path, n_classes=n_classes)
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
c.model.bn_place = ModelConfig.BN_PLACE_AFTER_RELU
c.model.relu_config.fitting_method = fitting.fixed_point_only
c.model.convolution_config.learnable_radius = False
c.log_tensorboard_renderings = False
c.model.dropout = 0.05
c.log_interval = 200
c.batch_size = 50

kernel_radius = 3.0

#c.n_epochs = 2

for seed in (1, 2):
    c.model.layers = [Layer(8, kernel_radius, 512, tuple()),
                      Layer(16, kernel_radius, 256, tuple()),
                      Layer(32, kernel_radius, 128, tuple()),
                      Layer(64, kernel_radius, 64, tuple()),
                      Layer(128, kernel_radius, 32, tuple()),
                      Layer(n_classes, kernel_radius, 16, tuple())]
    main.experiment(device=device, desc_string=f"bs{c.batch_size}_{fitting_name}_{c.produce_description()}", config=c, ablation_name=f"modelnet{n_classes}_abl_fitting", random_seed=seed)
    
    c.model.layers = [Layer(8, kernel_radius, 64, tuple()),
                      Layer(16, kernel_radius, 32, tuple()),
                      Layer(32, kernel_radius, 16, tuple()),
                      Layer(64, kernel_radius, 8, tuple()),
                      Layer(128, kernel_radius, 4, tuple()),
                      Layer(n_classes, kernel_radius, 2, tuple())]
    main.experiment(device=device, desc_string=f"bs{c.batch_size}_{fitting_name}_{c.produce_description()}", config=c, ablation_name=f"modelnet{n_classes}_abl_fitting", random_seed=seed)
    
    c.model.layers = [Layer(8, kernel_radius, 128, tuple()),
                      Layer(16, kernel_radius, 64, tuple()),
                      Layer(32, kernel_radius, 32, tuple()),
                      Layer(64, kernel_radius, 16, tuple()),
                      Layer(128, kernel_radius, 8, tuple()),
                      Layer(n_classes, kernel_radius, 4, tuple())]
    main.experiment(device=device, desc_string=f"bs{c.batch_size}_{fitting_name}_{c.produce_description()}", config=c, ablation_name=f"modelnet{n_classes}_abl_fitting", random_seed=seed)
    
    c.model.layers = [Layer(8, kernel_radius, 256, tuple()),
                      Layer(16, kernel_radius, 128, tuple()),
                      Layer(32, kernel_radius, 64, tuple()),
                      Layer(64, kernel_radius, 32, tuple()),
                      Layer(128, kernel_radius, 16, tuple()),
                      Layer(n_classes, kernel_radius, 8, tuple())]
    main.experiment(device=device, desc_string=f"bs{c.batch_size}_{fitting_name}_{c.produce_description()}", config=c, ablation_name=f"modelnet{n_classes}_abl_fitting", random_seed=seed)
