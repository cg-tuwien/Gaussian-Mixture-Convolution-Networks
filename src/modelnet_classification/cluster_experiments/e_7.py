import sys

import torch

import modelnet_classification.main as main
from modelnet_classification.config import Config
from gmc.model import Layer, Config as ModelConfig

from pcfitting import MaxIterationTerminationCriterion, RelChangeTerminationCriterion
from pcfitting.generators import EMGenerator, PreinerGenerator, EckartGeneratorSP

import cluster_experiments.pc_fit as pcfit

# device = list(sys.argv)[1]
device = "cuda"

# tmp_gmm_base_path = "/scratch/acelarek/gmms/e0"
tmp_gmm_base_path = None

nGs_per_node = 5
nLevels = 3
fitting_name = f"Eckart_{nGs_per_node}_{nLevels}"
pcfit.run(fitting_name,
          EckartGeneratorSP(n_gaussians_per_node=nGs_per_node, n_levels=nLevels, termination_criterion=RelChangeTerminationCriterion(0.1, 20), initialization_method="fpsmax", partition_threshold=0.1, m_step_points_subbatchsize=10000,
                            e_step_pair_subbatchsize=5120000, dtype=torch.float32, eps=1e-5),  #
          gengmm_path=tmp_gmm_base_path,
          batch_size=1)


c: Config = Config(gmms_fitting=fitting_name, gengmm_path=tmp_gmm_base_path, n_classes=10)
c.model.bn_type = ModelConfig.BN_TYPE_COVARIANCE
c.model.dropout = 0.0
c.log_tensorboard_renderings = False
c.n_epochs = 121

# network size
c.model.layers = [Layer(8, 2.5, 64),
                  Layer(16, 2.5, 32),
                  Layer(32, 2.5, 16),
                  Layer(10, 2.5, -1)]
# c.model.mlp = (-1, 40)

main.experiment(device=device, desc_string=f"fit_{fitting_name}_{c.produce_description()}", config=c)
