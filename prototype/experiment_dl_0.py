from experiment_gm_fitting import test_dl_fitting
import gm_fitting
import experiment_gm_mnist
import gm_modules

experiment_gm_mnist.experiment_alternating(device='cuda:0', n_epochs=100, desc_string="default",
                                           layer1_m2m_fitting=gm_modules.generate_default_fitting_module,
                                           layer2_m2m_fitting=gm_modules.generate_default_fitting_module,
                                           layer3_m2m_fitting=gm_modules.generate_default_fitting_module)
