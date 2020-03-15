from experiment_gm_fitting import test_dl_fitting
import gm_fitting
import experiment_gm_mnist

# experiment_gm_mnist.experiment_alternating(device='cuda:0', n_epochs=30, desc_string="learn_all",
#                                            learn_covariances=False, learn_positions=False)

experiment_gm_mnist.experiment_alternating(device='cuda:0', n_epochs=70, desc_string="learn_all",
                                           learn_covariances=True, learn_positions=True)
