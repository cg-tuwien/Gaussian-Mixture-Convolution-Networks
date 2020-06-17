import torch.utils.tensorboard
import numpy as np
import gm

"""
Demonstrated features
* Single threaded rendering
* Automatic camera positioning
* Reading pcs/gms from files
* Getting render result as return result (only possible in single threaded mode, when no callback is registered)
"""

tensor_board_writer = torch.utils.tensorboard.SummaryWriter("../data/tensorboard/test_sync/")

vis = gm.vis.create_visualizer(False, width=500, height=500)
vis.set_camera_auto(True)
vis.set_pointclouds_from_paths(["../cpp_modules/gm_vis/da-gm-1/data/chair_0030.off",
                               "../cpp_modules/gm_vis/da-gm-1/data/chair_0129.off"])
vis.set_gaussian_mixtures_from_paths(["../cpp_modules/gm_vis/da-gm-1/data/c_30fix.ply",
                                     "../cpp_modules/gm_vis/da-gm-1/data/c1228.ply"],
                                    isgmm=True)
vis.set_pointclouds_from_paths(["../cpp_modules/gm_vis/da-gm-1/data/chair_0048.off"])
# mixture = gm.generate_random_mixtures(n_batch=1, n_layers=1, n_components=1, n_dims=3,
#                                               pos_radius=0.1, cov_radius=0.01,
#                                               weight_min=0, weight_max=1, device='cpu')
# vis.set_gaussian_mixtures(mixture, isgmm=False)

res = vis.render(11)
assert len(res.shape) == 5
assert res.shape[0] == 2 # amount of gms
assert res.shape[1] == 2 # amount of images per gm
tensor_board_writer.add_image("c30: a. ellipsoids", res[0,0,:,:,:], 11, dataformats="HWC")
tensor_board_writer.add_image("c30: b. density", res[0,1,:,:,:], 11, dataformats="HWC")
tensor_board_writer.add_image("c128: a. ellipsoids", res[1,0,:,:,:], 11, dataformats="HWC")
tensor_board_writer.add_image("c128: b. density", res[1,1,:,:,:], 11, dataformats="HWC")

vis.finish()

tensor_board_writer.close()
