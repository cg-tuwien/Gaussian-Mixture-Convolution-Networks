import torch.utils.tensorboard
import numpy as np
import pygmvis

"""
Demonstrated features
* Single threaded rendering
* Automatic camera positioning
* Reading pcs/gms from files
* Getting render result as return result (only possible in single threaded mode, when no callback is registered)
"""

tensor_board_writer = torch.utils.tensorboard.SummaryWriter("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/tensorboard/pylibtest2")

vis = pygmvis.create_visualizer(async=False, width=500, height=500)
vis.set_camera_auto(True)
vis.set_pointclouds_from_paths(["D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/chair_0030.off",
                                "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/chair_0128HR.off"])
vis.set_gaussian_mixtures_from_paths(["D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30fix.ply",
                                      "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c1228.ply"],
                                     isgmm=True)
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
