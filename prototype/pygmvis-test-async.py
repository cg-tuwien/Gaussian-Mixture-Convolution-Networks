import torch
import torch.utils.tensorboard
import numpy as np
import pointcloud
import gm
import pygmvis

"""
Demonstrated features
* Asynchronous rendering
* Manual camera positioning
* Passing pcs/gms directly
* Accessing render result via callback
"""

tensor_board_writer = torch.utils.tensorboard.SummaryWriter("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/tensorboard/pylibtest2")

def callback(epoch, data, gmidx, frame):
    tensor_board_writer.add_image(("c30: " if (gmidx == 0) else "c128: ") + ("a. ellipsoids" if (frame == 0) else "b. density"), data, epoch, dataformats="HWC")
    tensor_board_writer.flush()

pc1 = pointcloud.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/chair_0030HR.off")
pc2 = pointcloud.load_pc_from_off("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/chair_0128HR.off")
pcs = torch.empty((2,pc1.shape[1],pc1.shape[2]))
pcs[0,:,:] = pc1[0,:,:]
pcs[1,:,:] = pc2[0,:,:]

mixture = gm.generate_random_mixtures(n_batch=2, n_components=2000, n_dims=3,
                                          pos_radius=40, cov_radius=5.0 / (2000**(1/3)),
                                          weight_min=0, weight_max=1)

vis = pygmvis.create_visualizer(async=True, width=500, height=500)
vis.set_view_matrix(np.array([[0.707107,0,-0.707107,0.0113297],[-0.5,0.707107,-0.5,0.00418091],[0.5,0.707107,0.5,-100.877],[0,0,0,1]]))
vis.set_pointclouds(pcs)
vis.set_gaussian_mixtures(mixture, isgmm=False)
vis.set_density_rendering(True, pygmvis.GMDensityRenderMode.ADDITIVE_ACC_PROJECTED)
vis.set_callback(callback)
vis.render(11)
vis.finish()
tensor_board_writer.close()
