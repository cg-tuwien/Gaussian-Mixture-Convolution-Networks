import torch.utils.tensorboard
import pygmvis

tensor_board_writer = torch.utils.tensorboard.SummaryWriter("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/tensorboard/pylibtest2")

# define render callback
def callback(epoch, data, frame):
    tensor_board_writer.add_image("a. ellipsoids" if (frame == 0) else "b. density", data, epoch, dataformats="HWC")
    tensor_board_writer.flush()

# Initialize
pygmvis.initialize(async=False,width=500,height=500)
pygmvis.set_callback(callback)
pygmvis.set_camera_auto(True)
pygmvis.set_pointcloud("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/chair_0030.off")
pygmvis.render("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30fix.ply", epoch=10)
pygmvis.finish()    #w aits for all commands to finish and frees data

tensor_board_writer.close()
