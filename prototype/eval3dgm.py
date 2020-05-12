import gm
import pointcloud
import torch

PC_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud/ModelNet10/chair/train/chair_0030.off"
#PC_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-lores/ModelNet10/chair/train/chair_0030.off"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30HR.ply"
GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/default-v2-C30HR-32684/pcgmm-0-19000.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/default-v2-C30HR-10000-constantweights/pcgmm-0-15000.ply"

print("Start")
pc = pointcloud.load_pc_from_off(PC_PATH).view(1, 1, -1, 3).cuda()
print("Read in PC")
mix = gm.read_gm_from_ply(GM_PATH, True).cuda()
print("Read in GM")
output = gm.evaluate(mix, pc).view(-1)
print("Calculating Loss")
loss = -torch.mean(torch.log(output + 0.001))
print("Loss is ", loss.item()) # WRONG

# TODO: Improve evaluation