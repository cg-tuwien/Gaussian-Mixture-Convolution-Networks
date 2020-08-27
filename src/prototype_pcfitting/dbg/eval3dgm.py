import torch
import pointcloud
import gmc.mixture as gm

#PC_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/TEST3.off"
PC_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud/ModelNet10/chair/train/chair_0030.off"
#PC_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-lores/ModelNet10/chair/train/chair_0030.off"
#PC_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/ModelNet10/pointcloud-hires/ModelNet10/chair/train/chair_0030.off"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30HR.ply"
GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30Ls3.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/new-C30HR-32684-w0.0005-PREINER/pcgmm-0-initial.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/new-C30HR-32684-w0.0005-PREINER/pcgmm-0-05000.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/new-C30HR-32684-P-RMS7-CVSep-W-0.0005-initeq/pcgmm-0-13000.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/default-v2-C30HR-32684/pcgmm-0-19000.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/TEST3-fixwe/pcgmm-0-43750.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/TEST3-PREINER3/pcgmm-0-initial.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/TEST3-PREINER3/pcgmm-0-109750.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/losstest2/pcgmm-0-initial.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/testloss7/pcgmm-0-00250.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/testloss7/pcgmmX-0-00250.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/gmc_net/gmc_net_data/models/default-v2-C30HR-10000-constantweights/pcgmm-0-15000.ply"
#GM_PATH = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/TEST3-preiner2.ply"
#PREINER GMS AND GMMS DO NOT RESULT IN THE SAME THING WTF
SCALE_DOWN = True
IS_MODEL = True

print("Start")
pc = pointcloud.load_pc_from_off(PC_PATH).view(1, 1, -1, 3).cuda()
print("Read in PC")
mix = gm.read_gm_from_ply(GM_PATH, IS_MODEL).cuda()
if SCALE_DOWN:
    bbmin = torch.min(pc, dim=2)[0]  # shape: (m, 3)
    bbmax = torch.max(pc, dim=2)[0]  # shape: (m, 3)
    extends = bbmax - bbmin  # shape: (m, 3)
    # Scale point clouds to [0,1] in the smallest dimension
    scale = torch.min(extends, dim=2)[0]  # shape: (m)
    scale = scale.view(1, 1, 1)  # shape: (m,1,1)
    scale2 = scale ** 2
    pc = pc / scale
    pc += 0.5
    scale = scale.view(1, 1, 1, 1)  # shape: (m,1,1,1)
    scale2 = scale2.view(1, 1, 1, 1, 1)  # shape: (m,1,1,1,1)
    positions = gm.positions(mix)
    positions /= scale
    positions += 0.5
    covariances = gm.covariances(mix)
    amplitudes = gm.weights(mix)
    pi = amplitudes * (covariances.det().sqrt() * 15.74960995)
    covariances /= scale2
    amplitudes = pi / (covariances.det().sqrt() * 15.74960995)
    mix = gm.pack_mixture(amplitudes, positions, covariances)
print("Read in GM")
#sample_point_idz = torch.randperm(pc.shape[2])[0:3000] #Shape: (s), where s is #samples
#sample_points = pc[:, :, sample_point_idz, :]  #shape: (m,s,3)
#output = gm.evaluate(mix, sample_points).view(-1)
output = gm.evaluate(mix, pc).view(-1)
# positions = gm.positions(mix)
# weights = gm.weights(mix)
# covariances = gm.covariances(mix)
# invcov = covariances.inverse().clone()
# mixinv = gm.pack_mixture(weights, positions, invcov)
# output = gm.evaluate_inversed(mixinv, pc).view(-1)
print("Calculating Loss")
loss = -torch.mean(torch.log(output + 0.001))
print("Loss is ", loss.item()) # WRONG

"""
Evaluation Results:
default-v2-C30HR-32684 on C30: 
-3.48120379447937
c_30HR.ply (Preiner) on C30:
-7.384815216064453
TEST3-PREINER3 on TEST3:
-5.967143535614014
"""
