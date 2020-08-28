import gmc.mixture
from prototype_pcfitting import Scaler, pointcloud

pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/pointclouds/n20000/train/chair_0001.off"
pc = pointcloud.load_pc_from_off(pc_path).cuda()
scaler = Scaler()
scaler.set_pointcloud_batch(pc)

#gma = gmc.mixture.read_gm_from_ply("D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_test/" + \
#    "gmms/200827-02-eval/GD1/train/chair_0001.off.gma.ply", False).cuda()
gma = gmc.mixture.read_gm_from_ply('D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/da-gm-1/da-gm-1/data/c_30fix.ply', True).cuda()

gmm = gmc.mixture.convert_amplitudes_to_priors(gma)
gma_test = gmc.mixture.convert_priors_to_amplitudes(gmm)

gma_down = scaler.scale_down_gm(gma)
gmm_down = scaler.scale_down_gmm(gmm)
gma_down_test = gmc.mixture.convert_priors_to_amplitudes(gmm_down)
gmm_down_test = gmc.mixture.convert_amplitudes_to_priors(gma_down)
gma_down_test2 = gmc.mixture.convert_priors_to_amplitudes(gmm_down_test)
gmm_down_test2 = gmc.mixture.convert_amplitudes_to_priors(gma_down_test)

print("Done")