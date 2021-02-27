
from pcfitting import data_loading
import torch

# scales a pc so that it's diagonal is 1

pc_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bed_0003.off"
pc_out_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_vartest/pointclouds/n100000/bed_0003_scaled.off"

pc = data_loading.load_pc_from_off(pc_path)[0]

pmin = torch.min(pc, dim=0)[0]
pmax = torch.max(pc, dim=0)[0]
pext = pmax - pmin
pscale = torch.norm(pext)

print("scale: ", pscale.item())

pc_scaled = pc/pscale

print(torch.norm(torch.max(pc_scaled, dim=0)[0] - torch.min(pc_scaled, dim=0)[0]))

data_loading.write_pc_to_off(pc_out_path, pc_scaled)
