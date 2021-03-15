
from pcfitting import data_loading
import torch

# scales a pc so that it's diagonal is 1

pc_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\fitpcs\n100000\bathtub_0001-1-original.off"
pc_out_path = r"D:\Simon\Studium\S-11 (WS19-20)\Diplomarbeit\data\dataset_diff_scales\fitpcs\n100000\bathtub_0001-2-half.off"

pc = data_loading.load_pc_from_off(pc_path)[0]

pmin = torch.min(pc, dim=0)[0]
pmax = torch.max(pc, dim=0)[0]
pext = pmax - pmin
pscale = torch.norm(pext)

# print("scale: ", pscale.item())

# pc_scaled = pc/pscale
pc_scaled = pc/2.0

print(torch.norm(torch.max(pc_scaled, dim=0)[0] - torch.min(pc_scaled, dim=0)[0]))

data_loading.write_pc_to_off(pc_out_path, pc_scaled)
