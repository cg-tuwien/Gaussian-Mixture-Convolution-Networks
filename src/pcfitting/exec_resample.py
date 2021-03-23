import os
from pcfitting import data_loading, GMSampler

gmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/gmms/"
out_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_diff_scales/resampled/"

n_points = 10000

# training_name = '210306-01-EmEckPre'
training_name = '210312-EMepsvar'
# training_name = '210312-ng64'

gmm_path += training_name
out_path += training_name

for root, dirs, files in os.walk(gmm_path):
    for name in files:
        if name.lower().endswith(".gma.ply"):
            path = os.path.join(root, name)
            relpath = path[len(gmm_path) + 1:]
            print(relpath)
            gm = data_loading.read_gm_from_ply(path, False)
            pc = GMSampler.sample(gm, n_points)
            data_loading.write_pc_to_off(os.path.join(out_path, relpath + ".off"), pc)