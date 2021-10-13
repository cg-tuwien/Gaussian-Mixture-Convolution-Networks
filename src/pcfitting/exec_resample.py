import os
from pcfitting import data_loading, GMSampler

gmm_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_plane/gmms/"
out_path = "D:/Simon/Studium/S-11 (WS19-20)/Diplomarbeit/data/dataset_plane/resampled/"

single_file_name = "plane0-original.off.gma.ply"

n_points = 100000

gmm = False

# training_name = "210406-EmEckPre"
# training_name = "210405-Plane1G-small"
# training_name = '210306-01-EmEckPre'
# training_name = '210312-EMepsvar'
# training_name = '210312-ng64'
# training_name = "210413-1-preinercheck"
# training_name = "210421-em579"
training_name = "210923-01-irr/EM510K"

gmm_path += training_name
out_path += "n" + str(n_points) + "/" + training_name

for root, dirs, files in os.walk(gmm_path):
    for name in files:
        if name.lower().endswith(".gmm.ply" if gmm else ".gma.ply") and (single_file_name is None or name.lower() == single_file_name):
            path = os.path.join(root, name)
            relpath = path[len(gmm_path) + 1:]
            print(relpath)
            gm = data_loading.read_gm_from_ply(path, gmm)
            pc = GMSampler.sampleGM_ext(gm, n_points)
            data_loading.write_pc_to_off(os.path.join(out_path, relpath + ".off"), pc)