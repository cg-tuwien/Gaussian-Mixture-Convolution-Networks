import torch
import os

# pc is returned as a 3d tensor with shape [1,n,3] on cpu
def load_pc_from_off(path: str) -> torch.Tensor:
    file = open(path, "r")
    if 'OFF' != file.readline().strip():
        raise("Not a valid OFF header!")
    n_points = int(file.readline().strip().split(" ")[0])
    points = [[[float(s) for s in file.readline().strip().split(" ") if s != ''] for pt in range(n_points)]]
    file.close()
    return torch.tensor(points, dtype=torch.float32)

# pc is given as a 2d tensor with shape [n, 3]
def write_pc_to_off(path: str, pc: torch.Tensor):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    file = open(path, "w+")
    file.write("OFF\n")
    file.write(f"{pc.shape[0]} 0 0\n")
    for point in pc:
        file.write(f"{point[0]} {point[1]} {point[2]}\n")
    file.close()

def sample(pcbatch: torch.Tensor, n_sample_points: int) -> torch.Tensor:
    batch_size, point_count, _ = pcbatch.shape
    sample_point_idz = torch.randperm(point_count)[0:n_sample_points]   # Shape: (s)
    sample_points = pcbatch[:, sample_point_idz, :]                     # Shape: (m,s,3)
    return sample_points.view(batch_size, 1, min(point_count, n_sample_points), 3)