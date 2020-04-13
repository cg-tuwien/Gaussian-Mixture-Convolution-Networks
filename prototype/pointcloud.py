import torch

def load_pc_from_off(path: str) -> torch.Tensor:
    file = open(path, "r")
    if 'OFF' != file.readline().strip():
        raise("Not a valid OFF header!")
    n_points = int(file.readline().strip().split(" ")[0])
    points = [[[float(s) for s in file.readline().strip().split(" ") if s != ''] for pt in range(n_points)]]
    return torch.tensor(points, dtype=torch.float32)