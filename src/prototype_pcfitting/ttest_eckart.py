from prototype_pcfitting.generators.level_scaler import LevelScaler
import torch

pcbatch = torch.tensor([[
    [1.0, 1.2, 0.8],
    [2.1, 2.8, 3.4],
    [0.9, 1.8, 3.0],
    [1.9, 2.9, 3.0],
    [0.9, 2.4, 2.5]
]]).cuda()
parent_per_point = torch.tensor([[0, 0, 0, 1, 1]])
relevant_parents = torch.tensor([[0, 1]])

scaler = LevelScaler()
scaler.set_pointcloud(pcbatch, parent_per_point, relevant_parents)
scaled_down_pc = scaler.scale_down_pc(pcbatch)

gmpositions = torch.tensor([[[
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0]
]]]).cuda()
gmcovariances = torch.eye(3, 3).view(1, 1, 1, 3, 3).expand(1, 1, 8, 3, 3).cuda()
gmweights = torch.zeros(1, 1, 8).cuda()
gmweights[:, :, :] = 1 / 4.0

gmwn, gmpn, gmcn = scaler.scale_up_gmm_wpc(gmweights, gmpositions, gmcovariances)

pass