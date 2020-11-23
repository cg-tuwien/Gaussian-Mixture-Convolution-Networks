from prototype_pcfitting.generators.level_scaler import LevelScaler
from prototype_pcfitting.generators.level_scaler2 import LevelScaler2
from prototype_pcfitting import RelChangeTerminationCriterion
import torch

pcbatch = torch.tensor([[
    [1.0, 1.2, 0.8],
    [2.1, 2.8, 3.4],
    [0.9, 1.8, 3.0],
    [1.9, 2.9, 3.0],
    [0.9, 2.4, 2.5],
    [9.9, 9.8, 9.7]
]]).cuda()
parent_per_point = torch.tensor([[0, 0, 0, 1, 1, 2]])
relevant_parents = torch.tensor([[0, 1]])

# scaler = LevelScaler()
# scaler.set_pointcloud(pcbatch, parent_per_point, relevant_parents)
# scaled_down_pc = scaler.scale_down_pc(pcbatch)
#
# gmpositions = torch.tensor([[[
#     [0.0, 0.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [1.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0],
#     [0.0, 0.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [1.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0]
# ]]]).cuda()
# gmcovariances = torch.eye(3, 3).view(1, 1, 1, 3, 3).expand(1, 1, 8, 3, 3).cuda()
# gmweights = torch.zeros(1, 1, 8).cuda()
# gmweights[:, :, :] = 1 / 4.0
#
# gmwn, gmpn, gmcn = scaler.scale_up_gmm_wpc(gmweights, gmpositions, gmcovariances)

# --------------------------------------------------------------------------------------

# scaler = LevelScaler2()
# scaler.set_pointcloud(pcbatch, parent_per_point, 4)
# scaled_down_pc = scaler.scale_down_pc(pcbatch)
#
# gmpositions = torch.tensor([[[
#     [0.0, 0.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [1.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0],
#     [0.0, 0.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [1.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0],
#     [0.0, 0.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [1.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0],
#     [0.0, 0.0, 0.0],
#     [1.0, 0.0, 0.0],
#     [1.0, 1.0, 0.0],
#     [0.0, 0.0, 1.0]
# ]]]).cuda()
# gmcovariances = torch.eye(3, 3).view(1, 1, 1, 3, 3).expand(1, 1, 16, 3, 3).cuda()
# gmweights = torch.zeros(1, 1, 16).cuda()
# gmweights[:, :, :] = 1 / 4.0
#
# gmwn, gmpn, gmcn = scaler.scale_up_gmm_wpc(gmweights, gmpositions, gmcovariances)

# --------------------------------------------------------------------------------------

criterion = RelChangeTerminationCriterion(0.1, 5)
print(criterion.may_continue(1, torch.tensor([2.0, 3.0, 5.0])))
print(criterion.may_continue(2, torch.tensor([1.5, 3.99, 4.8])))
print(criterion.may_continue(3, torch.tensor([1.3, 3.98, 4.6])))
print(criterion.may_continue(4, torch.tensor([1.4, 3.97, 4.5])))
print(criterion.may_continue(5, torch.tensor([1.0, 3.96, 4.45])))
print(criterion.may_continue(6, torch.tensor([0.98, 3.95, 4.44])))
print(criterion.may_continue(7, torch.tensor([0.97, 6.7, 4.43])))
print(criterion.may_continue(8, torch.tensor([0.96, 6.7, 4.42])))
print(criterion.may_continue(9, torch.tensor([0.95, 6.7, 4.42])))
print(criterion.may_continue(10, torch.tensor([9.94, 6.7, 4.415])))
criterion.reset()
print(criterion.may_continue(1, torch.tensor([2.0, 3.0, 5.0])))
print(criterion.may_continue(2, torch.tensor([1.5, 3.99, 4.8])))
print(criterion.may_continue(3, torch.tensor([1.3, 3.98, 4.6])))
print(criterion.may_continue(4, torch.tensor([1.4, 3.97, 4.5])))
print(criterion.may_continue(5, torch.tensor([1.0, 3.96, 4.45])))
print(criterion.may_continue(6, torch.tensor([0.98, 3.95, 4.44])))
print(criterion.may_continue(7, torch.tensor([0.97, 6.7, 4.43])))
print(criterion.may_continue(8, torch.tensor([0.96, 6.7, 4.42])))
print(criterion.may_continue(9, torch.tensor([0.95, 6.7, 4.42])))
print(criterion.may_continue(10, torch.tensor([9.94, 6.7, 4.415])))
pass