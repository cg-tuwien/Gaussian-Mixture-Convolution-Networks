from pcfitting.gm_sampler import GMSampler
import gmc.mixture as mixture
import torch
from pcfitting.data_loading import write_pc_to_off

positions = torch.tensor([[[
    [-30.0, -30.0, -30.0],
    [30.0, 30.0, 30.0]
]]])

covariances = torch.tensor([[[
    [[1.0,0,0],[0,1.0,0],[0,0,1.0]],
    [[1.0,0,0],[0,1.0,0],[0,0,1.0]]
]]])

weights = torch.tensor([[[
    0.4, 0.6
]]])

pc = GMSampler.sampleGMM(mixture.pack_mixture(weights, positions, covariances), 50)
write_pc_to_off("C:\\Users\\SimonFraiss\\Desktop\\test.off", pc)