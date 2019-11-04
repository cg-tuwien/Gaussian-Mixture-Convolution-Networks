import gm
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as nprnd

from gm import Mixture
from torch import Tensor


def to_image(image: Tensor, n_components: int) -> Mixture:
    assert len(image.size()) == 2
    assert n_components > 0
    width = image.size()[1]
    height = image.size()[0]

    mixture = gm.generate_random_mixtures(n_components, 2,
                                          0.5,
                                          5 * min(width, height) / math.sqrt(n_components),
                                          0, 1)
    mixture.positions += 0.5
    mixture.positions *= torch.tensor([[width], [height]], dtype=torch.float).expand_as(mixture.positions)
    
    xv, yv = torch.meshgrid([torch.arange(0, width, 1, dtype=torch.float), torch.arange(0, height, 1, dtype=torch.float)])
    xes = torch.cat((xv.reshape(1, -1), yv.reshape(1, -1)), 0)
    values = image.view(-1)
    
    selected_components = mixture.max_component(xes).view(xv.size()[0], xv.size()[1])
    new_mixture = gm.generate_null_mixture(n_components, 2)
    new_mixture_value_sum = torch.zeros(new_mixture.number_of_components())
    #for i in range(values.size()[0]):
        
        #new_mixture.positions
    
    
    #dbg_components = .numpy()
    #plt.imshow(dbg_components)
    #plt.show()

    return mixture


image:np.ndarray = plt.imread("/home/madam/cloud/Photos/_20160325_234800.JPG.jpg")
image = image.mean(axis = 2) / 255
mixture = to_image(torch.tensor(image), 100)
mixture.debug_show(0 - 100, 0 - 100, image.shape[1] + 100, image.shape[0] + 100, 4)