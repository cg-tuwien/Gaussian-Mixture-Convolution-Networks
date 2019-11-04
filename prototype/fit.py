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


# https://en.wikipedia.org/w/index.php?title=Algorithms_for_calculating_variance&oldid=922055093#Online
# https://stats.stackexchange.com/questions/61225/correct-equation-for-weighted-unbiased-sample-covariance
# essentially, there are two weighting schemes. one of them adds up to 1, the other is int (repeats / frequency).
# i don't know whether it's possible to handle float weights that don't add up to 1 in an online algorithm
# this method returns the same as np.cov(xes, fweights=w)
def my_funs(data: np.ndarray, weights: np.ndarray, dims: int):
    w_sum = 0
    mean = np.zeros(dims)
    cov = np.zeros((dims, dims))
    for i in range(data.shape[1]):
        w = weights[i]
        w_sum += w
        x = data[:, i]
        dx = x - mean
        mean += w/w_sum * dx
        dx.shape = (dims, 1)
        # update scheme in wikipedia uses a different w_sum for x and y. the term (1-w/wsum) corrects that
        cov += w * (1 - w/w_sum) * dx @ dx.T
    cov /= (w_sum - 1)
    
    print(f"mean = {mean}, \ncov=\n{cov}")


dims = 4
xes = nprnd.rand(dims, 20)
w = (nprnd.rand(20)*20).astype(int)
#w = np.ones(20)
print(f"numpy mean = {np.average(xes, axis=1, weights=w)}, \n numpy cov=\n{np.cov(xes, fweights=w)}")
my_funs(xes, w, dims)


#image: np.ndarray = plt.imread("/home/madam/cloud/Photos/toy_small.jpg")
#image = image.mean(axis = 2) / 255
#mixture = to_image(torch.tensor(image), 100)
##mixture.debug_show(0 - 100, 0 - 100, image.shape[1] + 100, image.shape[0] + 100, 4)
#mixture.debug_show(0, 0, image.shape[1], image.shape[0], 1)

a = 3
