import random

import torch.utils.data

import gmc.mixture as gm
import qm9.xyz_reader as xyz_reader
from qm9.config import Config


class DataSet(torch.utils.data.Dataset):
    def __init__(self, config: Config, start_index: int, end_index: int):
        data = xyz_reader.read_dataset(config)
        random.Random(0).shuffle(data)
        data = data[start_index:end_index]
        gm_data = list()
        self.targets = list()
        for d in data:
            gm_data.append(d.as_gaussian_mixture())
            self.targets.append(torch.tensor(d.properties[config.inference_on], dtype=torch.float))

        gm_data = torch.cat(gm_data, 0)
        # # needs an edit to include only molecules that have an atom for a given molecule (something like torch.sum() / torch.sum(weight > 0) )
        # x_abs_integral = gm.integrate(gm_data).abs()
        # x_abs_integral = torch.mean(x_abs_integral, dim=0, keepdim=True)
        # x_abs_integral = torch.max(x_abs_integral, torch.tensor(0.01))
        # new_weights = gm.weights(gm_data) / torch.unsqueeze(x_abs_integral, dim=-1)
        # gm_data = gm.pack_mixture(new_weights, gm.positions(gm_data), gm.covariances(gm_data))
        self.data = gm_data

        # // test reading again. how much memory is taken? can we convert directly to torch.tensor (on cpu)?
        # // implement storage, __len__ ,and get_item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mixture = self.data[index:index+1, :, :, :]

        assert len(mixture.shape) == 4

        assert gm.n_batch(mixture) == 1
        assert gm.n_layers(mixture) > 0
        assert gm.n_components(mixture) > 0
        assert gm.n_dimensions(mixture) == 3

        return mixture[0], self.targets[index]
