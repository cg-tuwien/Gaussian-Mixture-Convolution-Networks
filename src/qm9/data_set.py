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
        self.data = list()
        self.targets = list()
        for d in data:
            self.data.append(d.as_gaussian_mixture())
            self.targets.append(torch.tensor(d.properties[config.inference_on], dtype=torch.float))

        # // test reading again. how much memory is taken? can we convert directly to torch.tensor (on cpu)?
        # // implement storage, __len__ ,and get_item

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        mixture = self.data[index]

        assert len(mixture.shape) == 4

        assert gm.n_batch(mixture) == 1
        assert gm.n_layers(mixture) > 0
        assert gm.n_components(mixture) > 0
        assert gm.n_dimensions(mixture) == 3

        return mixture[0], self.targets[index]
