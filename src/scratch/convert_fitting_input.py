import torch.nn
import torch.jit

import gmc.mixture


class Container(torch.nn.Module):
    def __init__(self, my_values):
        super().__init__()
        for key in my_values:
            setattr(self, key, my_values[key])


for batch in range(10):
    d = dict()
    for layer in range(3):
        m = gmc.mixture.load(f"fitting_input/fitting_input_batch{batch}_netlayer{layer}")[0]
        d[f"{layer}"] = m

    c = torch.jit.script(Container(d))
    c.save(f"/home/madam/Documents/work/tuw/gmc_net/data/fitting_input/fitting_input_batch{batch}.pt")
