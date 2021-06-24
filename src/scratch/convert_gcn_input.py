import os
import sys

import torch
import numpy as np
import gmc.inout

for root, subFolders, files in os.walk("/home/madam/Downloads/tmp"):
    for filename in files:
        if not filename.endswith(".torch"):
            continue

        file_path = os.path.join(root, filename)
        data = torch.load(file_path)
        data_np = data.view(-1, 13).numpy()
        np.savez_compressed(file_path[:-len(".torch")], mixture=data_np)
