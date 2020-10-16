import torch


class DummyPcGenerator:

    @staticmethod
    def generate_dummy_pc1():
        return torch.tensor([
            [0, 0, 0],
            [0.5, 0, 0],
            [-0.5, 0, 0],
            [0, 0.5, 0],
            [0, -0.5, 0],
            [0, 0, 0.5],
            [0, 0, -0.5]
        ]).view(1, -1, 3).cuda()
