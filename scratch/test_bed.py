# -*- coding: utf-8 -*-
"""
Spyder Editor

estimate the centres of fuzzy circles with sub-pixel precision
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np


def draw_random_samples(n, width, height):
    radius = 0.05
    coords = torch.rand(n, 2).cuda()
    
    yv, xv = torch.meshgrid([torch.arange(0, height, dtype=torch.float32).cuda(), torch.arange(0, width, dtype=torch.float32).cuda()])
    xv = (xv / width).reshape(1, height, width).repeat(n, 1, 1)
    yv = (yv / height).reshape(1, height, width).repeat(n, 1, 1)
    
    xv = (xv - coords[:, 0].reshape(-1, 1, 1).expand_as(xv)) / radius
    yv = (yv - coords[:, 1].reshape(-1, 1, 1).expand_as(yv)) / radius
    
    canvas = 5 * (1 - torch.sqrt(xv * xv + yv * yv))
    canvas = 1 / (1 + torch.exp(-canvas))
    
    return coords, canvas

#x = draw_sample((4, 6), 4, (20, 8))
#torch.set_printoptions(linewidth=200)
#print(x)
#print(x.view(-1).size())

WIDTH = 100    
HEIGHT = 100  

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(WIDTH * HEIGHT, WIDTH * HEIGHT // 2)
        self.fc2 = nn.Linear(WIDTH * HEIGHT // 2, WIDTH * HEIGHT // 16)
#        self.fc3 = nn.Linear(WIDTH * HEIGHT // 4, WIDTH * HEIGHT // 8)
        self.fc4 = nn.Linear(WIDTH * HEIGHT // 16, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
#        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#        # If the size is a square you can only specify a single number
#        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
#        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net().cuda()
for parameter in net.parameters():
    print (f"parameter: {parameter.shape}")

criterion = nn.MSELoss()
optimiser = optim.Adam(net.parameters(), lr=0.000025)
print(net)


for i in range(2000):
    target, input_ = draw_random_samples(256, WIDTH, HEIGHT)
    optimiser.zero_grad()
    output = net(input_)
    loss = criterion(output, target)
    loss.backward()
    optimiser.step()
    if i % 10 == 0:
        print(f"iteration {i}: loss {loss}")

target, input_ = draw_random_samples(10, WIDTH, HEIGHT)
output = net(input_)
print(f"target={target}")
print(f"output={output}")
print(f"diff={output-target}")

