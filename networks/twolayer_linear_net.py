"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F


class LinearNet(nn.Module):

    def __init__(self, input_size, bottleneck_size, output_size):
        super().__init__()
        # self.linear_layer = nn.Linear(input_size, output_size)
        # self.linear_layer2 = nn.Linear(output_size, output_size)
        self.network = nn.Sequential(
            nn.Linear(input_size, bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.ReLU(),
            nn.Linear(bottleneck_size, output_size),
            nn.Tanh()
        )
        self.network.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
            m.bias.data.fill_(0.01)

    def forward(self, input):
        input_shape = input.shape
        output = self.network(torch.flatten(input, start_dim=1))
        output = torch.reshape(output, shape=input_shape)

        return output
