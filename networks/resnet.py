"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F

import torch
import torch.nn as nn


class nblock_resnet(nn.Module):
    def __init__(self, tag, inc=2, onc=2, n_residual_blocks=2, conv3d=False):
        super(nblock_resnet, self).__init__()
        self.conv3d=conv3d
        self.tag = tag
        n_interm_c = 128

        init_layer = nn.Conv2d(inc, n_interm_c, 1, 1, 0)
        model = [init_layer]

        # residual blocks
        for _ in range(n_residual_blocks):
            block = residual_block(n_interm_c)
            model = model + [block]

        # 1x1 convolutions
        conv_layer0 = nn.Conv2d(n_interm_c, n_interm_c, 1, 1, 0)
        act0 = nn.LeakyReLU(0.1, inplace=True)
        conv_layer1 = nn.Conv2d(n_interm_c, n_interm_c, 1, 1, 0)
        act1 = nn.LeakyReLU(0.1, inplace=True)
        conv_layer2 = nn.Conv2d(n_interm_c, onc, 1, 1, 0)
        act2 = nn.LeakyReLU(0.1, inplace=True)

        model = model + [conv_layer0, act0, conv_layer1, act1, conv_layer2, act2]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        patch_means = torch.mean(input, dim=(2, 3), keepdim=True)
        input -= patch_means
        return patch_means + self.model(input)


class residual_block(nn.Module):
    def __init__(self, nc):
        super(residual_block, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 3, 1, 1),
            nn.BatchNorm2d(nc),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, input):
        return torch.clamp(input + self.model(input), min=-1, max=1)