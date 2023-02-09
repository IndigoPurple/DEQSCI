"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch import nn
from torch.nn import functional as F
from utils.spectral_norm import conv_spectral_norm
import utils.spectral_norm_chen as chen
import utils.spectral_norm as sn


class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans, out_chans, drop_prob, conv3d=False):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob
        layers = []

        if not conv3d:
            layers.append(conv_spectral_norm(nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
                                            sigma=1.0, out_channels=out_chans))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            layers.append(conv_spectral_norm(nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
                                            sigma=1.0,  out_channels=out_chans))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            layers.append(nn.Conv3d(in_chans, out_chans, kernel_size=(3,3,3), padding=(1,1,1), bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            layers.append(nn.Conv3d(out_chans, out_chans, kernel_size=(3,3,3), padding=(1,1,1), bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose layers followed by
    instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans, out_chans, conv3d=False):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        layers = []

        if not conv3d:
            layers.append(conv_spectral_norm(nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2, bias=False),
                                            sigma=1.0, out_channels=out_chans, leakflag=True))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            layers.append(nn.ConvTranspose3d(in_chans, out_chans, kernel_size=(3,2,2), stride=(1,2,2), padding=(1,0,0), bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        return self.layers(input)

class ZerosNet(nn.Module):
    def __init__(self):
        super(ZerosNet, self).__init__()

    def forward(self, input):
        return input*0.0 + 0.0

class UnetModel(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, tag):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.tag = tag

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        self.up_conv += [
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )]

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        output = torch.clamp(output, -1,1)
        return output

class UnetNorm(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, tag):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.tag = tag

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch)]
        layers = []
        layers.append(ConvBlock(ch * 2, ch, drop_prob))
        layers.append(conv_spectral_norm(nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
                                         sigma=1.0, out_channels=self.out_chans, kernelsize=1))
        self.up_conv += [nn.Sequential(*layers)]

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # Padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # Padding bottom
            if sum(padding) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output

class Unet3D(nn.Module):
    """
    PyTorch implementation of a U-Net model.
    This is based on:
        Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-net: Convolutional networks
        for biomedical image segmentation. In International Conference on Medical image
        computing and computer-assisted intervention, pages 234–241. Springer, 2015.
    """

    def __init__(self, in_chans, out_chans, chans, num_pool_layers, drop_prob, tag, conv3d=True):
        """
        Args:
            in_chans (int): Number of channels in the input to the U-Net model.
            out_chans (int): Number of channels in the output to the U-Net model.
            chans (int): Number of output channels of the first convolution layer.
            num_pool_layers (int): Number of down-sampling and up-sampling layers.
            drop_prob (float): Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.tag = tag
        self.conv3d = conv3d

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob, conv3d=conv3d)])
        ch = chans
        for i in range(num_pool_layers - 1):
            self.down_sample_layers += [ConvBlock(ch, ch * 2, drop_prob, conv3d=conv3d)]
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob, conv3d=conv3d)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for i in range(num_pool_layers - 1):
            self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch, conv3d=conv3d)]
            self.up_conv += [ConvBlock(ch * 2, ch, drop_prob, conv3d=conv3d)]
            ch //= 2

        self.up_transpose_conv += [TransposeConvBlock(ch * 2, ch, conv3d=conv3d)]
        layers = []
        layers.append(ConvBlock(ch * 2, ch, drop_prob, conv3d=conv3d))
        # if not conv3d:
        #     layers.append(conv_spectral_norm(nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
        #                                  sigma=1.0, out_channels=self.out_chans, kernelsize=1))
        # else:
        layers.append(nn.Conv3d(ch, self.out_chans, kernel_size=1, stride=1))

        self.up_conv += [nn.Sequential(*layers)]

        # self.init_weights(init_type='xavier')

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, height, width]
        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, height, width]
        """
        stack = []
        output = input

        # Apply down-sampling layers
        for i, layer in enumerate(self.down_sample_layers):
            output = layer(output)
            stack.append(output)
            if not self.conv3d:
                output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            else:
                output = F.avg_pool3d(output, kernel_size=(1,2,2), stride=(1,2,2), padding=0)

        output = self.conv(output)

        # Apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # Reflect pad on the right/botton if needed to handle odd input dimensions.
            if not self.conv3d:
                padding = [0, 0, 0, 0]
                if output.shape[-1] != downsample_layer.shape[-1]:
                    padding[1] = 1  # Padding right
                if output.shape[-2] != downsample_layer.shape[-2]:
                    padding[3] = 1  # Padding bottom
                if sum(padding) != 0:
                    output = F.pad(output, padding, "reflect")
            else:
                assert output.shape[-1] == downsample_layer.shape[-1] and output.shape[-2] == downsample_layer.shape[-2], "currently only support 32x2^n resolution"

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)
        return output


    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    torch.nn.init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    torch.nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


# class DnCNN(nn.Module):
#     def __init__(self, channels, num_of_layers=17, lip=1.0):
#         super(DnCNN, self).__init__()
#         self.conv3d = False
#         kernel_size = 3
#         padding = 1
#         features = 64
#         layers = []
#
#         channels = (channels if channels!=1 else 3)
#         layers.append(chen.spectral_norm(nn.Conv2d(
#                 in_channels=channels,
#                 out_channels=features,
#                 kernel_size=kernel_size,
#                 padding=padding,
#                 bias=False
#             )))
#         layers.append(nn.ReLU(inplace=True))
#         for _ in range(num_of_layers-2):
#             layers.append(chen.spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
#             layers.append(nn.BatchNorm2d(features))
#             layers.append(nn.ReLU(inplace=True))
#         layers.append(chen.spectral_norm(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)))
#
#         # yaping
#         # layers.append(
#         #     nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
#         #               bias=False))
#         # layers.append(nn.ReLU(inplace=True))
#         # for _ in range(num_of_layers - 2):
#         #     layers.append(
#         #         nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
#         #                   bias=False))
#         #     layers.append(nn.BatchNorm2d(features))
#         #     layers.append(nn.ReLU(inplace=True))
#         # layers.append(
#         #     nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
#         #               bias=False))
#
#         # yaping
#         # layers.append(sn.spectral_norm(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
#         # layers.append(nn.ReLU(inplace=True))
#         # for _ in range(num_of_layers-2):
#         #     layers.append(sn.spectral_norm(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)))
#         #     layers.append(nn.BatchNorm2d(features))
#         #     layers.append(nn.ReLU(inplace=True))
#         # layers.append(sn.spectral_norm(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False)))
#
#         self.dncnn = nn.Sequential(*layers)
#     def forward(self, x):
#         modify_flag = (x.size(1)==1)
#         if modify_flag:
#             x = x.expand(-1, 3, -1, -1)
#         out = self.dncnn(x)
#         if modify_flag:
#             out = out[:,0:1,:,:]
#         return out
