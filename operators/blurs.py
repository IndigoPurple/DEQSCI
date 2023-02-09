import numpy as np
import numbers
import math
import cv2
import torch
import torch.nn.functional as torchfunc
from operators.operator import LinearOperator

class GaussianBlur(LinearOperator):
    def __init__(self, sigma, kernel_size=5, n_channels=3, n_spatial_dimensions = 2):
        super(GaussianBlur, self).__init__()
        self.groups = n_channels
        if isinstance(kernel_size, numbers.Number):
            self.padding = int(math.floor(kernel_size/2))
            kernel_size = [kernel_size] * n_spatial_dimensions
        else:
            print('KERNEL SIZE MUST BE A SINGLE INTEGER - RECTANGULAR KERNELS NOT SUPPORTED AT THIS TIME')
            exit()
        self.gaussian_kernel = torch.nn.Parameter(self.create_gaussian_kernel(sigma, kernel_size, n_channels),
                                                  requires_grad=False)

    def create_gaussian_kernel(self, sigma, kernel_size, n_channels):
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, mgrid in zip(kernel_size, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) / sigma) ** 2 / 2)

        # Make sure norm of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(n_channels, *[1] * (kernel.dim() - 1))
        return kernel

    def forward(self, x):
        return torchfunc.conv2d(x, weight=self.gaussian_kernel, groups=self.groups, padding=self.padding)

    def adjoint(self, x):
        return torchfunc.conv2d(x, weight=self.gaussian_kernel, groups=self.groups, padding=self.padding)

class SingleAngleMotionBlur(LinearOperator):
    def __init__(self, angle, kernel_size=5, n_channels=3, n_spatial_dimensions = 2):
        super(SingleAngleMotionBlur, self).__init__()
        self.groups = n_channels
        self.blur_kernel = torch.nn.Parameter(self.create_motionblur_kernel(angle, kernel_size, n_channels),
                                              requires_grad=False)
        if isinstance(kernel_size, numbers.Number):
            self.padding = int(math.floor(kernel_size/2))
            kernel_size = [kernel_size] * n_spatial_dimensions
        else:
            print('KERNEL SIZE MUST BE A SINGLE INTEGER - RECTANGULAR KERNELS NOT SUPPORTED AT THIS TIME')
            exit()

    def create_motionblur_kernel(self, angle, kernel_size, n_channels):
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[(kernel_size - 1) // 2, :] = np.ones(kernel_size, dtype=np.float32)
        kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5),
                                                                angle, 1.0), (kernel_size, kernel_size))
        kernel = torch.tensor(kernel, dtype=torch.float32)
        # Make sure norm of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(n_channels, *[1] * (kernel.dim() - 1))
        return kernel

    def forward(self, x):
        convolution_weight = self.blur_kernel
        return torchfunc.conv2d(x, weight=convolution_weight, groups=self.groups, padding=self.padding)

    def adjoint(self, x):
        convolution_weight = torch.transpose(self.blur_kernel, dim0=2, dim1=3)
        return torchfunc.conv2d(x, weight=convolution_weight, groups=self.groups, padding=self.padding)