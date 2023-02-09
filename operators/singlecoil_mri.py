import torch, numbers, math
import torch.nn as nn
import torch.nn.functional as torchfunc
from operators.operator import LinearOperator



import numpy as np
import torch


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def apply_mask(data, mask_func, seed=None, padding=None):
    """
    Subsample given k-space by multiplying with a mask.
    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.
    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, :padding[0]] = 0
        mask[:, :, padding[1]:] = 0 # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0 # The + 0.0 removes the sign of the zeros
    return masked_data, mask


def mask_center(x, mask_from, mask_to):
    # b, c, h, w, two = x.shape
    mask = torch.zeros_like(x)
    mask[:, :, :, mask_from:mask_to] = x[:, :, :, mask_from:mask_to]
    return mask


def complex_mul(x, y):
    assert x.shape[-1] == y.shape[-1] == 2
    re = x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1]
    im = x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]
    return torch.stack((re, im), dim=-1)


def complex_conj(x):
    assert x.shape[-1] == 2
    return torch.stack((x[..., 0], -x[..., 1]), dim=-1)


def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data

def dft_matrix(N, mask):
    learnable_parameters = torch.arange(0,N, dtype=torch.float32)
    learnable_parameters.requires_grad_(True)
    mask_vec = fftshift(mask[0, :], dim=0)
    mask_vec = mask_vec > 0
    mask_vec = mask_vec.squeeze()
    masked_params = torch.masked_select(learnable_parameters, mask_vec)
    normalizer = np.sqrt(N)

    ii, jj = torch.meshgrid(masked_params, torch.arange(0,N, dtype=torch.float32))

    W = torch.exp(-2.0 * np.pi * 1j * ii*jj / N) / normalizer

    return W

def onedfft(data, dim):
    # data = ifftshift(data, dim=dim)
    dim_size = data.shape[dim]
    for ii in range(dim_size):
        if dim==1:
            data[:,ii,:] = torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=0, norm="ortho")
        else:
            data[ii, :, :] = torch.fft.fftn(  # type: ignore
            torch.view_as_complex(data), dim=1, norm="ortho")
    # data = ifftshift(data, dim=dim)
    return data

def onedifft(data, dim):
    # data = ifftshift(data, dim=dim)
    dim_size = data.shape[dim]
    for ii in range(dim_size):
        if dim==1:
            data[:,ii,:] = torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=0, norm="ortho")
        else:
            data[ii, :, :] = torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=1, norm="ortho")
    # data = ifftshift(data, dim=dim)
    return data

def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    if not data.shape[-1] == 2:
        raise ValueError("Tensor does not have separate complex dim.")

    data = ifftshift(data, dim=[-3, -2])
    data = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(data), dim=(-2, -1), norm="ortho"
        )
    )
    data = fftshift(data, dim=[-3, -2])

    return data

def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.
    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.
    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def complex_abs_sq(data):
    """
    Compute the squared absolute value of a complex tensor
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1)


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def root_sum_of_squares_complex(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.
    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform
    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.
    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.
    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def center_crop_to_smallest(x, y):
    """
    Apply a center crop on the larger image to the size of the smaller image.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))
    return x, y


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)
    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero
    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.
        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero
        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)


def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)


def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

class ApplyKSpaceMask(nn.Module):
    def __init__(self, mask):
        super(ApplyKSpaceMask, self).__init__()
        self.mask = mask

    def forward(self, input):
        kspace_data = fft2(ifftshift(input))
        masked_kspace_data = kspace_data * self.mask + 0.0
        visual_data = fftshift(ifft2(masked_kspace_data))
        return visual_data

def gaussian_oned(x):
    return 1.0 / np.sqrt(2.0*np.pi) * np.exp(-1*x**2 / 2.0)

def find_nearest(x, array):
    idx = (np.abs(x - array)).argmin()
    return idx

def exhaustive_sample(center_frac, acceleration, n_cols, seed):
    grid = np.linspace(-3.0,3.0,n_cols)
    sample_grid = np.zeros((n_cols,))
    num_low_freqs = int(round(n_cols * center_frac))
    pad = (n_cols - num_low_freqs + 1) // 2
    sample_grid[pad:pad+num_low_freqs] = [True]*num_low_freqs
    rng = np.random.RandomState(seed=seed)
    while True:
        sample_point = rng.standard_normal()
        if np.abs(sample_point) < 3.0:
            nearest_index = find_nearest(sample_point, grid)
            sample_grid[nearest_index] = True

        ratio_sampled = n_cols / sum(sample_grid)
        if acceleration > ratio_sampled:
            return sample_grid


def create_mask(shape, center_fraction, acceleration, seed=0, flipaxis=False):
    num_cols = shape[-2]

    # Create the mask
    mask = exhaustive_sample(center_fraction, acceleration, num_cols, seed)
    # num_low_freqs = int(round(num_cols * center_fraction))
    # prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    # rng = np.random.RandomState(seed=seed)
    #
    # mask = rng.standard_normal(size=num_cols) < prob
    # pad = (num_cols - num_low_freqs + 1) // 2
    # mask[pad:pad + num_low_freqs] = True

    # Reshape the mask
    mask_shape = [1 for _ in shape]
    if flipaxis:
        mask_shape[0] = num_cols
    else:
        mask_shape[-2] = num_cols
    # mask = mask.astype(np.float32)
    mask = mask.reshape(*mask_shape).astype(np.float32)
    # print(mask.shape)
    # exit()

    mask = torch.tensor(mask, requires_grad=False)
    return mask



class toKspace(nn.Module):
    def __init__(self, mask=None):
        super(toKspace, self).__init__()
        if mask is None:
            self.mask = mask
        else:
            self.register_buffer('mask', tensor=mask)


    def forward(self, input):
        kspace_data = fft2(ifftshift(input.permute((0,2,3,1))))
        if self.mask is not None:
            kspace_data = kspace_data * self.mask + 0.0
        return kspace_data.permute((0,3,1,2))

class toKspaceMulti(nn.Module):
    def __init__(self, masks):
        super(toKspaceMulti, self).__init__()
        self.masks = masks
        self.ii = 0

    def advance_ii(self):
        self.ii = (self.ii + 1) % 3

    def forward(self, input):
        kspace_data = fft2(ifftshift(input.permute((0,2,3,1))))
        mask = self.masks[self.ii]

        kspace_data = kspace_data * mask + 0.0
        return kspace_data.permute((0,3,1,2))


class fromKspace(nn.Module):
    def __init__(self, mask=None):
        super(fromKspace, self).__init__()
        if mask is None:
            self.mask = mask
        else:
            self.register_buffer('mask', tensor=mask)

    def forward(self, input):
        if self.mask is not None:
            input = input.permute((0,2,3,1)) * self.mask + 0.0
        else:
            input = input.permute((0,2,3,1))
        image_data = ifftshift(ifft2(input))
        return image_data.permute((0,3,1,2))

class cartesianSingleCoilMRI(LinearOperator):
    def __init__(self, kspace_mask):
        super(cartesianSingleCoilMRI, self).__init__()
        self.register_buffer('mask', tensor=kspace_mask)

    def forward(self, input):
        input = ifftshift(input.permute((0, 2, 3, 1)))
        complex_input = torch.view_as_complex(input)
        kspace = torch.fft.fftn(complex_input, dim=1, norm="ortho")
        kspace = torch.fft.fftn(kspace, dim=2, norm="ortho")
        kspace = fftshift(kspace)
        if self.mask is not None:
            kspace_data = kspace * self.mask + 0.0
            kspace_data = ifftshift(kspace_data)
        return torch.view_as_real(kspace_data)

    def gramian(self, input):
        input = ifftshift(input.permute((0, 2, 3, 1)))
        complex_input = torch.view_as_complex(input)
        kspace = torch.fft.fftn(complex_input, dim=1, norm="ortho")
        kspace = torch.fft.fftn(kspace, dim=2, norm="ortho")
        kspace = fftshift(kspace)
        if self.mask is not None:
            kspace_data = kspace * self.mask + 0.0
            kspace_data = ifftshift(kspace_data)

        kspace_data = torch.fft.ifftn(kspace_data, dim=1, norm="ortho")
        realspace = torch.fft.ifftn(kspace_data, dim=2, norm="ortho")
        realspace = torch.view_as_real(realspace)

        output = ifftshift(realspace).permute((0,3,1,2))
        return output

    def adjoint(self, input):
        complex_input = torch.view_as_complex(input)
        complex_input = torch.fft.ifftn(complex_input, dim=1, norm="ortho")
        realspace = torch.fft.ifftn(complex_input, dim=2, norm="ortho")

        realspace = torch.view_as_real(realspace)

        output = ifftshift(realspace).permute((0, 3, 1, 2))
        return output