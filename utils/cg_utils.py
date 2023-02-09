import torch.nn as nn
import torch
#yaping
import math
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from typing import Any, List, Tuple, Union

def complex_conj(x):
    assert x.shape[1] == 2
    return torch.stack((x[:,0, ...], -x[:,1,...]), dim=1)

def torchdotproduct(x,y):
    # if complexdata:
    # y = complex_conj(y)
    return torch.sum(x*y,dim=[1,2,3])

def single_cg_iteration(x, d, g, b, ATA, regularization_lambda):

    def regATA(input, ATA):
        return ATA(input) + regularization_lambda*input

    Qd = regATA(d, ATA)
    dQd = torchdotproduct(d, Qd)
    alpha = -torchdotproduct(g,d) / dQd
    alpha = alpha.view((-1,1,1,1))
    x = x + alpha * d
    g = regATA(x, ATA) - b
    gQd = torchdotproduct(g, Qd)
    beta = gQd / dQd
    beta = beta.view((-1,1,1,1))
    d = -g + beta*d
    return x, d, g

# This function solves the system ATA x = ATy, where initial_point is supposed
# to be ATy. This can be backpropagated through.
def conjugate_gradient(initial_point, ATA, regularization_lambda, n_iterations=10):
    x = torch.zeros_like(initial_point)
    d = initial_point
    g = -d
    for ii in range(n_iterations):
        x, d, g = single_cg_iteration(x, d, g, initial_point, ATA, regularization_lambda)
    return x

def complex_dotproduct(x, y):
    return torchdotproduct(complex_conj(x), y)

def single_cg_iteration_MRI(rTr, x, r, p, ATA, regularization_lambda):

    batch_size = x.shape[0]
    def regATA(input):
        return ATA(input) + regularization_lambda*input

    Ap = regATA(p)

    rTr = rTr.view(batch_size, 1, 1, 1)
    alpha = rTr / complex_dotproduct(p, Ap).view(batch_size, 1, 1, 1)

    x_new = x + alpha * p
    r_new = r - alpha * Ap
    rTr_new = complex_dotproduct(r_new, r_new)
    rTr_new = rTr_new.view(batch_size, 1, 1, 1)

    beta = rTr_new / rTr
    p_new = r + beta * p
    return rTr_new, x_new, r_new, p_new

def conjugate_gradient_MRI(initial_point, ATA, regularization_lambda, n_iterations=10):
    '''Strightforward implementation of MoDLs code'''
    x = torch.zeros_like(initial_point)
    r = initial_point
    p = initial_point
    rTr = complex_dotproduct(r, r)
    for ii in range(n_iterations):
        rTr, x, r, p = single_cg_iteration_MRI(rTr, x, r, p, ATA, regularization_lambda)
    return x

def A_(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return np.sum(x*Phi, axis=3)  # element-wise product, default axis = 1

def A_torch_(x, Phi):
    '''
    Forward model of snapshot compressive imaging (SCI), where multiple coded
    frames are collapsed into a snapshot measurement.
    '''
    return torch.sum(x*Phi, axis=3)  # element-wise product, default axis = 1


def At_(y, Phi):
    '''
    Tanspose of the forward model.
    '''
    # (nrow, ncol, nmask) = Phi.shape
    # x = np.zeros((nrow, ncol, nmask))
    # (bsz, nmask, nrow, ncol) = Phi.shape
    # x = np.zeros((bsz, nmask, nrow, ncol))
    (bsz, nrow, ncol, nmask) = Phi.shape
    x = np.zeros((bsz, nrow, ncol, nmask))
    for nt in range(nmask):
         # x[:,:,nt] = np.multiply(y, Phi[:,:,nt])
         # x[:, nt, :, :] = np.multiply(y, Phi[:, nt, :, :])
         x[:, :, :, nt] = np.multiply(y, Phi[:, :, :, nt])
    return x
    #return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)


# def At_torch_(y, Phi):
#     '''
#     Tanspose of the forward model.
#     '''
#     (bsz, nrow, ncol, nmask) = Phi.shape
#     x = torch.zeros_like(Phi).cuda()
#     for nt in range(nmask):
#          # x[:,:,nt] = y * Phi[:,:,nt] # yaping mute
#          # x[:, nt, :, :] = y * Phi[:, nt, :, :]
#          x[:, :, :, nt] = y * Phi[:, :, :, nt]
#     return x
#     #return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)

def At_torch_(y, Phi):
    # debug
    '''
        fast At_torch_
    '''
    return y[:,:,:,None] * Phi


def ADMM_TV_rec(y, Phi, A, At, maxiter, step_size, tv_weight, eta):
    # y1 = np.zeros((row,col))
    y = y.cpu().numpy()
    Phi = Phi.cpu().numpy()
    theta = At(y, Phi)
    v = theta
    b = np.zeros_like(Phi)
    Phi_sum = np.sum(Phi, axis=1)
    Phi_sum[Phi_sum == 0] = 1
    # print('begin computing TV initial value')
    for ni in range(maxiter):
        # print(ni)
        yb = A(theta + b, Phi)
        # y1 = y1+ (y-fb)
        v = (theta + b) + np.multiply(step_size, At(np.divide(y - yb, Phi_sum + eta), Phi))
        # vmb = v-b
        theta = denoise_tv_chambolle(v - b, tv_weight, n_iter_max=30, multichannel=True)

        b = b - (v - theta)
        tv_weight = 0.999 * tv_weight
        eta = 0.998 * eta
    # print('get TV initial value')
    return torch.from_numpy(v).type(torch.FloatTensor).cuda()

def psnr(ref, img):
    '''
    Peak signal-to-noise ratio (PSNR).
    '''
    if type(ref) is torch.Tensor:
        ref = ref.detach().cpu().numpy()
    if type(img) is torch.Tensor:
        img = img.detach().cpu().numpy()

    mse = np.mean( (ref - img) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def GAP_TV_rec_test(y, Phi, Phi_sum, gt, A, At, maxiter, step_size, tv_weight):
    y_np = y.cpu().numpy()
    Phi_np = Phi.cpu().numpy()
    Phi_sum_np = Phi_sum.cpu().numpy()

    y = y.cuda()
    Phi = Phi.cuda()
    Phi_sum = Phi_sum.cuda()

    y1 = torch.zeros_like(y)

    f = At_torch_(y, Phi)
    y1_np = np.zeros_like(y_np)
    f_np = At(y_np, Phi_np)
    diff1 = f.cpu() -f_np
    print('1111111:%d,%d,%d' % diff1.mean())
    # Phi_sum = np.sum(Phi, axis=3)
    # Phi_sum[Phi_sum == 0] = 1
    for ni in range(maxiter):
        fb = A_torch_(f, Phi)
        fb_np = A(f_np, Phi_np)
        diff2 = fb.cpu() - fb_np
        print('222222222:%d,%d,%d' % diff2.mean())
        y1 = y1 + (y-fb)
        f = f + torch.multiply(torch.tensor(1), At_torch_((y1 - fb)/Phi_sum, Phi))
        y1_np = y1_np + (y_np - fb_np)
        f_np = f_np + np.multiply(step_size, At(np.divide(y1_np - fb_np, Phi_sum_np), Phi_np))
        diff3 = f.cpu() - f_np
        print('333333333:%d,%d,%d' % diff3.mean())
        exit()
        f = denoise_tv_chambolle(f, tv_weight, n_iter_max=30, multichannel=True)
        # if (ni+1) % 5 == 0:
        #     print("GAP-TV: Iteration %3d, PSNR = %2.2f dB" % ((ni+1), psnr(f, gt)) )
    print("GAP-TV: PSNR = %2.2f dB" % (psnr(f, gt)))
    return torch.from_numpy(f).type(torch.FloatTensor).cuda()

def GAP_TV_rec(y, Phi, Phi_sum, gt, A, At, maxiter, step_size, tv_weight):
    y = y.cpu().numpy()
    Phi = Phi.cpu().numpy()
    Phi_sum = Phi_sum.cpu().numpy()

    y1 = np.zeros_like(y)
    f = At(y, Phi)
    # Phi_sum = np.sum(Phi, axis=3)
    # Phi_sum[Phi_sum == 0] = 1
    for ni in range(maxiter):
        fb = A(f, Phi)
        y1 = y1 + (y - fb)
        f = f + np.multiply(step_size, At(np.divide(y1 - fb, Phi_sum), Phi))
        f = denoise_tv_chambolle(f, tv_weight, n_iter_max=30, multichannel=True)
        # if (ni+1) % 5 == 0:
        #     print("GAP-TV: Iteration %3d, PSNR = %2.2f dB" % ((ni+1), psnr(f, gt)) )
    print("GAP-TV: PSNR = %2.2f dB" % (psnr(f, gt)))
    return torch.from_numpy(f).type(torch.FloatTensor).cuda()

'''This function compute the initial point for SCI,
    corresponding to f = At(y,Phi) in GAP_FFDNet_rec'''
def initial_point(y,Phi, Phi_sum, gt):
    return At_torch_(y, Phi)
    # return ADMM_TV_rec(y, Phi, A_, At_, maxiter=40, step_size=1, tv_weight=0.3,eta=1e-8)

    # Phi = Phi.permute(0, 3, 1, 2)
    # gt = gt.permute(0, 3, 1, 2)
    # init = GAP_TV_rec(y, Phi, gt, A_, At_, maxiter=40, step_size=1, tv_weight=0.3)
    # return GAP_TV_rec(y, Phi, Phi_sum, gt, A_, At_, maxiter=40, step_size=1, tv_weight=0.3)
    # return GAP_TV_rec_test(y, Phi, Phi_sum, gt, A_, At_, maxiter=40, step_size=1, tv_weight=0.3)

def initial_point_admm(y,Phi, Phi_sum, gt):
    return [At_torch_(y, Phi), torch.zeros_like(Phi)]

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]



#############################################################################
##########################################################################
# gt_location = '/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/gt/'
# meas_location = '/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/measurement/'
# mask_location = "/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/mask.mat"
# def transform(input, mean=0.5, std=0.5):
#     return (input-mean)/std
#
#
#
# batch_size = 2
# from utils.sci_dataloader import SCITrainingDatasetSubset
# dataset = SCITrainingDatasetSubset(gt_location, meas_location, mask_location, transform=transform)
# # dataset = CelebaTrainingDatasetSubset(data_location, subset_indices=initial_indices, transform=transform)
# dataloader = torch.utils.data.DataLoader(
#     dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
# )
#
# for ii, sample_batch in enumerate(dataloader):
#     gt_batch = sample_batch['gt'].to(device=device)
#     # y = measurement_process(sample_batch)
#     y = sample_batch['meas'].to(device=device)
#     Phi = sample_batch['mask'].to(device=device)
#     print(y.shape, Phi.shape, gt_batch.shape)

