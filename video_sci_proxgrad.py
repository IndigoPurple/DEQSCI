import torch
import os
import random
import sys
import argparse

import torch.nn as nn
import torch.optim as optim

from networks.normalized_equilibrium_u_net_yaping import UnetModel, Unet3D, UnetNorm
from networks.resnet import nblock_resnet as resnet
from solvers.equilibrium_solvers_yaping import EquilibriumProxGradSCI
from training import sci_equilibrium_training
from solvers import new_equilibrium_utils_yaping as eq_utils

# yaping
from networks.ffdnet.models import FFDNet
from utils.sci_dataloader import SCITrainingDatasetSubset, SCITestDataset
from utils.cg_utils import A_torch_, At_torch_
import re
import cv2

parser = argparse.ArgumentParser()
# parser.add_argument('--gpu_ids', nargs='+', default=0, type=int)
parser.add_argument('--gpu_ids', default='0')
parser.add_argument('--n_epochs', default=80)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--and_maxiters', default=100) # default = 100
parser.add_argument('--and_beta', type=float, default=1.0)
parser.add_argument('--and_m', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.0001) # default=0.1, GAPnet = 0.001
parser.add_argument('--etainit', type=float, default=0.9)
parser.add_argument('--lr_gamma', type=float, default=0.9) # defualt=0.1, GAPnet = 0.9
parser.add_argument('--sched_step', type=int, default=10) # defualt=10, GAPnet = 10
parser.add_argument('--savepath',
                    default="/home/yaping/projects/deep_equilibrium_inverse/save/test_ffdnet1/")
parser.add_argument('--trainpath',
                    default="/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/")
parser.add_argument('--testpath',
                    default="/home/yaping/projects/data/test_gray/")
parser.add_argument('--loadpath',
                    default='') #/home/yaping/projects/deep_equilibrium_inverse/save/server/exp2_bsz21/model/epoch_8.ckpt #./save/test_ffdnet/model/best.ckpt
parser.add_argument('--denoiser',
                    default='ffdnet')
parser.add_argument('--inference', default=False, help='turning model to training or testing mode.')
parser.add_argument('--print_every_n_steps', type=int, default=1)
parser.add_argument('--save_every_n_steps', type=int, default=50)
parser.add_argument('--sigma', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
gpu_ids = [int(args.gpu_ids)]

# gpu_ids = args.gpu_ids
inference = args.inference

# Parameters to modify
n_epochs = int(args.n_epochs)
current_epoch = 0
batch_size = int(args.batch_size)
n_channels = 1 # default=3
max_iters = int(args.and_maxiters)
anderson_m = int(args.and_m)
anderson_beta = float(args.and_beta)
denoiser = args.denoiser

learning_rate = float(args.lr)
print_every_n_steps = args.print_every_n_steps
save_every_n_steps = args.save_every_n_steps
initial_eta = 0.2

kernel_size = 5
kernel_sigma = 5.0
noise_sigma = 1e-2

# modify this for your machine
save_location = args.savepath
load_location = args.loadpath
test_location = args.testpath

save_model_path = save_location + 'model/'
save_train_img_path = save_location + 'img/train/'
save_test_img_path = save_location + 'img/test/'
save_best_img_path = save_location + 'img/best/'
# log_location = save_location + 'out_file.out'
tflog_location = save_location

path_list = [save_model_path, save_train_img_path, save_test_img_path]
for path in path_list:
    if not os.path.exists(path):
        os.makedirs(path)

# __console = sys.stdout
# log = open(log_location,'a+')
# sys.stdout = log

# yaping
mask_location = args.trainpath + 'mask.mat'
gt_location = args.trainpath + 'gt/'
meas_location = args.trainpath + 'measurement/'

# for ii in range(1):
#     try:
#         torch.cuda.get_device_properties(ii)
#         print(str(ii), flush=True)
#         if not gpu_ids:
#             gpu_ids = [ii]
#         else:
#             gpu_ids.append(ii)
#     except AssertionError:
#         print('Not ' + str(ii) + "!", flush=True)

# yaping
print('cuda', torch.cuda.is_available())

# print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
# if isinstance(gpu_ids, int):
#     gpu_ids = [gpu_ids]
# else:
#     gpu_ids = [int(x) for x in gpu_ids]
# devices = ''
# for x in gpu_ids:
#     devices = devices+str(x)+','
# devices = devices[:-1]
# os.environ["CUDA_VISIBLE_DEVICES"] = devices
# device management
# device = torch.device('cuda:{0,1,2}' if torch.cuda.is_available() else 'cpu')
use_dataparallel = len(gpu_ids) > 1
print("GPU IDs: " + str([int(x) for x in gpu_ids]), flush=True)

# yaping: dataloader
if not inference:
    dataset = SCITrainingDatasetSubset(gt_location, meas_location, mask_location)
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory=True
    )

test_dataset = SCITestDataset(test_location)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False, drop_last=True,
)

### Set up solver and problem setting
# standard u-net
if denoiser == 'unet':
    learned_component = UnetModel(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                                       drop_prob=0.0, chans=32, tag='conv2d')
elif denoiser == 'unet3d':
    learned_component = Unet3D(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                                       drop_prob=0.0, chans=32, tag='conv3d')
elif denoiser == 'unet_norm':
    learned_component = UnetNorm(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                                       drop_prob=0.0, chans=32, tag='conv2d')
elif denoiser == 'resnet':
    learned_component = resnet(inc=n_channels, onc=n_channels)
# elif denoiser == 'dncnn':
#     from networks.models import DnCNN
#     learned_component = DnCNN(channels=1, num_of_layers=17,tag='denoiser')
elif denoiser == 'dncnn_norm':
    from networks.realSN_models import DnCNN
    learned_component = DnCNN(channels=1, num_of_layers=17,tag='denoiser')
elif denoiser == 'ffdnet':
    learned_component = FFDNet(num_input_channels=n_channels, tag='ffdnet')
elif denoiser == 'unet3d_noise':
    learned_component = Unet3D(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                               drop_prob=0.0, chans=32, tag='3d_denoiser')
elif denoiser == 'unetnorm_noise':
    learned_component = UnetNorm(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                                       drop_prob=0.0, chans=32, tag='denoiser')
elif denoiser == 'resnet_noise':
    learned_component = resnet(inc=n_channels, onc=n_channels, tag='denoiser')
elif denoiser == "DnCNN":
    from networks.provable.model.models import DnCNN
    learned_component = DnCNN(channels=1, num_of_layers=17, tag='denoiser')
elif denoiser == "SimpleCNN":
    from networks.provable.model.SimpleCNN_models import DnCNN
    learned_component = DnCNN(1, num_of_layers = 4, lip = 0.0, no_bn = True, tag='denoiser')
elif denoiser == "RealSN_DnCNN":
    from networks.provable.model.realSN_models import DnCNN
    learned_component = DnCNN(channels=1, num_of_layers=17, tag='denoiser')
elif denoiser == "RealSN_SimpleCNN":
    from networks.provable.model.SimpleCNN_models import DnCNN
    learned_component = DnCNN(1, num_of_layers = 4, lip = 1.0, no_bn = True, tag='denoiser')
else:
    raise NotImplementedError('unknown denoiser!')



# inference = True
if inference:
    learned_component.eval()
# yaping
solver = EquilibriumProxGradSCI( A=A_torch_, At=At_torch_, nonlinear_operator=learned_component,
                        eta=initial_eta, minval=-1, maxval = 1)

if use_dataparallel:
    solver = nn.DataParallel(solver)
solver = solver.cuda()

start_epoch = 0
optimizer = optim.Adam(params=solver.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(args.sched_step), gamma=float(args.lr_gamma))
cpu_only = not torch.cuda.is_available()

sigma = args.sigma
if sigma:
    load_location = "./networks/provable/Pretrained_models/" + denoiser + "_noise" + str(sigma) + ".pth"
# learned_component.load_state_dict(torch.load(path))
pretrained_denoiser = False
if not pretrained_denoiser:
    if os.path.exists(load_location):
        if not cpu_only:
            saved_dict = torch.load(load_location)
        else:
            saved_dict = torch.load(load_location, map_location='cpu')
        start_epoch = saved_dict['epoch'] + 1
        if use_dataparallel:
            saved_dict['solver_state_dict'] = {(('module.' + k) if not k.startswith('module.') else k): v for (k, v) in
                                               saved_dict['solver_state_dict'].items()}
        else:
            saved_dict['solver_state_dict'] = {(k[7:] if k.startswith('module.') else k): v for (k, v) in
                                               saved_dict['solver_state_dict'].items()}
        solver.load_state_dict(saved_dict['solver_state_dict'])
        # optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
        # scheduler.load_state_dict(saved_dict['scheduler_state_dict'])
        print('loaded dict!')
        # learned_component.load_state_dict(saved_dict['solver_state_dict'])
else:
    state_dict = torch.load(load_location)
    if use_dataparallel:
        state_dict = {(('module.' + k) if not k.startswith('module.') else k): v for (k, v) in
                                           state_dict.items()}
    else:
        state_dict = {(k[7:] if k.startswith('module.') else k): v for (k, v) in
                                           state_dict.items()}
    learned_component.load_state_dict(state_dict)
    print('loaded dict!')

# set up loss and train
# lossfunction = torch.nn.MSELoss(reduction='sum')
lossfunction = torch.nn.MSELoss(reduction='mean')

forward_iterator = eq_utils.andersonexp
deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, m=anderson_m, beta=anderson_beta, lam=1e-2,
                                        max_iter=max_iters, tol=1e-5) # debug: reduce number of iterations for solver
                                        # max_iter=max_iters, tol=1e-5)

'''torch.Size([16, 8, 256, 256])
torch.Size([16, 256, 256])
torch.Size([16, 256, 256, 8])'''

# Do test
# test_img_path = save_img_path + 'test/'
# sci_equilibrium_training.test_solver_sci(test_dataloader=test_dataloader,
#             deep_eq_module=deep_eq_module, device=device, save_img_path=test_img_path)

# training
if not inference:
    sci_equilibrium_training.train_solver_sci(
                               single_iterate_solver=solver, train_dataloader=dataloader, test_dataloader=test_dataloader,
                               optimizer=optimizer, save_model_path=save_model_path,
                               deep_eq_module=deep_eq_module, loss_function=lossfunction, n_epochs=n_epochs,
                               use_dataparallel=use_dataparallel, scheduler=scheduler,
                               print_every_n_steps=print_every_n_steps, save_every_n_steps=save_every_n_steps,
                               start_epoch=start_epoch, train_img_path=save_train_img_path, test_img_path=save_test_img_path, 
                               best_img_path=save_best_img_path, tflog_path=tflog_location
    )
else:
    cur_psnr, all_images = sci_equilibrium_training.test_solver_sci(test_dataloader=test_dataloader,
                deep_eq_module=deep_eq_module, save_img_path=save_test_img_path)
    for k in all_images:
        cv2.imwrite(k, all_images[k])

# sys.stdout = __console
