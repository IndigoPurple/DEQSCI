import torch
import os
import random
import sys
import argparse

import torch.nn as nn
import torch.optim as optim

from networks.normalized_equilibrium_u_net_yaping import UnetModel, DnCNN
from solvers.equilibrium_solvers_yaping import EquilibriumProxGradSCI
from training import sci_equilibrium_training
from solvers import new_equilibrium_utils_yaping as eq_utils

# yaping
from networks.ffdnet.models import FFDNet
from utils.sci_dataloader import SCITrainingDatasetSubset, SCITestDataset
from utils.cg_utils import A_torch_, At_torch_
import re

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=80)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--and_maxiters', default=100) # default = 100
parser.add_argument('--and_beta', type=float, default=1.0)
parser.add_argument('--and_m', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001) # default=0.1, GAPnet = 0.001
parser.add_argument('--etainit', type=float, default=0.9)
parser.add_argument('--lr_gamma', type=float, default=0.9) # defualt=0.1, GAPnet = 0.9
parser.add_argument('--sched_step', type=int, default=10) # defualt=10, GAPnet = 10
parser.add_argument('--savepath',
                    default="/home/yaping/projects/deep_equilibrium_inverse/save/video_sci_unet_test/")
parser.add_argument('--trainpath',
                    default="/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/")
parser.add_argument('--testpath',
                    default="/home/yaping/projects/data/test_gray/")
parser.add_argument('--loadpath',
                    default='None')
parser.add_argument('--save_img_path',
                    default='/home/yaping/projects/deep_equilibrium_inverse/save/deq_test/test_classic/')
parser.add_argument('--logpath',
                    default='/home/yaping/projects/deep_equilibrium_inverse/save/log/outfile.log')
args = parser.parse_args()

log_location = args.logpath
__console = sys.stdout
log = open(log_location,'a+')
sys.stdout = log

# Parameters to modify
n_epochs = int(args.n_epochs)
current_epoch = 0
batch_size = int(args.batch_size)
n_channels = 1 # default=3
max_iters = int(args.and_maxiters)
anderson_m = int(args.and_m)
anderson_beta = float(args.and_beta)

learning_rate = float(args.lr)
print_every_n_steps = 100
save_every_n_epochs = 1
initial_eta = 0.2

kernel_size = 5
kernel_sigma = 5.0
noise_sigma = 1e-2

# modify this for your machine
save_location = args.savepath
load_location = args.loadpath
test_location = args.testpath
save_img_path = args.save_img_path
# yaping
mask_location = args.trainpath + 'mask.mat'
gt_location = args.trainpath + 'gt/'
meas_location = args.trainpath + 'measurement/'

gpu_ids = []
for ii in range(1):
    try:
        torch.cuda.get_device_properties(ii)
        print(str(ii), flush=True)
        if not gpu_ids:
            gpu_ids = [ii]
        else:
            gpu_ids.append(ii)
    except AssertionError:
        print('Not ' + str(ii) + "!", flush=True)

# yaping
print('cuda', torch.cuda.is_available())

print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
gpu_ids = [int(x) for x in gpu_ids]
# device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_dataparallel = len(gpu_ids) > 1
print("GPU IDs: " + str([int(x) for x in gpu_ids]), flush=True)

# yaping: dataloader
dataset = SCITrainingDatasetSubset(gt_location, meas_location, mask_location)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True,
)

test_dataset = SCITestDataset(test_location)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
)

### Set up solver and problem setting

# standard u-net
learned_component = UnetModel(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                                       drop_prob=0.0, chans=32)
# learned_component = DnCNN(channels=n_channels)


# yaping
solver = EquilibriumProxGradSCI( A=A_torch_, At=At_torch_, nonlinear_operator=learned_component,
                    eta=initial_eta, minval=-1, maxval = 1)

if use_dataparallel:
    solver = nn.DataParallel(solver, device_ids=gpu_ids)
solver = solver.to(device=device)

start_epoch = 0
optimizer = optim.Adam(params=solver.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(args.sched_step), gamma=float(args.lr_gamma))
cpu_only = not torch.cuda.is_available()

if os.path.exists(load_location):
    if not cpu_only:
        saved_dict = torch.load(load_location)
    else:
        saved_dict = torch.load(load_location, map_location='cpu')

    start_epoch = saved_dict['epoch']
    solver.load_state_dict(saved_dict['solver_state_dict'])
    optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
    scheduler.load_state_dict(saved_dict['scheduler_state_dict'])
    # learned_component.load_state_dict(saved_dict['solver_state_dict'])

# set up loss and train
# lossfunction = torch.nn.MSELoss(reduction='sum')
lossfunction = torch.nn.MSELoss(reduction='mean')

forward_iterator = eq_utils.andersonexp
deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, m=anderson_m, beta=anderson_beta, lam=1e-2,
                                        max_iter=max_iters, tol=1e-5)

'''torch.Size([16, 8, 256, 256])
torch.Size([16, 256, 256])
torch.Size([16, 256, 256, 8])'''

# Do test
# test_img_path = save_img_path + 'test/'
# sci_equilibrium_training.test_solver_sci(test_dataloader=test_dataloader,
#             deep_eq_module=deep_eq_module, device=device, save_img_path=test_img_path)

# yaping
sci_equilibrium_training.train_solver_sci(
                               single_iterate_solver=solver, train_dataloader=dataloader, test_dataloader=test_dataloader,
                               optimizer=optimizer, save_location=save_location,
                               deep_eq_module=deep_eq_module, loss_function=lossfunction, n_epochs=n_epochs,
                               use_dataparallel=use_dataparallel, device=device, scheduler=scheduler,
                               print_every_n_steps=print_every_n_steps, save_every_n_epochs=save_every_n_epochs,
                               start_epoch=start_epoch, save_img_path = save_img_path)

sys.stdout = __console