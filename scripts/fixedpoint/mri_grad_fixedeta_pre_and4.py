import torch
import os
import random
import sys
import argparse
sys.path.append('/home-nfs/gilton/learned_iterative_solvers')
# sys.path.append('/Users/dgilton/PycharmProjects/learned_iterative_solvers')

import torch.nn as nn
import torch.optim as optim

import operators.singlecoil_mri as mrimodel
from operators.operator import OperatorPlusNoise
from utils.fastmri_dataloader import singleCoilFastMRIDataloader
from networks.normalized_equilibrium_u_net import UnetModel, DnCNN
from solvers.equilibrium_solvers import EquilibriumProxGradMRI, EquilibriumGrad
from training import refactor_equilibrium_training
from solvers import new_equilibrium_utils as eq_utils

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=80)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--and_maxiters', default=100)
parser.add_argument('--and_beta', type=float, default=1.0)
parser.add_argument('--and_m', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--etainit', type=float, default=0.4)
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--sched_step', type=int, default=10)
parser.add_argument('--acceleration', type=float, default=4.0)
parser.add_argument('--savepath',
                    default="/share/data/vision-greg2/users/gilton/celeba_equilibriumgrad_mri_save_inf.ckpt")
parser.add_argument('--loadpath',
                    default="/share/data/vision-greg2/users/gilton/celeba_equilibriumgrad_mri_save_inf.ckpt")
args = parser.parse_args()


# Parameters to modify
n_epochs = int(args.n_epochs)
current_epoch = 0
batch_size = int(args.batch_size)
n_channels = 2
max_iters = int(args.and_maxiters)
anderson_m = int(args.and_m)
anderson_beta = float(args.and_beta)

learning_rate = float(args.lr)
print_every_n_steps = 2
save_every_n_epochs = 1
initial_eta = float(args.etainit)

dataheight = 320
datawidth = 320
mri_center_fraction = 0.04
mri_acceleration = float(args.acceleration)

mask = mrimodel.create_mask(shape=[dataheight, datawidth, 2], acceleration=mri_acceleration,
                              center_fraction=mri_center_fraction, seed=10)

noise_sigma = 1e-2

# modify this for your machine
# save_location = "/share/data/vision-greg2/users/gilton/mnist_equilibriumgrad_blur.ckpt"
save_location = args.savepath
load_location = "/share/data/willett-group/users/gilton/denoisers/mri_denoiser_unetnorm_4.ckpt"

gpu_ids = []
for ii in range(6):
    try:
        torch.cuda.get_device_properties(ii)
        print(str(ii), flush=True)
        if not gpu_ids:
            gpu_ids = [ii]
        else:
            gpu_ids.append(ii)
    except AssertionError:
        print('Not ' + str(ii) + "!", flush=True)

print(os.getenv('CUDA_VISIBLE_DEVICES'), flush=True)
gpu_ids = [int(x) for x in gpu_ids]
# device management
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_dataparallel = len(gpu_ids) > 1
print("GPU IDs: " + str([int(x) for x in gpu_ids]), flush=True)

# Set up data and dataloaders
data_location = "/share/data/vision-greg2/users/gilton/singlecoil_curated_clean/"
trainset_size = 2000
total_data = 2194
random.seed(10)
all_indices = list(range(trainset_size))
train_indices = random.sample(range(total_data), k=trainset_size)
dataset = singleCoilFastMRIDataloader(data_location, data_indices=train_indices)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
)

### Set up solver and problem setting

forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)
measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)

internal_forward_operator = mrimodel.cartesianSingleCoilMRI(kspace_mask=mask).to(device=device)

# standard u-net
# learned_component = UnetModel(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
#                                        drop_prob=0.0, chans=32)
learned_component = DnCNN(channels=n_channels)

cpu_only = not torch.cuda.is_available()
if os.path.exists(load_location):
    if not cpu_only:
        saved_dict = torch.load(load_location)
    else:
        saved_dict = torch.load(load_location, map_location='cpu')
    learned_component.load_state_dict(saved_dict['solver_state_dict'])

# learned_component = Autoencoder()
solver = EquilibriumGrad(linear_operator=internal_forward_operator, nonlinear_operator=learned_component,
                    eta=initial_eta, minval=-1, maxval = 1)

if use_dataparallel:
    solver = nn.DataParallel(solver, device_ids=gpu_ids)
solver = solver.to(device=device)

start_epoch = 0
optimizer = optim.Adam(params=solver.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(args.sched_step), gamma=float(args.lr_gamma))
cpu_only = not torch.cuda.is_available()


if os.path.exists(save_location):
    if not cpu_only:
        saved_dict = torch.load(save_location)
    else:
        saved_dict = torch.load(save_location, map_location='cpu')

    start_epoch = saved_dict['epoch']
    solver.load_state_dict(saved_dict['solver_state_dict'])
    # optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
    scheduler.load_state_dict(saved_dict['scheduler_state_dict'])


# set up loss and train
lossfunction = torch.nn.MSELoss(reduction='sum')

forward_iterator = eq_utils.andersonexp
deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, m=anderson_m, beta=anderson_beta, lam=1e-2,
                                        max_iter=max_iters, tol=1e-5)
# forward_iterator = eq_utils.forward_iteration
# deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, max_iter=max_iters, tol=1e-8)

# Do train
refactor_equilibrium_training.train_solver_precond(
                               single_iterate_solver=solver, train_dataloader=dataloader,
                               measurement_process=measurement_process, optimizer=optimizer, save_location=save_location,
                               deep_eq_module=deep_eq_module, loss_function=lossfunction, n_epochs=n_epochs,
                               use_dataparallel=use_dataparallel, device=device, scheduler=scheduler,
                               print_every_n_steps=print_every_n_steps, save_every_n_epochs=save_every_n_epochs,
                               start_epoch=start_epoch, forward_operator = forward_operator, noise_sigma=0.3,
                               precond_iterates=50)
