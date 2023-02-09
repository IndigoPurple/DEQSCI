import torch
import os
import random
import sys
import argparse
# sys.path.append('/home-nfs/gilton/learned_iterative_solvers')
# sys.path.append('/Users/dgilton/PycharmProjects/learned_iterative_solvers')

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import operators.blurs as blurs
from operators.operator import OperatorPlusNoise
from utils.celeba_dataloader import CelebaTrainingDatasetSubset, CelebaTestDataset
from networks.normalized_equilibrium_u_net import UnetModel, DnCNN
from solvers.equilibrium_solvers import EquilibriumProxGrad
from training import refactor_equilibrium_training
from solvers import new_equilibrium_utils as eq_utils
from networks.ffdnet.models import FFDNet

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=80)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--and_maxiters', default=100)
parser.add_argument('--and_beta', type=float, default=1.0)
parser.add_argument('--and_m', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--etainit', type=float, default=0.9)
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--sched_step', type=int, default=10)
parser.add_argument('--savepath',
                    default="/home/yaping/projects/deep_equilibrium_inverse/save/test_ep25_dncnn.ckpt")
args = parser.parse_args()


# Parameters to modify
n_epochs = int(args.n_epochs)
current_epoch = 0
batch_size = int(args.batch_size)
n_channels = 3
max_iters = int(args.and_maxiters)
anderson_m = int(args.and_m)
anderson_beta = float(args.and_beta)

learning_rate = float(args.lr)
print_every_n_steps = 1
save_every_n_epochs = 1
initial_eta = 0.2

initial_data_points = 10000
# point this towards your celeba files
data_location = "/home/yaping/projects/data/CelebA-20210820T123614Z-016/CelebA/Img/img_align_celeba/"

kernel_size = 5
kernel_sigma = 5.0
noise_sigma = 1e-2

# modify this for your machine
# save_location = "/share/data/vision-greg2/users/gilton/mnist_equilibriumgrad_blur.ckpt"
save_location = args.savepath
load_location = "/home/yaping/projects/deep_equilibrium_inverse/save/dncnn_train.ckpt"

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

# Set up data and dataloaders
transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)
celeba_train_size = 162770
total_data = initial_data_points
total_indices = random.sample(range(celeba_train_size), k=total_data)
initial_indices = total_indices

dataset = CelebaTrainingDatasetSubset(data_location, subset_indices=initial_indices, transform=transform)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, drop_last=True,
)

test_dataset = CelebaTestDataset(data_location, transform=transform)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True,
)

### Set up solver and problem setting

forward_operator = blurs.GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size,
                                      n_channels=3, n_spatial_dimensions=2).to(device=device)
measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)

internal_forward_operator = blurs.GaussianBlur(sigma=kernel_sigma, kernel_size=kernel_size,
                                      n_channels=3, n_spatial_dimensions=2).to(device=device)

# standard u-net
# learned_component = UnetModel(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
#                                        drop_prob=0.0, chans=32)
learned_component = DnCNN(channels=n_channels)

#yaping
# learned_component = FFDNet(num_input_channels=n_channels)

# if os.path.exists(load_location):
#     if torch.cuda.is_available():
#         saved_dict = torch.load(load_location)
#     else:
#         saved_dict = torch.load(load_location, map_location='cpu')
#
#     start_epoch = saved_dict['epoch']
#     learned_component.load_state_dict(saved_dict['solver_state_dict'])

# learned_component = Autoencoder()
solver = EquilibriumProxGrad(linear_operator=internal_forward_operator, nonlinear_operator=learned_component,
                    eta=initial_eta, minval=-1, maxval = 1)

if use_dataparallel:
    solver = nn.DataParallel(solver, device_ids=gpu_ids)
solver = solver.to(device=device)

start_epoch = 0
optimizer = optim.Adam(params=solver.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(args.sched_step), gamma=float(args.lr_gamma))
cpu_only = not torch.cuda.is_available()


# if os.path.exists(save_location):
#     if not cpu_only:
#         saved_dict = torch.load(save_location)
#     else:
#         saved_dict = torch.load(save_location, map_location='cpu')
#
#     start_epoch = saved_dict['epoch']
#     solver.load_state_dict(saved_dict['solver_state_dict'])
#     # optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
#     scheduler.load_state_dict(saved_dict['scheduler_state_dict'])

if os.path.exists(load_location):
    if not cpu_only:
        saved_dict = torch.load(load_location)
    else:
        saved_dict = torch.load(load_location, map_location='cpu')

    # start_epoch = saved_dict['epoch']
    # solver.load_state_dict(saved_dict['solver_state_dict'])
    # optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
    # scheduler.load_state_dict(saved_dict['scheduler_state_dict'])
    learned_component.load_state_dict(saved_dict['solver_state_dict'])

# set up loss and train
lossfunction = torch.nn.MSELoss(reduction='sum')

forward_iterator = eq_utils.andersonexp
deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, m=anderson_m, beta=anderson_beta, lam=1e-2,
                                        max_iter=max_iters, tol=1e-5)
# forward_iterator = eq_utils.forward_iteration
# deep_eq_module = eq_utils.DEQFixedPoint(solver, forward_iterator, max_iter=100, tol=1e-8)

# #yaping test
# for ii, sample_batch in enumerate(dataloader):
#     print(sample_batch.shape)
#     exit()

# Do train
refactor_equilibrium_training.train_solver_precond1(
                               single_iterate_solver=solver, train_dataloader=dataloader, test_dataloader=test_dataloader,
                               measurement_process=measurement_process, optimizer=optimizer, save_location=save_location,
                               deep_eq_module=deep_eq_module, loss_function=lossfunction, n_epochs=n_epochs,
                               use_dataparallel=use_dataparallel, device=device, scheduler=scheduler,
                               print_every_n_steps=print_every_n_steps, save_every_n_epochs=save_every_n_epochs,
                               start_epoch=start_epoch, forward_operator = forward_operator, noise_sigma=noise_sigma,
                               precond_iterates=60)
