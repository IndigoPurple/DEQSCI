import torch
import os
import random
import sys
import argparse
sys.path.append('/home-nfs/gilton/learned_iterative_solvers')
# sys.path.append('/Users/dgilton/PycharmProjects/learned_iterative_solvers')

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import operators.operator as lin_operator
from operators.operator import OperatorPlusNoise
from utils.celeba_dataloader import CelebaTrainingDatasetSubset, CelebaTestDataset
from networks.equilibrium_u_net import UnetModel
from training import denoiser_training
from solvers import new_equilibrium_utils as eq_utils

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', default=80)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--sched_step', type=int, default=10)
parser.add_argument('--noise_sigma', type=float, default=0.25) # default = 0.01
parser.add_argument('--savepath',
                    # default="/share/data/vision-greg2/users/gilton/celeba_equilibriumgrad_blur_save_inf.ckpt")
                    #yaping
                    default="/home/yaping/projects/deep_equilibrium_inverse/save/unet_train.ckpt")

args = parser.parse_args()


# Parameters to modify
n_epochs = int(args.n_epochs)
current_epoch = 0
batch_size = int(args.batch_size)
n_channels = 3

learning_rate = float(args.lr)
print_every_n_steps = 10
save_every_n_epochs = 5

initial_data_points = 10000
# point this towards your celeba files
data_location = "/home/yaping/projects/data/CelebA-20210820T123614Z-016/CelebA/Img/img_align_celeba/"

noise_sigma = float(args.noise_sigma)

# modify this for your machine
# save_location = "/share/data/vision-greg2/users/gilton/mnist_equilibriumgrad_blur.ckpt"
save_location = args.savepath
load_location = args.savepath

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
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
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

forward_operator = lin_operator.Identity().to(device=device)
measurement_process = OperatorPlusNoise(forward_operator, noise_sigma=noise_sigma).to(device=device)

# learned_component = Autoencoder()
solver = UnetModel(in_chans=n_channels, out_chans=n_channels, num_pool_layers=4,
                                       drop_prob=0.0, chans=32)

if use_dataparallel:
    solver = nn.DataParallel(solver, device_ids=gpu_ids)
solver = solver.to(device=device)

start_epoch = 0
optimizer = optim.Adam(params=solver.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=int(args.sched_step), gamma=float(args.lr_gamma))
cpu_only = not torch.cuda.is_available()


# if os.path.exists(load_location):
#     if not cpu_only:
#         saved_dict = torch.load(load_location)
#     else:
#         saved_dict = torch.load(load_location, map_location='cpu')
#
#     start_epoch = saved_dict['epoch']
#     solver.load_state_dict(saved_dict['solver_state_dict'])
#     # optimizer.load_state_dict(saved_dict['optimizer_state_dict'])
#     scheduler.load_state_dict(saved_dict['scheduler_state_dict'])


# set up loss and train
lossfunction = torch.nn.MSELoss()

# Do train
denoiser_training.train_denoiser(denoising_net=solver, train_dataloader=dataloader, test_dataloader=test_dataloader,
                               measurement_process=measurement_process, optimizer=optimizer, save_location=save_location,
                               loss_function=lossfunction, n_epochs=n_epochs,
                               use_dataparallel=use_dataparallel, device=device, scheduler=scheduler,
                               print_every_n_steps=print_every_n_steps, save_every_n_epochs=save_every_n_epochs,
                               start_epoch=start_epoch)
