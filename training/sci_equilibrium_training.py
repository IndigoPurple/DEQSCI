from collections import OrderedDict
import torch
import numpy as np
from solvers import new_equilibrium_utils as eq_utils
from torch import autograd
from utils import cg_utils
# yaping
import cv2
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio
from utils.cg_utils import EasyDict
from tqdm import tqdm
import time

import torch.utils.tensorboard as tensorboard

# yaping
def tensor_to_np(tensor):
    img = tensor.clip(0, 1).cpu().detach().unsqueeze(2).numpy() * 255.
    return img

def tensor_to_PIL(tensor):
    nu = tensor.cpu().numpy()
    image = Image.fromarray(nu)
    return  image

def train_solver_sci(single_iterate_solver, train_dataloader, optimizer,
                 save_model_path, loss_function, n_epochs, deep_eq_module,
                 use_dataparallel=False, scheduler=None,
                 print_every_n_steps=100, save_every_n_steps=1000, start_epoch=0,
                         test_dataloader = None, train_img_path=None, test_img_path=None, best_img_path=None,
                         tflog_path=None):
    start_time = time.time()
    cur_nimg = 0
    stats_tfevents = tensorboard.SummaryWriter(tflog_path)

    previous_loss = 10.0
    reset_flag = False

    best_psnr = 0

    for epoch in range(start_epoch, n_epochs):

        #yaping
        if reset_flag:
            save_state_dict = torch.load(save_model_path)
            single_iterate_solver.load_state_dict(save_state_dict['solver_state_dict'])
            optimizer.load_state_dict(save_state_dict['optimizer_state_dict'])
        reset_flag = False

        psnr_sum = 0

        for ii, sample_batch in tqdm(enumerate(train_dataloader)):
            cur_nimg += sample_batch['gt'].size(0)
            optimizer.zero_grad()

            gt_batch = sample_batch['gt'].cuda()
            y = sample_batch['meas'].cuda()
            Phi = sample_batch['mask'].cuda()
            Phi_sum = torch.sum(Phi, axis=3)
            Phi_sum[Phi_sum == 0] = 1
            with torch.no_grad():
                initial_point = cg_utils.initial_point(y, Phi, Phi_sum, gt_batch) # debug: .to(device=device)

            '''deep_eq_module is eq_utils.DEQFixedPoint(solver, forward_iterator, 
                m=anderson_m, beta=anderson_beta, lam=1e-2, max_iter=max_iters, tol=1e-5)'''
            reconstruction = deep_eq_module.forward(y, Phi, Phi_sum, initial_point=initial_point)
            loss = loss_function(reconstruction, gt_batch)
            if np.isnan(loss.item()):
                print('Loss is nan!')
                reset_flag = True
                break
            loss.backward()
            optimizer.step()
            if ii == 0:
                previous_loss = loss.item()

            PSNR = peak_signal_noise_ratio(reconstruction.clip(0, 1).cpu().detach().numpy(),
                                           gt_batch.cpu().detach().numpy())
            psnr_sum += PSNR

            ## write log
            stats_dict = OrderedDict([
                ('main/PSNR', PSNR),
                ('main/loss', loss.mean().item()),
                ('config/lr', optimizer.param_groups[0]['lr']),
                ('main/best_PSNR', best_psnr),
            ])
            timestamp = time.time()
            if stats_tfevents is not None:
                global_step = int(cur_nimg)
                walltime = timestamp - start_time
                for name, value in stats_dict.items():
                    stats_tfevents.add_scalar(name, value, global_step=global_step, walltime=walltime)
                stats_tfevents.flush()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                            " Loss: " + str(loss.cpu().detach().numpy()) + " PSNR: %2.2f dB" % PSNR + \
                            " best PSNR (test): %2.2f dB" % best_psnr + \
                            " lr: %.8f" % optimizer.param_groups[0]['lr']
                print(logging_string, flush=True)

            if (ii+1) % save_every_n_steps == 0:
                # cv2.imwrite(
                #     train_img_path + '%02d_%05d_measurement.png' % (epoch, ii),
                #     tensor_to_np(y[0]))
                # for frame_id in range(8):
                #     cv2.imwrite(train_img_path + '%02d_%05d_gt_%d.png' % (epoch, ii, frame_id),
                #                 tensor_to_np(gt_batch[0, :, :, frame_id]))
                #     cv2.imwrite(
                #         train_img_path + '%02d_%05d_reconstruction_%d.png' % (epoch, ii, frame_id),
                #         tensor_to_np(reconstruction[0, :, :, frame_id]))
                    # cv2.imwrite(train_img_path + '%02d_%05d_init_%d.png' % (epoch, ii, frame_id),
                    #             tensor_to_np(initial_point[0, :, :, frame_id]))
                cur_psnr, all_images = test_solver_sci(test_dataloader=test_dataloader,
                    deep_eq_module=deep_eq_module, save_img_path=best_img_path,
                    verbose=True, save_image=False)
                if cur_psnr > best_psnr:
                    best_psnr = cur_psnr
                    for k in all_images:
                        cv2.imwrite(k, all_images[k])
                    # save the best checkpoint
                    print('saving best model')
                    torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                                'epoch': epoch,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict()
                                }, save_model_path + 'best.ckpt')


        avg_psnr = psnr_sum / len(train_dataloader)
        print('avg PSNR in epoch %d: %.2f dB' %(epoch, avg_psnr))
        

        if (previous_loss - loss.item()) / previous_loss < -10.0 or np.isnan(loss.item()):
            reset_flag = True

        scheduler.step()

        if not reset_flag:
            torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_model_path + 'epoch_%d.ckpt' % epoch)
            print('dict saved!')
        test_solver_sci(test_dataloader=test_dataloader,
            deep_eq_module=deep_eq_module, save_img_path=test_img_path)

def test_solver_sci(deep_eq_module, test_dataloader=None, save_img_path=None, verbose=True, save_image=True):
    all_images = {}

    psnr_sum_for_avg = 0
    num_for_avg = 0
    for ii, sample_batch in enumerate(test_dataloader):

        gt_batch = sample_batch['gt'].cuda()
        y_batch = sample_batch['meas'].cuda()
        Phi = sample_batch['mask'].cuda()
        Phi_sum = torch.sum(Phi, axis=3)
        Phi_sum[Phi_sum == 0] = 1

        file_name = sample_batch['file']

        if ('drop' in file_name[0]) or ('runner' in file_name[0]):
            y_batch = y_batch[:,:,:,0].unsqueeze(3)
        psnr_sum = 0
        bsz, h, w, f = y_batch.shape
        for fi in range(f):
            gt = gt_batch[:,:,:,fi*8:(fi+1)*8]
            y = y_batch[:,:,:,fi]

            with torch.no_grad():
                initial_point = cg_utils.initial_point(y, Phi, Phi_sum, gt_batch).cuda()

            '''deep_eq_module is eq_utils.DEQFixedPoint(solver, forward_iterator, 
                m=anderson_m, beta=anderson_beta, lam=1e-2, max_iter=max_iters, tol=1e-5)'''

            reconstruction = deep_eq_module.forward(y, Phi, Phi_sum, initial_point=initial_point, train_flag = False)
            PSNR = peak_signal_noise_ratio(reconstruction.clip(0, 1).cpu().detach().numpy(),
                                           gt.cpu().detach().numpy())

            for frame_id in range(8):
                all_images[save_img_path + '%s_reconstruction_%d.png' % (file_name[0], fi*8 + frame_id)] = \
                    tensor_to_np(reconstruction[0, :, :, frame_id])
                # cv2.imwrite(
                #     save_img_path + '%s_reconstruction_%d.png' % (file_name[0], fi*8 + frame_id),
                #     tensor_to_np(reconstruction[0, :, :, frame_id]))

            psnr_sum += PSNR
        current_psnr = psnr_sum/f
        psnr_sum_for_avg += current_psnr
        num_for_avg += 1
        if verbose:
            print(file_name, '  PSNR: %.2f dB' % current_psnr)
    avg_psnr = psnr_sum_for_avg/num_for_avg
    if verbose:
        print('---------------------------------', 'Total Average PSNR: %.2f dB' % avg_psnr)
    if save_image:
        for k in all_images:
            cv2.imwrite(k, all_images[k])

    return avg_psnr, all_images

    #####################TEST##########################
    # loss_accumulator = []
    # mse_loss = torch.nn.MSELoss()
    # for ii, sample_batch in enumerate(test_dataloader):
    #     sample_batch = sample_batch.to(device=device)
    #     y = measurement_process(sample_batch)
    #     initial_point = y
    #     reconstruction = solver(initial_point, iterations=6)
    #
    #     reconstruction = torch.clamp(reconstruction, -1 ,1)
    #
    #     loss = mse_loss(reconstruction, sample_batch)
    #     loss_logger = loss.cpu().detach().numpy()
    #     loss_accumulator.append(loss_logger)
    #
    # loss_array = np.asarray(loss_accumulator)
    # loss_mse = np.mean(loss_array)
    # PSNR = -10 * np.log10(loss_mse)
    # percentiles = np.percentile(loss_array, [25,50,75])
    # percentiles = -10.0*np.log10(percentiles)
    # print("TEST LOSS: " + str(sum(loss_accumulator) / len(loss_accumulator)), flush=True)
    # print("MEAN TEST PSNR: " + str(PSNR), flush=True)
    # print("TEST PSNR QUARTILES AND MEDIAN: " + str(percentiles[0]) +
    #       ", " + str(percentiles[1]) + ", " + str(percentiles[2]), flush=True)


#############################################################################
##########################################################################
# gt_location = '/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/gt/'
# meas_location = '/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/measurement/'
# mask_location = "/home/yaping/projects/data/DAVIS-2017-trainval-480p/DAVIS/matlab/mask.mat"
# def transform(input, mean=0.5, std=0.5):
#     return (input-mean)/std
#
# batch_size = 1
# from utils.sci_dataloader import SCITrainingDatasetSubset
# dataset = SCITrainingDatasetSubset(gt_location, meas_location, mask_location, transform=transform)
# # dataset = CelebaTrainingDatasetSubset(data_location, subset_indices=initial_indices, transform=transform)
# dataloader = torch.utils.data.DataLoader(
#     dataset=dataset, batch_size=batch_size, shuffle=False, drop_last=True,
# )
# save_img_path = '/home/yaping/projects/deep_equilibrium_inverse/save/deq_test/test_init_point/'
# for ii, sample_batch in enumerate(dataloader):
#     if ii == 9:
#         gt_batch = sample_batch['gt']
#         # y = measurement_process(sample_batch)
#         y = sample_batch['meas']
#         Phi = sample_batch['mask']
#         Phi_sum = torch.sum(Phi, axis=3)
#         Phi_sum[Phi_sum == 0] = 1
#         print(y.shape, Phi.shape, gt_batch.shape)
#         with torch.no_grad():
#             initial_point = cg_utils.initial_point(y, Phi, Phi_sum, gt_batch)
#         for frame_id in range(8):
#             cv2.imwrite(save_img_path + '%05d_init_%d.png' % (ii,frame_id),
#                 tensor_to_np(initial_point[0,:,:,frame_id]))
#             cv2.imwrite(save_img_path + '%05d_gt_%d.png' % (ii, frame_id),
#                         tensor_to_np(gt_batch[0, :, :, frame_id]))
#         exit()