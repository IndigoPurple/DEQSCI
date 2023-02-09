import torch
import numpy as np
from solvers import new_equilibrium_utils as eq_utils
from torch import autograd
from utils import cg_utils
# yaping
import cv2
from PIL import Image

from skimage.metrics import peak_signal_noise_ratio

# yaping
def tensor_to_np(tensor):
    # tensor = np.clip(tensor.cpu(), -1, 1) +1 / 2
    # img = tensor.mul(255).byte()
    img = tensor.clip(-1, 1).cpu().detach().numpy().squeeze().transpose(1,2,0)[:,:,::-1] * 255.
    return img

def tensor_to_PIL(tensor):
    nu = tensor.cpu().numpy()
    image = Image.fromarray(nu)
    return  image

def train_solver(single_iterate_solver, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs, deep_eq_module,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0):

    for epoch in range(start_epoch, n_epochs):

        if epoch % save_every_n_epochs == 0:
            if use_dataparallel:
                torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)

        for ii, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            sample_batch = sample_batch.to(device=device)
            y = measurement_process(sample_batch)
            single_iterate_solver.set_initial_point(y)
            reconstruction = deep_eq_module.forward(y)
            loss = loss_function(reconstruction, sample_batch)
            loss.backward()
            optimizer.step()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy())
                print(logging_string, flush=True)

        if scheduler is not None:
            scheduler.step(epoch)
        if use_dataparallel:
            torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, save_location)
        else:
            torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, save_location)

def train_solver_precond(single_iterate_solver, train_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs, deep_eq_module,
                 use_dataparallel=False, device='cpu', scheduler=None, noise_sigma=0.000001, precond_iterates=100,
                 print_every_n_steps=2, save_every_n_epochs=5, start_epoch=0, forward_operator = None,
                         test_dataloader = None):
    previous_loss = 10.0
    reset_flag = False

    for epoch in range(start_epoch, n_epochs):

        if reset_flag:
            save_state_dict = torch.load(save_location)
            single_iterate_solver.load_state_dict(save_state_dict['solver_state_dict'])
            optimizer.load_state_dict(save_state_dict['optimizer_state_dict'])
        reset_flag = False

        for ii, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            sample_batch = sample_batch[0].to(device=device)
            target_img = sample_batch[1].to(device=device)
            y = measurement_process(sample_batch)
            if forward_operator is not None:
                with torch.no_grad():
                    initial_point = forward_operator.adjoint(y)
                reconstruction = deep_eq_module.forward(y, initial_point=initial_point)
            else:
                reconstruction = deep_eq_module.forward(y)
            loss = loss_function(reconstruction, target_img)
            if np.isnan(loss.item()):
                reset_flag = True
                break
            loss.backward()
            optimizer.step()

            if ii == 0:
                previous_loss = loss.item()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy())
                print(logging_string, flush=True)
            if ii % 200 == 0:
                if use_dataparallel:
                    torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                                'epoch': epoch+1,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict()
                                }, save_location)
                else:
                    torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                                'epoch': epoch+1,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict()
                                }, save_location)

        if (previous_loss - loss.item()) / previous_loss < -10.0 or np.isnan(loss.item()):
            reset_flag = True

        if scheduler is not None:
            scheduler.step(epoch)
        if not reset_flag:
            if use_dataparallel:
                # torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                #             'epoch': epoch,
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': scheduler.state_dict()
                #             }, save_location + "_" + str(epoch))
                torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                # torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                #             'epoch': epoch,
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': scheduler.state_dict()
                #             }, save_location + "_" + str(epoch))
                torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)

def train_solver_precond1(single_iterate_solver, train_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs, deep_eq_module,
                 use_dataparallel=False, device='cpu', scheduler=None, noise_sigma=0.000001, precond_iterates=100,
                 print_every_n_steps=100, save_every_n_epochs=5, start_epoch=0, forward_operator = None,
                         test_dataloader = None):
    previous_loss = 10.0
    reset_flag = False

    for epoch in range(start_epoch, n_epochs):

        #yaping
        if reset_flag:
            save_state_dict = torch.load(save_location)
            single_iterate_solver.load_state_dict(save_state_dict['solver_state_dict'])
            optimizer.load_state_dict(save_state_dict['optimizer_state_dict'])
        reset_flag = False

        for ii, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            sample_batch = sample_batch.to(device=device)
            y = measurement_process(sample_batch)
            if forward_operator is not None:
                with torch.no_grad():
                    initial_point = cg_utils.conjugate_gradient(initial_point=forward_operator.adjoint(y),
                                                                 ATA=forward_operator.gramian,
                                                                 regularization_lambda=noise_sigma, n_iterations=precond_iterates)
                reconstruction = deep_eq_module.forward(y, initial_point=initial_point)
            else:
                reconstruction = deep_eq_module.forward(y)
            loss = loss_function(reconstruction, sample_batch)
            if np.isnan(loss.item()):
                reset_flag = True
                break
            loss.backward()
            optimizer.step()
            # print('ii ',ii)
            if ii == 0:
                previous_loss = loss.item()

            # yaping
            # if ii % print_every_n_steps == 0:
            #     logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
            #                      " Loss: " + str(loss.cpu().detach().numpy())
            #     print(logging_string, flush=True)

            if ii % print_every_n_steps == 0:
                # yaping begin
                cv2.imwrite('/home/yaping/projects/deep_equilibrium_inverse/save/deq_test/1_d/%05d_gt.png' % ii,
                            tensor_to_np(sample_batch[0]))
                # cv2.waitKey()
                PSNR_input = peak_signal_noise_ratio(y.clip(-1, 1).cpu().detach().numpy(),
                                               sample_batch.cpu().detach().numpy())
                PSNR = peak_signal_noise_ratio(reconstruction.clip(-1, 1).cpu().detach().numpy(),
                                               sample_batch.cpu().detach().numpy())
                cv2.imwrite(
                    '/home/yaping/projects/deep_equilibrium_inverse/save/deq_test/1_d/%05d_measurement.png' % ii,
                    tensor_to_np(y[0]))
                cv2.imwrite(
                    '/home/yaping/projects/deep_equilibrium_inverse/save/deq_test/1_d/%05d_reconstruction.png' % ii,
                    tensor_to_np(reconstruction[0]))
                # cv2.imwrite(
                #     '/home/yaping/projects/deep_equilibrium_inverse/save/trained_unet_test/2/%05d_net_output.png' % ii,
                #     tensor_to_np(net_output[0]))
                # yaping end
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy()) + " PSNR: input " + str(PSNR_input) + ' output '+ str(PSNR)
                print(logging_string, flush=True)

            if ii % 200 == 0:
                if use_dataparallel:
                    torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                                'epoch': epoch+1,
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict()
                                }, save_location)
                else:
                    #yaping
                    # torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                    #             'epoch': epoch+1,
                    #             'optimizer_state_dict': optimizer.state_dict(),
                    #             'scheduler_state_dict': scheduler.state_dict()
                    #             }, save_location)
                    continue

        if (previous_loss - loss.item()) / previous_loss < -10.0 or np.isnan(loss.item()):
            reset_flag = True

        if scheduler is not None:
            scheduler.step(epoch)
        #yaping
        # print(ii)
        # if ii == 1:
        #     torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
        #                 'epoch': epoch,
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #                 'scheduler_state_dict': scheduler.state_dict()
        #                 }, save_location)
        #     print('OK')

        if not reset_flag:
            if use_dataparallel:
                # torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                #             'epoch': epoch,
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': scheduler.state_dict()
                #             }, save_location + "_" + str(epoch))
                torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                # torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                #             'epoch': epoch,
                #             'optimizer_state_dict': optimizer.state_dict(),
                #             'scheduler_state_dict': scheduler.state_dict()
                #             }, save_location + "_" + str(epoch))
                torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)


def train_solver_mnist(single_iterate_solver, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0, max_iters=100):

    n_iterations = [5]*n_epochs
    for ee in range(n_epochs):
        if ee >= 20:
            n_iterations[ee] = 5
        if ee >= 23:
            n_iterations[ee] = 7
        if ee >= 28:
            n_iterations[ee] = 9
        if ee >= 38:
            n_iterations[ee] = 11
        if ee >= 44:
            n_iterations[ee] = 13
        if ee >= 50:
            n_iterations[ee] = 20
        if ee >= 58:
            n_iterations[ee] = 30

    forward_iterator = eq_utils.anderson
    deep_eq_module = eq_utils.DEQFixedPoint(single_iterate_solver, solver=forward_iterator,
                                            m=5, lam=1e-4, max_iter=max_iters, tol=1e-3, beta=1.5)

    for epoch in range(start_epoch, n_epochs):

        # We are lucky to have
        if epoch % save_every_n_epochs == 0:
            if use_dataparallel:
                torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)

        for ii, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            sample_batch = sample_batch[0].to(device=device)
            y = measurement_process(sample_batch)
            single_iterate_solver.set_initial_point(y)
            reconstruction = deep_eq_module.forward(y)
            loss = loss_function(reconstruction, sample_batch)
            loss.backward()
            optimizer.step()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy())
                print(logging_string, flush=True)

        if scheduler is not None:
            scheduler.step(epoch)
        if use_dataparallel:
            torch.save({'solver_state_dict': single_iterate_solver.module.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, save_location)
        else:
            torch.save({'solver_state_dict': single_iterate_solver.state_dict(),
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict()
                        }, save_location)

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
