import torch
import numpy as np
from solvers import new_equilibrium_utils as eq_utils
from torch import autograd

def train_solver(single_iterate_solver, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs, forward_iterator, iterator_kwargs,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0):

    forward_iterator = eq_utils.anderson
    deep_eq_module = eq_utils.DEQFixedPoint(single_iterate_solver, forward_iterator, iterator_kwargs)

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

def train_solver_noanderson(single_iterate_solver, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0, max_iters=100):

    forward_iterator = eq_utils.forward_iteration
    deep_eq_module = eq_utils.DEQFixedPoint(single_iterate_solver, solver=forward_iterator,
                                            max_iter=max_iters, tol=1e-3)

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
    deep_eq_module = eq_utils.DEQFixedPointNeumann(single_iterate_solver, neumann_k=100, solver=forward_iterator,
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
