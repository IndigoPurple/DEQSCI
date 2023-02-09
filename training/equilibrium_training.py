import torch
import numpy as np
from solvers import equilibrium_utils as eq_utils
from torch import autograd

def train_solver(single_iterate_solver, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0):

    n_iterations = [5]*n_epochs
    for ee in range(n_epochs):
        if ee >= 5:
            n_iterations[ee] = 5
        if ee >= 8:
            n_iterations[ee] = 8
        if ee >= 10:
            n_iterations[ee] = 10
        if ee >= 12:
            n_iterations[ee] = 15
        if ee >= 15:
            n_iterations[ee] = 20

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
            reconstruction = eq_utils.get_equilibrium_point(y, single_iterate_solver, max_iterations=n_iterations[epoch])

            reconstruction = torch.clamp(reconstruction, -1, 1)
            loss = loss_function(reconstruction, sample_batch)

            if epoch < 2:
                loss.backward()
                optimizer.step()
            else:

                # f_zstar = single_iterate_solver(static_zstar)

                # delf_deltheta = torch.autograd.grad(inputs=static_zstar, outputs=f_zstar,
                #                                     grad_outputs=torch.ones_like(f_zstar))

                dell_delz = torch.autograd.grad(inputs=reconstruction, outputs=loss,
                                                grad_outputs=torch.ones_like(loss))[0]

                delf_deltheta_invJ = eq_utils.conjugate_gradient_equilibriumgrad(b=dell_delz,
                                                                                 input_z=reconstruction,
                                                                                 f_function=single_iterate_solver,
                                                                                 n_iterations=5)

                # loss.backward(retain_graph=True)
                torch.autograd.backward(tensors=reconstruction, grad_tensors=delf_deltheta_invJ)
                optimizer.step()

            # exit()
            # for name, param in single_iterate_solver.named_parameters():
            #     jj = 0
            #     if param.grad is not None:
            #         print(name)
            #         print(param.shape)
            #         print(param.grad.shape)
            #         jj+=1
            #         if jj == 2:
            #             break
            # exit()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy())
                print(logging_string, flush=True)

        if scheduler is not None:
            scheduler.step(epoch)

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

def train_solver_mnist(single_iterate_solver, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0):

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

    for epoch in range(start_epoch, n_epochs):
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

            def jacobian_vector_product(f, z, v):
                z = z.detach().requires_grad_()
                v = v.detach().requires_grad_()

                vjp_val = autograd.grad(f(z), z, v, create_graph=True)[0]
                return vjp_val
                # jvp_val = autograd.grad(vjp_val, v, v.detach(), create_graph=True)[0]
                # return jvp_val

            if epoch < 10:
                reconstruction = eq_utils.get_equilibrium_point(y, single_iterate_solver,
                                                                max_iterations=n_iterations[epoch])

                reconstruction = torch.clamp(reconstruction, 0, 1)
                loss = loss_function(reconstruction, sample_batch)
                loss.backward()
                # for name, param in single_iterate_solver.named_parameters():
                #     if param.grad is not None:
                #         print(name)
                #         print(param.grad.shape)

                # torch.autograd.backward(reconstruction, grad_tensors=reconstruction)
                # for name, param in single_iterate_solver.named_parameters():
                #     if param.grad is not None:
                #         print(name)
                #         print(param.grad.shape)

                # print(autograd.functional.jacobian(single_iterate_solver, reconstruction).shape)
                # exit()
                optimizer.step()
            else:
                exit()

                # f_zstar = single_iterate_solver(static_zstar)
                # reconstruction = single_iterate_solver(sample_batch)

                # reconstruction = eq_utils.get_equilibrium_point(y, single_iterate_solver,
                #                                                 max_iterations=n_iterations[epoch])
                reconstruction = eq_utils.get_equilibrium_point(y, single_iterate_solver,
                                                                max_iterations=n_iterations[epoch])

                reconstruction = torch.clamp(reconstruction, 0, 1)
                loss = loss_function(reconstruction, sample_batch)

                # delf_deltheta = torch.autograd.grad(inputs=static_zstar, outputs=f_zstar,
                #                                     grad_outputs=torch.ones_like(f_zstar))

                dell_delz = torch.autograd.grad(inputs=reconstruction, outputs=loss,
                                                grad_outputs=torch.ones_like(loss))[0]



                # delf_deltheta_invJ = eq_utils.conjugate_gradient_equilibriumgrad(b=dell_delz,
                #                                                                  input_z=sample_batch.requires_grad_(),
                #                                                                  f_function=single_iterate_solver,
                #                                                                  n_iterations=10)
                # torch.autograd.backward(tensors=single_iterate_solver(sample_batch), grad_tensors=delf_deltheta_invJ)

                delf_deltheta_invJ = eq_utils.conjugate_gradient_equilibriumgrad(b=dell_delz,
                                                                                 input_z=reconstruction,
                                                                                 f_function=single_iterate_solver,
                                                                                 n_iterations=10)
                torch.autograd.backward(tensors=reconstruction, grad_tensors=-delf_deltheta_invJ)

                torch.nn.utils.clip_grad_norm_(single_iterate_solver.parameters(), 1.0)

                # for name, param in single_iterate_solver.named_parameters():
                #     if param.grad is not None:
                #         print(name)
                #         print(torch.norm(param.grad))

                # jacobian_vect_product = delf_deltheta_invJ#.flatten(start_dim=1)

                # vector_jacobian_product = jacobian_vector_product(single_iterate_solver, reconstruction, jacobian_vect_product)
                # print(vector_jacobian_product.shape)
                # exit()

                # gradient = torch.reshape(jacobian_vect_product, (8,1,28,28))
                # gradient = torch.squeeze(torch.mean(gradient, dim=0))
                # print(single_iterate_solver.nonlinear_op.linear_layer(torch.flatten(delf_deltheta_invJ, start_dim=1)))
                # print(delf_deltheta_invJ.shape)
                #
                # exit()

                # torch.autograd.backward(tensors=reconstruction, grad_tensors=delf_deltheta_invJ)
                optimizer.step()

            # exit()
            # for name, param in single_iterate_solver.named_parameters():
            #     jj = 0
            #     if param.grad is not None:
            #         print(name)
            #         print(param.shape)
            #         print(param.grad.shape)
            #         jj+=1
            #         if jj == 2:
            #             break
            # exit()

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
