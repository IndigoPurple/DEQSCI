import torch
import numpy as np

def train_solver(solver, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs, forward_model=None,
                 use_dataparallel=False, device='cpu', scheduler=None, n_blocks=10,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0):

    n_blocks = 6

    for epoch in range(start_epoch, n_epochs):

        # We are lucky to have
        if epoch % save_every_n_epochs == 0:
            if use_dataparallel:
                torch.save({'solver_state_dict': solver.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                torch.save({'solver_state_dict': solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)

        for ii, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            sample_batch = sample_batch.to(device=device)
            y = measurement_process(sample_batch)
            if forward_model is None:
                initial_point = y
            else:
                initial_point = forward_model.adjoint(y)
            reconstruction = solver(initial_point, iterations=n_blocks)

            reconstruction = torch.clamp(reconstruction, -1 ,1)

            loss = loss_function(reconstruction, sample_batch)

            loss.backward()
            optimizer.step()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy())
                print(logging_string, flush=True)

        if scheduler is not None:
            scheduler.step(epoch)

    #####################TEST##########################
    loss_accumulator = []
    mse_loss = torch.nn.MSELoss()
    for ii, sample_batch in enumerate(test_dataloader):
        sample_batch = sample_batch.to(device=device)
        y = measurement_process(sample_batch)
        initial_point = y
        reconstruction = solver(initial_point, iterations=n_blocks)

        reconstruction = torch.clamp(reconstruction, -1 ,1)

        loss = mse_loss(reconstruction, sample_batch)
        loss_logger = loss.cpu().detach().numpy()
        loss_accumulator.append(loss_logger)

    loss_array = np.asarray(loss_accumulator)
    loss_mse = np.mean(loss_array)
    PSNR = -10 * np.log10(loss_mse)
    percentiles = np.percentile(loss_array, [25,50,75])
    percentiles = -10.0*np.log10(percentiles)
    print("TEST LOSS: " + str(sum(loss_accumulator) / len(loss_accumulator)), flush=True)
    print("MEAN TEST PSNR: " + str(PSNR), flush=True)
    print("TEST PSNR QUARTILES AND MEDIAN: " + str(percentiles[0]) +
          ", " + str(percentiles[1]) + ", " + str(percentiles[2]), flush=True)


def train_solver_mnist(solver, train_dataloader, test_dataloader,
                 measurement_process, optimizer,
                 save_location, loss_function, n_epochs,
                 use_dataparallel=False, device='cpu', scheduler=None,
                 print_every_n_steps=10, save_every_n_epochs=5, start_epoch=0):

    for epoch in range(start_epoch, n_epochs):

        # We are lucky to have
        if epoch % save_every_n_epochs == 0:
            if use_dataparallel:
                torch.save({'solver_state_dict': solver.module.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)
            else:
                torch.save({'solver_state_dict': solver.state_dict(),
                            'epoch': epoch,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }, save_location)

        for ii, sample_batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            sample_batch = sample_batch[0].to(device=device)
            y = measurement_process(sample_batch)
            initial_point = y
            reconstruction = solver(initial_point, iterations=6)

            reconstruction = torch.clamp(reconstruction, -1 ,1)

            loss = loss_function(reconstruction, sample_batch)

            loss.backward()
            optimizer.step()

            if ii % print_every_n_steps == 0:
                logging_string = "Epoch: " + str(epoch) + " Step: " + str(ii) + \
                                 " Loss: " + str(loss.cpu().detach().numpy())
                print(logging_string, flush=True)

        if scheduler is not None:
            scheduler.step(epoch)

    #####################TEST##########################
    loss_accumulator = []
    mse_loss = torch.nn.MSELoss()
    for ii, sample_batch in enumerate(test_dataloader):
        sample_batch = sample_batch[0].to(device=device)
        y = measurement_process(sample_batch)
        initial_point = y
        reconstruction = solver(initial_point, iterations=6)

        reconstruction = torch.clamp(reconstruction, -1 ,1)

        loss = mse_loss(reconstruction, sample_batch)
        loss_logger = loss.cpu().detach().numpy()
        loss_accumulator.append(loss_logger)

    loss_array = np.asarray(loss_accumulator)
    loss_mse = np.mean(loss_array)
    PSNR = -10 * np.log10(loss_mse)
    percentiles = np.percentile(loss_array, [25,50,75])
    percentiles = -10.0*np.log10(percentiles)
    print("TEST LOSS: " + str(sum(loss_accumulator) / len(loss_accumulator)), flush=True)
    print("MEAN TEST PSNR: " + str(PSNR), flush=True)
    print("TEST PSNR QUARTILES AND MEDIAN: " + str(percentiles[0]) +
          ", " + str(percentiles[1]) + ", " + str(percentiles[2]), flush=True)
