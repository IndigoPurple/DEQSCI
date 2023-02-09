import torch.nn as nn
import torch
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import imageio
import numpy as np
from PIL import Image

def complex_conj(x):
    assert x.shape[1] == 2
    return torch.stack((x[:,0, ...], -x[:,1,...]), dim=1)

def torchdotproduct(x,y):
    # if complexdata:
    # y = complex_conj(y)
    return torch.sum(x*y,dim=[1,2,3])

def single_cg_iteration(x, d, g, b, ATA, regularization_lambda):

    def regATA(input, ATA):
        return ATA(input) + regularization_lambda*input

    Qd = regATA(d, ATA)
    dQd = torchdotproduct(d, Qd)
    alpha = -torchdotproduct(g,d) / dQd
    alpha = alpha.view((-1,1,1,1))
    x = x + alpha * d
    g = regATA(x, ATA) - b
    gQd = torchdotproduct(g, Qd)
    beta = gQd / dQd
    beta = beta.view((-1,1,1,1))
    d = -g + beta*d
    return x, d, g

# This function solves the system ATA x = ATy, where initial_point is supposed
# to be ATy. This can be backpropagated through.
def conjugate_gradient(initial_point, ATA, regularization_lambda, n_iterations=10):
    x = torch.zeros_like(initial_point)
    d = initial_point
    g = -d
    for ii in range(n_iterations):
        x, d, g = single_cg_iteration(x, d, g, initial_point, ATA, regularization_lambda)
    return x

def complex_dotproduct(x, y):
    return torchdotproduct(complex_conj(x), y)

def single_cg_iteration_MRI(rTr, x, r, p, ATA, regularization_lambda):

    batch_size = x.shape[0]
    def regATA(input):
        return ATA(input) + regularization_lambda*input

    Ap = regATA(p)

    rTr = rTr.view(batch_size, 1, 1, 1)
    alpha = rTr / complex_dotproduct(p, Ap).view(batch_size, 1, 1, 1)

    x_new = x + alpha * p
    r_new = r - alpha * Ap
    rTr_new = complex_dotproduct(r_new, r_new)
    rTr_new = rTr_new.view(batch_size, 1, 1, 1)

    beta = rTr_new / rTr
    p_new = r + beta * p
    return rTr_new, x_new, r_new, p_new

def conjugate_gradient_MRI(initial_point, ATA, regularization_lambda, n_iterations=10):
    '''Strightforward implementation of MoDLs code'''
    x = torch.zeros_like(initial_point)
    r = initial_point
    p = initial_point
    rTr = complex_dotproduct(r, r)
    for ii in range(n_iterations):
        rTr, x, r, p = single_cg_iteration_MRI(rTr, x, r, p, ATA, regularization_lambda)
    return x

def jacobian_vector_product(g, z, v):
    JTv = torch.autograd.grad(outputs=g, inputs=z, grad_outputs=v)[0]
    return JTv

def conjugate_gradient_equilibriumgrad(b, input_z, f_function, n_iterations=10):
    initial_guess = b.clone()
    x_k = initial_guess
    r_k = b
    p_k = r_k
    batch_size = b.shape[0]
    g = f_function(input_z) - input_z

    for ii in range(n_iterations):
        # g = f_function(initial_guess) - initial_guess
        # Ap_k = jacobian_vector_product(g, input_z, x_k)
        Ap_k = (torch.autograd.grad(outputs=g, inputs=input_z, grad_outputs=x_k, retain_graph=True)[0] + 0.00001 * x_k)
        rTr_k = torchdotproduct(r_k, r_k)
        rTr_k = rTr_k.view(batch_size, 1, 1, 1)

        pAp_k = torchdotproduct(Ap_k, p_k)
        pAp_k = pAp_k.view(batch_size, 1, 1, 1)

        alpha = rTr_k / pAp_k

        x_k = x_k + alpha * p_k
        r_kplus1 = r_k - alpha * Ap_k
        rTr_kplus1 = torchdotproduct(r_kplus1, r_kplus1)
        rTr_kplus1 = rTr_kplus1.view(batch_size, 1, 1, 1)

        beta = rTr_k / rTr_kplus1
        p_k = r_kplus1 + beta * p_k
        r_k = r_kplus1
    return x_k

#tt= 0
def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0):
    """ Anderson acceleration for fixed point iteration.
    This was taken from the Deep Equilibrium tutorial here: http://implicit-layers-tutorial.org/deep_equilibrium_models/
    """

    #global tt
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape(x0.shape)).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    res = []
    current_k = 0
    past_iterate = x0
    for k in range(2, max_iter):
        current_k = k
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        current_iterate = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape(x0.shape)).reshape(bsz, -1)
        res.append((F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item()))

        if (res[-1] < tol):
            break
    #tt += bsz
    return X[:, current_k % m].view_as(x0), res

'''f is EquilibriumProxGradSCI, x0 is init_point'''
def andersonexp(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0): # yaping mute
# def andersonexp(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta=1.0): #yaping add
    """ Anderson acceleration for fixed point iteration. """
    # global tt
    bsz, d, H, W = x0.shape
    X = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d * H * W, dtype=x0.dtype, device=x0.device)
    # yaping mute
    X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
    X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape(x0.shape)).reshape(bsz, -1)
    # yaping add
    # X[:, 0], F[:, 0] = x0.reshape(bsz, -1), f(x0, x0).reshape(bsz, -1)
    # X[:, 1], F[:, 1] = F[:, 0], f(F[:, 0].reshape(x0.shape), y).reshape(bsz, -1)

    H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
    H[:, 0, 1:] = H[:, 1:, 0] = 1
    y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
    y[:, 0] = 1

    current_k = 0
    for k in range(2, max_iter):
        current_k = k
        n = min(k, m)
        G = F[:, :n] - X[:, :n]
        H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[
            None]
        alpha = torch.solve(y[:, :n + 1], H[:, :n + 1, :n + 1])[0][:, 1:n + 1, 0]  # (bsz x n)

        X[:, k % m] = beta * (alpha[:, None] @ F[:, :n])[:, 0] + (1 - beta) * (alpha[:, None] @ X[:, :n])[:, 0]
        F[:, k % m] = f(X[:, k % m].reshape(x0.shape)).reshape(bsz, -1)
        res = (F[:, k % m] - X[:, k % m]).norm().item() / (1e-5 + F[:, k % m].norm().item())

        if (res < tol):
            break
    # tt += bsz
    return X[:, current_k % m].view_as(x0), res

def L2Norm(x):
    return torch.sum(x**2, dim=[1,2,3], keepdim=True)

def epsilon2(f, x0, max_iter=50, tol=1e-2, lam=1e-4):

    x = x0

    for k in range(max_iter):
        f_x = f(x)
        delta_x = f_x - x
        delta_f = f(f_x) - f_x
        delta2_x = delta_f - delta_x
        # term1 = delta_f * L2Norm(delta_x)
        # term2 = delta_x * L2Norm(delta_f)
        x_new = f_x + (delta_f * L2Norm(delta_x) - delta_x * L2Norm(delta_f)) / (L2Norm(delta2_x) + lam)
        residual = (x_new - x).norm().item() / x_new.norm().item()
        x = x_new
        if (residual < tol):
            break

    return x, residual

def forward_iteration(f, x0, max_iter=50, tol=1e-5):
    f0 = f(x0)
    res = []
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        res.append((f0 - x).norm().item() / (1e-7 + f0.norm().item()))
        if (res[-1] < tol):
            break
    return f0, res

def forward_iteration_plot(f, x0, max_iter=50, tol=1e-5):
    f0 = f(x0)
    res = []
    fig = plt.figure()
    for k in range(max_iter):
        x = f0
        f0 = f(x)
        # sub = fig.add_subplot(10,10, k)
        # plt.imshow(f0[0, : , :, :].detach().cpu().numpy())
        # plt.show()
        res.append((f0 - x).norm().item() / (1e-7 + f0.norm().item()))
        if (res[-1] < tol):
            break
    plt.show()
    return f0, res


class DEQFixedPoint(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x, initial_point = None):
        if initial_point is None:
            init_point = torch.zeros_like(x)
        else:
            init_point = initial_point
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            '''solver is anderson, 
                f is EquilibriumProxGradSCI, 
                init_point is At(y,Phi),
                x is y, measurement'''
            '''aderson(EquilibriumProxGradSCI, init_point, args)'''
            z, self.forward_res = self.solver(lambda z: self.f(z, x), init_point, **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z


class DEQFixedPointExp(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x, initial_point = None):
        if initial_point is None:
            init_point = torch.zeros_like(x)
        else:
            init_point = initial_point
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), init_point, **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g, self.backward_res = self.solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
                                               grad, **self.kwargs)
            return g

        z.register_hook(backward_hook)
        return z

class DEQFixedPointTest(nn.Module):
    def __init__(self, f, solver, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.kwargs = kwargs

    def forward(self, x, truth = None, initial_point = None):
        if initial_point is None:
            init_point = torch.zeros_like(x)
        else:
            init_point = initial_point
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), init_point, **self.kwargs)

        return z

def neumann_iteration(f, x0,k=10):
    accumulator = x0
    current_iterate = x0
    for _ in range(k):
        current_iterate = f(current_iterate)
        accumulator = accumulator + current_iterate

    return accumulator


class DEQFixedPointNeumann(nn.Module):
    def __init__(self, f, solver, neumann_k, **kwargs):
        super().__init__()
        self.f = f
        self.solver = solver
        self.neumann_k = neumann_k
        self.kwargs = kwargs

    def forward(self, x):
        # compute forward pass and re-engage autograd tape
        with torch.no_grad():
            z, self.forward_res = self.solver(lambda z: self.f(z, x), torch.zeros_like(x), **self.kwargs)
        z = self.f(z, x)

        # set up Jacobian vector product (without additional forward calls)
        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)

        def backward_hook(grad):
            g = neumann_iteration(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0],
                                               grad, self.neumann_k)
            return g

        z.register_hook(backward_hook)
        return z

def get_equilibrium_point(solver, z, max_iterations=50, tolerance = 0.001):
    old_iterate = z
    for iteration in range(max_iterations):
        new_iterate = solver(old_iterate)
        res = (new_iterate-old_iterate).norm().item() / (1e-5 + new_iterate.norm().item())
        old_iterate = new_iterate
        if res < 1e-3:
            break
    return new_iterate, new_iterate

def get_equilibrium_point_plot(solver, z, truth, max_iterations=50, tolerance = 0.001):
    running_iterate = z
    # fig = plt.figure()
    jj = 0
    for iteration in range(max_iterations):
        # if iteration % 10 == 0:
        #     sub = fig.add_subplot(2, 5, jj+1)
        #     img_to_show = torch.abs(running_iterate[0, :, :, :] - truth[0,:,:,:])*5.0
        #     # plt.imshow((running_iterate[0, :, :, :].permute(1,2,0).cpu().detach().numpy() + 1.0) / 2.0)
        #     # plt.show()
        #     # sub.imshow((img_to_show.permute(1,2,0).detach().cpu().numpy() + 1.0)/2.0)
        #     sub.imshow(img_to_show.permute(1,2,0).detach().cpu().numpy())
        #
        #     jj += 1
        running_iterate = solver(running_iterate)
    # plt.show()

    return running_iterate, running_iterate
