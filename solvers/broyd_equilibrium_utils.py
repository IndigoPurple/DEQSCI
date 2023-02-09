import torch.nn as nn
import torch
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import imageio
import numpy as np
from PIL import Image


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)  # First do an update with step size 1
    if phi_a0 <= phi0 + c1 * alpha0 * derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0 ** 2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:  # we are assuming alpha>0 is a descent direction
        factor = alpha0 ** 2 * alpha1 ** 2 * (alpha1 - alpha0)
        a = alpha0 ** 2 * (phi_a1 - phi0 - derphi0 * alpha1) - \
            alpha1 ** 2 * (phi_a0 - phi0 - derphi0 * alpha0)
        a = a / factor
        b = -alpha0 ** 3 * (phi_a1 - phi0 - derphi0 * alpha1) + \
            alpha1 ** 3 * (phi_a0 - phi0 - derphi0 * alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b ** 2 - 3 * a * derphi0))) / (3.0 * a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1 * alpha2 * derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2 / alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.
    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0) ** 2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]  # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new) ** 2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new

    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bij, bijd -> bd', x, part_Us)  # (N, threshold)
    return -x + torch.einsum('bd, bdij -> bij', xTU, part_VTs)  # (N, 2d, L'), but should really be (N, 1, (2d*L'))


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (N, 2d, L')
    # part_Us: (N, 2d, L', threshold)
    # part_VTs: (N, threshold, 2d, L')
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bdij, bij -> bd', part_VTs, x)  # (N, threshold)
    return -x + torch.einsum('bijd, bd -> bij', part_Us, VTx)  # (N, 2d, L'), but should really be (N, (2d*L'), 1)


def broyden(g, x0, threshold=9, eps=1e-5, ls=False):
    x0_shape = x0.shape
    x0 = x0.reshape((x0.shape[0], -1, 1))
    bsz, total_hsize, n_elem = x0.size()
    dev = x0.device

    x_est = x0  # (bsz, 2d, L')
    gx = g(x_est)  # (bsz, 2d, L')
    nstep = 0
    tnstep = 0
    LBFGS_thres = min(threshold, 27)

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, total_hsize, n_elem, LBFGS_thres).to(dev)
    VTs = torch.zeros(bsz, LBFGS_thres, total_hsize, n_elem).to(dev)
    update = gx
    new_objective = init_objective = torch.norm(gx).item()
    prot_break = False
    trace = [init_objective]
    new_trace = [-1]

    # To be used in protective breaks
    protect_thres = 1e6 * n_elem
    lowest = new_objective
    lowest_xest, lowest_gx, lowest_step = x_est, gx, nstep

    while new_objective >= eps and nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite + 1)
        new_objective = torch.norm(gx).item()
        trace.append(new_objective)
        try:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item())  # Relative residual
        except:
            new2_objective = torch.norm(delta_x).item() / (torch.norm(x_est - delta_x).item() + 1e-9)
        new_trace.append(new2_objective)
        if new_objective < lowest:
            lowest_xest, lowest_gx = x_est.clone().detach(), gx.clone().detach()
            lowest = new_objective
            lowest_step = nstep
        if new_objective < eps:
            # print(nstep)
            break
        if new_objective < 3 * eps and nstep > 30 and np.max(trace[-30:]) / np.min(trace[-30:]) < 1.3:
            # if there's hardly been any progress in the last 30 steps
            # print(nstep)
            break
        if new_objective > init_objective * protect_thres:
            # prot_break = True
            # print(nstep)
            break

        part_Us, part_VTs = Us[:, :, :, :(nstep - 1)], VTs[:, :(nstep - 1)]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bij, bij -> b', vT, delta_gx)[:, None, None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:, (nstep - 1) % LBFGS_thres] = vT
        Us[:, :, :, (nstep - 1) % LBFGS_thres] = u
        update = -matvec(Us[:, :, :, :nstep], VTs[:, :nstep], gx)

    Us, VTs = None, None
    lowest_xest = lowest_xest.reshape(x0_shape)
    return lowest_xest, torch.norm(lowest_gx).item()
    # return {"result": lowest_xest,
    #         "nstep": nstep,
    #         "tnstep": tnstep,
    #         "lowest_step": lowest_step,
    #         "diff": torch.norm(lowest_gx).item(),
    #         "diff_detail": torch.norm(lowest_gx, dim=1),
    #         "prot_break": prot_break,
    #         "trace": trace,
    #         "new_trace": new_trace,
    #         "eps": eps,
    #         "threshold": threshold}



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
    def __init__(self, f, **kwargs):
        super().__init__()
        self.f = f
        self.kwargs = kwargs

    def broyd_output_test(self, z, x, y_shape, input_shape):
        reshaped_x = torch.reshape(input=x, shape=y_shape)
        reshaped_z = torch.reshape(input=z, shape=input_shape)
        output = self.f(reshaped_z, reshaped_x) - reshaped_z
        flattened = torch.reshape(output, (output.shape[0], -1)).unsqueeze(-1)
        return flattened

    # def broyd_grad(self, g, z, x, g_shape, z_shape):
    #     self.solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
    #                 grad, **self.kwargs)

    def internal_g(self, z, x):
        return self.f(z, x) - z


    def forward(self, x, truth = None, initial_point = None):
        if initial_point is None:
            init_point = torch.zeros_like(x)
        else:
            init_point = initial_point
        # compute forward pass and re-engage autograd tape
        # init_point = torch.reshape(init_point, (init_point.shape[0], -1, 1))
        initial_point_shape = initial_point.shape
        g = lambda z: self.broyd_output_test(z, x, x.shape, initial_point_shape)
        with torch.no_grad():
            output_x, self.forward_res = broyden(g, init_point, threshold=self.kwargs['max_iter'], eps=1e-8)
        # output_x = torch.reshape(output_x, initial_point_shape)
        z = self.f(output_x, x)

        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)
        # g0 = f0 - z0

        def backward_hook(grad):

            def internal_function(y):
                input_shape = y.shape
                y = y.reshape(grad.shape)
                broyden_function = grad + torch.autograd.grad(f0, z0, y, retain_graph=True)[0]
                g_version = broyden_function - y
                g_version = g_version.reshape(input_shape)
                return g_version

            result = broyden(internal_function, grad, threshold=10, eps=1e-7)

            return result[0]

        z.register_hook(backward_hook)
        return z

class DEQFixedPointSimple(nn.Module):
    def __init__(self, f, **kwargs):
        super().__init__()
        self.f = f
        self.kwargs = kwargs

    def broyd_output_test(self, z, x, y_shape, input_shape):
        reshaped_x = torch.reshape(input=x, shape=y_shape)
        reshaped_z = torch.reshape(input=z, shape=input_shape)
        output = self.f(reshaped_z, reshaped_x) - reshaped_z
        flattened = torch.reshape(output, (output.shape[0], -1)).unsqueeze(-1)
        return flattened

    def internal_g(self, z, x):
        return self.f(z, x) - z

    def forward(self, x, truth=None, initial_point=None):
        if initial_point is None:
            init_point = torch.zeros_like(x)
        else:
            init_point = initial_point
        # compute forward pass and re-engage autograd tape
        # init_point = torch.reshape(init_point, (init_point.shape[0], -1, 1))
        initial_point_shape = initial_point.shape
        g = lambda z: self.broyd_output_test(z, x, x.shape, initial_point_shape)
        with torch.no_grad():
            output_x, self.forward_res = broyden(g, init_point, threshold=self.kwargs['max_iter'], eps=1e-7)
        # output_x = torch.reshape(output_x, initial_point_shape)
        z = self.f(output_x, x)

        return z

                # def forward(self, x, initial_point = None):
    #     if initial_point is None:
    #         init_point = torch.zeros_like(x)
    #     else:
    #         init_point = initial_point
    #     # compute forward pass and re-engage autograd tape
    #     with torch.no_grad():
    #         z, self.forward_res = self.solver(lambda z: self.f(z, x), init_point, **self.kwargs)
    #     z = self.f(z, x)
    #
    #     # set up Jacobian vector product (without additional forward calls)
    #     z0 = z.clone().detach().requires_grad_()
    #     f0 = self.f(z0, x)
    #
    #     def backward_hook(grad):
    #         g, self.backward_res = self.solver(lambda y: torch.autograd.grad(f0, z0, y, retain_graph=True)[0] + grad,
    #                                            grad, **self.kwargs)
    #         return g
    #
    #     z.register_hook(backward_hook)
    #     return z

class DEQFixedPoint2(nn.Module):
    def __init__(self, f, **kwargs):
        super().__init__()
        self.f = f
        self.kwargs = kwargs

    def broyd_output_test(self, z, x, y_shape, input_shape):
        reshaped_x = torch.reshape(input=x, shape=y_shape)
        reshaped_z = torch.reshape(input=z, shape=input_shape)
        output = self.f(reshaped_z, reshaped_x) - reshaped_z
        flattened = torch.reshape(output, (output.shape[0], -1)).unsqueeze(-1)
        return flattened

    def internal_g(self, z, x):
        return self.f(z, x) - z


    def forward(self, x, truth = None, initial_point = None):
        if initial_point is None:
            init_point = torch.zeros_like(x)
        else:
            init_point = initial_point
        # compute forward pass and re-engage autograd tape
        # init_point = torch.reshape(init_point, (init_point.shape[0], -1, 1))
        initial_point_shape = initial_point.shape
        g = lambda z: self.broyd_output_test(z, x, x.shape, initial_point_shape)
        with torch.no_grad():
            output_x, self.forward_res = broyden(g, init_point, threshold=100, eps=1e-7)
        # output_x = torch.reshape(output_x, initial_point_shape)
        z = self.f(output_x, x)

        z0 = z.clone().detach().requires_grad_()
        f0 = self.f(z0, x)
        # g0 = f0 - z0

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

    def broyd_output_test(self, z, x, y_shape, input_shape):
        reshaped_x = torch.reshape(input=x, shape=y_shape)
        reshaped_z = torch.reshape(input=z, shape=input_shape)
        output = self.f(reshaped_z, reshaped_x) - reshaped_z
        flattened = torch.reshape(output, (output.shape[0], -1)).unsqueeze(-1)
        return flattened


    def forward(self, x, truth = None, initial_point = None):
        if initial_point is None:
            init_point = torch.zeros_like(x)
        else:
            init_point = initial_point
        # compute forward pass and re-engage autograd tape
        init_point = torch.reshape(init_point, (init_point.shape[0], -1, 1))
        initial_point_shape = initial_point.shape
        g = lambda z: self.broyd_output_test(z, x, x.shape, initial_point_shape)
        with torch.no_grad():
            output_x, self.forward_res = broyden(g, init_point, threshold=50, eps=1e-7)
        output_x = torch.reshape(output_x, initial_point_shape)

        return output_x

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
    running_iterate = z
    for iteration in range(max_iterations):
        running_iterate = solver(running_iterate)
    return running_iterate, running_iterate

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
