import torch.nn as nn
import torch
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from solvers.cg_utils import conjugate_gradient

class EquilibriumGrad(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta, minval = -1, maxval = 1):
        super(EquilibriumGrad,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator
        # self.eta = eta

        self.minval = minval
        self.maxval = maxval

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

        self.register_parameter(name='eta', param=torch.nn.Parameter(torch.tensor(eta), requires_grad=True))


    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def set_initial_point(self, y):
        self.initial_point = self._linear_adjoint(y)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z)  - self._linear_adjoint(y) - self.nonlinear_op(z)

    def forward(self, z, y):
        z_tplus1 = z - self.eta * self.get_gradient(z, y)
        z_tplus1 = torch.clamp(z_tplus1, self.minval, self.maxval)
        return z_tplus1

class EquilibriumProxGrad(nn.Module):
    def  __init__(self, linear_operator, nonlinear_operator, eta, minval = -1, maxval = 1):
        super(EquilibriumProxGrad,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator

        self.minval = minval
        self.maxval = maxval
        self.register_parameter(name='eta', param=torch.nn.Parameter(torch.tensor(eta), requires_grad=True))
        # self.eta = torch.tensor(0.2)

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z)  - self._linear_adjoint(y)

    def forward(self, z, y):
        # self.eta = torch.tensor(0.2)
        # test = self.nonlinear_op(z)#test
        gradstep = z - self.eta * self.get_gradient(z, y)
        # gradstep = z - torch.tensor(0.2) * self.get_gradient(z, y)
        z_tplus1 = gradstep + self.nonlinear_op(gradstep) # yaping mute
        # z_tplus1 = self.nonlinear_op(gradstep) # yaping add
        z_tplus1 = torch.clamp(z_tplus1, self.minval, self.maxval)
        return z_tplus1


class EquilibriumProxGradMRI(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta, minval = -1, maxval = 1):
        super(EquilibriumProxGradMRI,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator

        self.minval = minval
        self.maxval = maxval
        self.eta = eta

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z)  - self._linear_adjoint(y)

    def forward(self, z, y):
        gradstep = z - self.eta * self.get_gradient(z, y)
        z_tplus1 = gradstep + self.nonlinear_op(gradstep)
        z_tplus1 = torch.clamp(z_tplus1, self.minval, self.maxval)
        return z_tplus1

class ProxPnP(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta, minval = -1, maxval = 1):
        super(ProxPnP,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator

        self.minval = minval
        self.maxval = maxval
        self.eta = eta

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def get_gradient(self, z, y):
        return self.linear_op.adjoint(self.linear_op.forward(z)  - y)

    def forward(self, z, y):
        gradstep = z - self.eta*(self.linear_op.adjoint(self.linear_op.forward(z)) - self.linear_op.adjoint(y))
        z_tplus1 = gradstep + self.nonlinear_op(gradstep)
        #z_tplus1 = torch.clamp(z_tplus1, self.minval, self.maxval)
        return z_tplus1

class DouglasRachford(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta, max_iters = 10, minval = -1, maxval = 1):
        super(DouglasRachford,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator

        self.minval = minval
        self.maxval = maxval
        self.lambdaval = eta
        self.max_cg_iterations = max_iters

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def internal_prox(self, x, y):
        initial_point = self.linear_op.adjoint(y) + self.lambdaval*x
        return conjugate_gradient(initial_point, self.linear_op.gramian, self.lambdaval,
                                  n_iterations=self.max_cg_iterations)

    def get_gradient(self, z, y):
        return self.linear_op.adjoint(self.linear_op.forward(z)  - y)

    def forward(self, z, y):
        prox_f = self.internal_prox(z, y)
        net_input = 2*prox_f - z
        z_tplus1 = (z + 2*(self.nonlinear_op(net_input) + net_input)-net_input) / 2.0
        z_tplus1 = torch.clamp(z_tplus1, self.minval, self.maxval)
        return z_tplus1

class EquilibriumADMM(nn.Module):
    def __init__(self, linear_operator, denoising_net, max_cg_iterations=20, x_alpha=0.4, eta = 0.1, minval=-1, maxval=1):
        super(EquilibriumADMM, self).__init__()
        self.linear_op = linear_operator
        self.denoising_net = denoising_net

        self.minval = minval
        self.maxval = maxval
        self.x_alpha = x_alpha
        self.eta = eta

        self.max_cg_iters = max_cg_iterations

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def _x_update(self, z, u, y):
        gramian = self.linear_op.gramian
        # initial_point = self._linear_adjoint(y) + 0.0000001 * (z - u)
        initial_point = self._linear_adjoint(y) + self.x_alpha*(z-u)

        x_update = conjugate_gradient(initial_point, gramian, self.x_alpha, n_iterations=self.max_cg_iters)
        return x_update, z, u

    def _z_update(self, x, z, u):
        net_input = x + u
        z_update = net_input + self.denoising_net(net_input)
        return x, z_update, u

    def _u_update(self, x, z, u):
        u_update = u + self.eta * (x - z)
        # u_update = u + z - x

        return x, z, u_update

    def forward(self, z, u, y):
        x_new, z, u = self._x_update(z, u, y)
        x_new, z_new, u = self._z_update(x_new, z, u)
        x_new, z_new, u_new = self._u_update(x_new, z_new, u)
        z_new = torch.clamp(z_new, self.minval, self.maxval)
        return z_new, u_new

class EquilibriumADMM2(nn.Module):
    def __init__(self, linear_operator, denoising_net, max_cg_iterations=20, x_alpha=0.4, eta = 0.1, minval=-1, maxval=1):
        super(EquilibriumADMM2, self).__init__()
        self.linear_op = linear_operator
        self.denoising_net = denoising_net

        self.minval = minval
        self.maxval = maxval
        self.x_alpha = x_alpha
        self.eta = eta

        self.max_cg_iters = max_cg_iterations

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def _x_update(self, z, u, y):
        gramian = self.linear_op.gramian
        # initial_point = self._linear_adjoint(y) + 0.0000001 * (z - u)
        initial_point = self._linear_adjoint(y) + self.x_alpha*(z-u)

        x_update = conjugate_gradient(initial_point, gramian, self.x_alpha, n_iterations=self.max_cg_iters)
        return x_update, z, u

    def _z_update(self, x, z, u):
        net_input = x + u
        z_update = net_input - self.denoising_net(net_input)
        return x, z_update, u

    def _u_update(self, x, z, u):
        u_update = u + self.eta * (x - z)
        # u_update = u + z - x

        return x, z, u_update

    def forward(self, z, u, y):
        x_new, z, u = self._x_update(z, u, y)
        x_new, z_new, u = self._z_update(x_new, z, u)
        x_new, z_new, u_new = self._u_update(x_new, z_new, u)
        z_new = torch.clamp(z_new, self.minval, self.maxval)
        return z_new, u_new

class EquilibriumADMM_minus(nn.Module):
    def __init__(self, linear_operator, denoising_net, max_cg_iterations=20, x_alpha=0.4, eta = 0.1, minval=-1, maxval=1):
        super(EquilibriumADMM_minus, self).__init__()
        self.linear_op = linear_operator
        self.denoising_net = denoising_net

        self.minval = minval
        self.maxval = maxval
        self.x_alpha = x_alpha
        self.eta = eta

        self.max_cg_iters = max_cg_iterations

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def _x_update(self, z, u, y):
        net_input = z - u
        x_update = net_input - self.denoising_net(net_input)
        return x_update, z, u

    def _z_update(self, x, u, y):
        gramian = self.linear_op.gramian
        # initial_point = self._linear_adjoint(y) + 0.0000001 * (z - u)
        initial_point = self._linear_adjoint(y) + self.x_alpha*(x+u)

        z_update = conjugate_gradient(initial_point, gramian, self.x_alpha, n_iterations=self.max_cg_iters)
        return x, z_update, u

    def _u_update(self, x, z, u):
        u_update = u + self.eta * (x - z)
        # u_update = u + z - x

        return x, z, u_update

    def forward(self, z, u, y):
        x_new, z, u = self._x_update(z, u, y)
        x_new, z_new, u = self._z_update(x_new, u, y)
        x_new, z_new, u_new = self._u_update(x_new, z_new, u)
        z_new = torch.clamp(z_new, self.minval, self.maxval)
        return z_new, u_new

class EquilibriumADMM_plus(nn.Module):
    def __init__(self, linear_operator, denoising_net, max_cg_iterations=20, x_alpha=0.4, eta = 0.1, minval=-1, maxval=1):
        super(EquilibriumADMM_plus, self).__init__()
        self.linear_op = linear_operator
        self.denoising_net = denoising_net

        self.minval = minval
        self.maxval = maxval
        self.x_alpha = x_alpha
        self.eta = eta

        self.max_cg_iters = max_cg_iterations

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    def _x_update(self, z, u, y):
        net_input = z - u
        x_update = net_input + self.denoising_net(net_input)
        return x_update, z, u

    def _z_update(self, x, u, y):
        gramian = self.linear_op.gramian
        # initial_point = self._linear_adjoint(y) + 0.0000001 * (z - u)
        initial_point = self._linear_adjoint(y) + self.x_alpha*(x+u)

        z_update = conjugate_gradient(initial_point, gramian, self.x_alpha, n_iterations=self.max_cg_iters)
        return x, z_update, u

    def _u_update(self, x, z, u):
        u_update = u + self.eta * (x - z)
        # u_update = u + z - x

        return x, z, u_update

    def forward(self, z, u, y):
        x_new, z, u = self._x_update(z, u, y)
        x_new, z_new, u = self._z_update(x_new, u, y)
        x_new, z_new, u_new = self._u_update(x_new, z_new, u)
        z_new = torch.clamp(z_new, self.minval, self.maxval)
        return z_new, u_new

# def tensor_to_np(tensor):
#     img = tensor.clip(0, 1).cpu().detach().unsqueeze(2).numpy() * 255.
#     return img

class EquilibriumProxGradSCI(nn.Module):
    def  __init__(self, A, At, nonlinear_operator, eta, minval = -1, maxval = 1):
        super(EquilibriumProxGradSCI,self).__init__()
        # self.linear_op = linear_operator
        # self.y1 = torch.zeros(y_shape).cuda()
        self.A = A
        self.At = At
        self.nonlinear_op = nonlinear_operator

        self.minval = minval
        self.maxval = maxval
        self.y = 0
        self.noise_sigma=torch.FloatTensor([60/255]).cuda().expand(8)

    def forward(self, z, y, Phi, Phi_sum):
        bsz, w, h, c = z.shape

        fb = self.A(z, Phi)
        z = z + self.At((y - fb) / Phi_sum, Phi)
        
        if self.nonlinear_op.tag == 'conv2d': # input is a batch of independent images
            z_tplus1 = self.nonlinear_op(z.permute(0, 3, 1, 2).contiguous().view(bsz * c, 1, w, h))
            z_tplus1 = z_tplus1.view(bsz, c, w, h).permute(0, 2, 3, 1)
        elif self.nonlinear_op.tag == 'conv3d': # input is a batch of video
            z_tplus1 = self.nonlinear_op(z.permute(0, 3, 1, 2).unsqueeze(1).contiguous())
            z_tplus1 = z_tplus1.squeeze(1).permute(0, 2, 3, 1)
        elif self.nonlinear_op.tag == 'ffdnet':
            if self.y != y.mean():
                self.noise_sigma = torch.FloatTensor([60 / 255]).cuda().expand(bsz*c)
                self.y = y.mean()
            else:
                self.noise_sigma = self.noise_sigma * 0.971
            # print(self.noise_sigma)
            noise = self.nonlinear_op(z.permute(0, 3, 1, 2).contiguous().view(bsz * c, 1, w, h), self.noise_sigma)
            # noise = (noise-noise.min())/(noise.max()-noise.min())
            z_tplus1 = z - noise.view(bsz, c, w, h).permute(0, 2, 3, 1)
        elif self.nonlinear_op.tag == 'denoiser':
            noise = self.nonlinear_op(z.permute(0, 3, 1, 2).contiguous().view(bsz * c, 1, w, h))
            z_tplus1 = z - noise.view(bsz, c, w, h).permute(0, 2, 3, 1)
        elif self.nonlinear_op.tag == '3d_denoiser':
            noise = self.nonlinear_op(z.permute(0, 3, 1, 2).unsqueeze(1).contiguous())
            z_tplus1 = z - noise.squeeze(1).permute(0, 2, 3, 1)
        else:
            print('unknown nonlinear_op tag!')
        # import cv2
        # img_path="/home/yaping/projects/deep_equilibrium_inverse/save/test/"
        # for frame_id in range(8):
        #     cv2.imwrite(
        #         img_path + 'z_%d.png' % (frame_id),
        #         tensor_to_np(z[0, :, :, frame_id]))
        #     cv2.imwrite(
        #         img_path + 'reconstruction_%d.png' % (frame_id),
        #         tensor_to_np(z_tplus1[0, :, :, frame_id]))
        # exit()
        return z_tplus1

class EquilibriumADMMSCI(nn.Module):
    def __init__(self, A, At, nonlinear_operator, eta, minval=-1, maxval=1):
        super(EquilibriumADMMSCI, self).__init__()
        self.A = A
        self.At = At
        self.nonlinear_op = nonlinear_operator

        self.minval = minval
        self.maxval = maxval
        # self.eta = eta


    def forward(self, z, u, y, Phi, Phi_sum):
        bsz, w, h, c = z.shape

        fb = self.A(z+u, Phi)
        z = (z+u) + self.At((y - fb) / (Phi_sum + 1e-8), Phi)

        if not self.nonlinear_op.conv3d:  # input is a batch of independent images
            z_tplus1 = self.nonlinear_op((z-u).permute(0, 3, 1, 2).contiguous().view(bsz * c, 1, w, h))
            z_tplus1 = z_tplus1.view(bsz, c, w, h).permute(0, 2, 3, 1)
        else:  # input is a batch of video
            z_tplus1 = self.nonlinear_op((z-u).permute(0, 3, 1, 2).unsqueeze(1).contiguous())
            z_tplus1 = z_tplus1.squeeze(1).permute(0, 2, 3, 1)

        u = u - (z-z_tplus1)

        return z, u

############################################
############################################
