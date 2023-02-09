import torch.nn as nn
import torch
from solvers.cg_utils import conjugate_gradient
from PIL import Image
import imageio
import numpy as np

tt=0
class ProxgradNet(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta_initial_val=0.1):
        super(ProxgradNet,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

        self.register_parameter(name='eta', param=torch.nn.Parameter(torch.tensor(eta_initial_val), requires_grad=True))

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    # This is a bit redundant
    def initial_point(self, y):
        return self._linear_adjoint(y)

    def initial_point_precond(self, y):
        initial_point = self._linear_adjoint(y)
        preconditioned_input = conjugate_gradient(initial_point, self.linear_op.gramian, regularization_lambda=self.eta,
                                                  n_iterations=60)
        return preconditioned_input

    def single_block(self, input, y):
        grad_update = input - self.eta * (self.linear_op.gramian(input)  - self._linear_adjoint(y))
        return self.nonlinear_op(grad_update) + grad_update

    def forward(self, y, iterations):
        initial_point = self.initial_point_precond(y)
        running_term = initial_point
        # global tt
        # bsz = initial_point.shape[0]
        # past_iterate = initial_point

        for bb in range(iterations):
            running_term = self.single_block(running_term, y)

        #     #img_array = (np.clip(np.transpose(running_term.cpu().detach().numpy(), (0, 2, 3, 1)), -1,
        #     #                     1) + 1.0) * 127.5
        #     img_array = torch.norm(running_term, dim=1).cpu().detach().numpy() * 255.0 / np.sqrt(2)
        #     img_array = img_array.astype(np.uint8)
        #
        #     residual = torch.norm(running_term - past_iterate, dim=1).cpu().detach().numpy()
        #     if bb % 10 == 2:
        #         for k in range(bsz):
        #             #filename = "/share/data/vision-greg2/users/gilton/test_imgs/deblur/img/" + str(tt + k) + "_" + str(bb) + ".png"
        #             filename = "/share/data/vision-greg2/users/gilton/test_imgs/mrie2e/img/" + str(tt + k) + "_" + str(bb) + ".png"
        #             #filename = "/share/data/vision-greg2/users/gilton/test_imgs/cs/img/" + str(tt + k) + "_" + str(bb) + ".png"
        #             output_img = Image.fromarray(img_array[k, ...])
        #             output_img = output_img.resize((512, 512), resample=Image.NEAREST)
        #             imageio.imwrite(filename, output_img, format='PNG-PIL')
        #
        #             #filename = "/share/data/vision-greg2/users/gilton/test_imgs/deblur/res/" + str(tt + k) + "_" + str(bb) + ".png"
        #             filename = "/share/data/vision-greg2/users/gilton/test_imgs/mrie2e/res/" + str(tt + k) + "_" + str(bb) + ".png"
        #             #filename = "/share/data/vision-greg2/users/gilton/test_imgs/cs/res/" + str(tt + k) + "_" + str(bb) + ".png"
        #
        #             normalized_res = np.clip(residual[k, :, :] * 8, 0, 1) * 255.0
        #             # print(np.shape(normalized_res))
        #             # exit()
        #             normalized_res = normalized_res.astype(np.uint8)
        #             output_img = Image.fromarray(normalized_res)
        #             output_img = output_img.resize((512, 512), resample=Image.NEAREST)
        #             imageio.imwrite(filename, output_img, format='PNG-PIL')
        #
        # tt += bsz
        return running_term

# tt=0
class ProxgradNetMulti(nn.Module):
    def __init__(self, linear_operator, nonlinear_operators, eta_initial_val=0.1):
        super(ProxgradNetMulti,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_ops = torch.nn.ModuleList(nonlinear_operators)

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

        self.register_parameter(name='eta', param=torch.nn.Parameter(torch.tensor(eta_initial_val), requires_grad=True))

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    # This is a bit redundant
    def initial_point(self, y):
        return self._linear_adjoint(y)

    def initial_point_precond(self, y):
        initial_point = self._linear_adjoint(y)
        preconditioned_input = conjugate_gradient(initial_point, self.linear_op.gramian, regularization_lambda=self.eta,
                                                  n_iterations=60)
        return preconditioned_input

    def single_block(self, input, y, iterate):
        grad_update = input - self.eta * (self.linear_op.gramian(input)  - self._linear_adjoint(y))
        return self.nonlinear_ops[iterate](grad_update) + grad_update

    def forward(self, y, iterations):
        initial_point = self.eta * self.initial_point_precond(y)
        running_term = initial_point
        # bsz = initial_point.shape[0]
        # past_iterate = initial_point

        for bb in range(iterations):
            running_term = self.single_block(running_term, y,  bb)

        #     #img_array = (np.clip(np.transpose(running_term.cpu().detach().numpy(), (0, 2, 3, 1)), -1,
        #     #                     1) + 1.0) * 127.5
        #     img_array = torch.norm(running_term, dim=1).cpu().detach().numpy() * 255.0 / np.sqrt(2)
        #     img_array = img_array.astype(np.uint8)
        #
        #     residual = torch.norm(running_term - past_iterate, dim=1).cpu().detach().numpy()
        #     if bb % 10 == 2:
        #         for k in range(bsz):
        #             #filename = "/share/data/vision-greg2/users/gilton/test_imgs/deblur/img/" + str(tt + k) + "_" + str(bb) + ".png"
        #             filename = "/share/data/vision-greg2/users/gilton/test_imgs/mrie2e/img/" + str(tt + k) + "_" + str(bb) + ".png"
        #             #filename = "/share/data/vision-greg2/users/gilton/test_imgs/cs/img/" + str(tt + k) + "_" + str(bb) + ".png"
        #             output_img = Image.fromarray(img_array[k, ...])
        #             output_img = output_img.resize((512, 512), resample=Image.NEAREST)
        #             imageio.imwrite(filename, output_img, format='PNG-PIL')
        #
        #             #filename = "/share/data/vision-greg2/users/gilton/test_imgs/deblur/res/" + str(tt + k) + "_" + str(bb) + ".png"
        #             filename = "/share/data/vision-greg2/users/gilton/test_imgs/mrie2e/res/" + str(tt + k) + "_" + str(bb) + ".png"
        #             #filename = "/share/data/vision-greg2/users/gilton/test_imgs/cs/res/" + str(tt + k) + "_" + str(bb) + ".png"
        #
        #             normalized_res = np.clip(residual[k, :, :] * 8, 0, 1) * 255.0
        #             # print(np.shape(normalized_res))
        #             # exit()
        #             normalized_res = normalized_res.astype(np.uint8)
        #             output_img = Image.fromarray(normalized_res)
        #             output_img = output_img.resize((512, 512), resample=Image.NEAREST)
        #             imageio.imwrite(filename, output_img, format='PNG-PIL')
        #
        # tt += bsz
        return running_term

class PrecondNeumannNet(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, lambda_initial_val=0.1, cg_iterations=10):
        super(PrecondNeumannNet,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator
        self.cg_iterations = cg_iterations

        # Check if the linear operator has parameters that can be learned:
        # if so, register them to be learned as part of the network.
        linear_param_name = 'linear_param_'
        for ii, parameter in enumerate(self.linear_op.parameters()):
            parameter_name = linear_param_name + str(ii)
            self.register_parameter(name=parameter_name, param=parameter)

        self.register_parameter(name='eta', param=torch.nn.Parameter(torch.tensor(lambda_initial_val), requires_grad=True))

    def _linear_op(self, x):
        return self.linear_op.forward(x)

    def _linear_adjoint(self, x):
        return self.linear_op.adjoint(x)

    # This is a bit redundant
    def initial_point(self, y):
        preconditioned_input = conjugate_gradient(y, self.linear_op.gramian, regularization_lambda=self.eta,
                                                  n_iterations=self.cg_iterations)
        return preconditioned_input

    def single_block(self, input):
        preconditioned_step = conjugate_gradient(input, self.linear_op.gramian, regularization_lambda=self.eta,
                                                  n_iterations=self.cg_iterations)
        return self.eta * preconditioned_step - self.nonlinear_op(input)

    def forward(self, y, iterations):
        initial_point = self.eta * self.initial_point(y)
        running_term = initial_point
        accumulator = initial_point

        for bb in range(iterations):
            running_term = self.single_block(running_term)
            accumulator = accumulator + running_term

        return accumulator
