import torch.nn as nn
import torch
from solvers.cg_utils import conjugate_gradient

class EquilibriumGrad(nn.Module):
    def __init__(self, linear_operator, nonlinear_operator, eta_initial_val=0.1, minval = -1, maxval = 1):
        super(EquilibriumGrad,self).__init__()
        self.linear_op = linear_operator
        self.nonlinear_op = nonlinear_operator

        self.minval = minval
        self.maxval = maxval

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

    def set_initial_point(self, y):
        self.initial_point = self._linear_adjoint(y)

    def get_gradient(self, z, y):
        return self.linear_op.gramian(z)  - self._linear_adjoint(y) - self.nonlinear_op(z)

    def forward(self, z, y):
        z_tplus1 = z - self.eta * self.get_gradient(z, y)
        z_tplus1 = torch.clamp(z_tplus1, self.minval, self.maxval)
        return z_tplus1

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
