import torch
import numpy as np
import torch.nn as nn
import pdb

from pyro.distributions.transforms import BlockAutoregressive, NeuralAutoregressive
torchType = torch.float32

from pyro.nn import AutoRegressiveNN
from pyro.distributions.transforms import NeuralAutoregressive


class Transition_new(nn.Module):
    """
    Base class for custom transitions of new type
    """
    def __init__(self, kwargs, device):
        super(Transition_new, self).__init__()
        self.use_barker = kwargs.use_barker  # If false, we are using standard MH ration, otherwise Barker ratio
        self.device = device
        self.device_zero = torch.tensor(0., dtype=torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=torchType, device=self.device)
        self.direction_distr = torch.distributions.Uniform(low=self.device_zero,
                                                            high=self.device_one)  # distribution for transition making
        self.logit = nn.Parameter(torch.tensor(np.log(kwargs['p']) - np.log(1 - kwargs['p']),
                                               dtype=torchType, device=self.device))  # probability of forward transition


    def _forward_step(self, z_old, x=None, k=None, target=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        # You should define the class for your custom transition
        raise NotImplementedError

    def _backward_step(self, z_old, x=None, k=None, target=None):
        """
        The function makes backward step
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        Output:
        z_new - new position T^(-1)(z_old)
        log_jac - log_jacobian of the transformation
        """
        # You should define the class for your custom transition
        raise NotImplementedError

    def make_transition(self, z_old, target_distr, k=None, x=None, detach=False, p_old=None):
        """
        The function returns directions (-1, 0 or +1), sampled in the current positions
        Input:
        z_old - point of evaluation
        target_distr - target distribution
        x - data object (optional)
        k - number of transition (optional)
        detach - whether to detach target (decoder) from computational graph or not
        p_old - auxilary variables for some types of transitions
        Output:
        z_new - new points
        log_jac - log jacobians of transformations
        current_log_alphas - current log_alphas, corresponding to sampled decision variables
        current_log_probs - current log probabilities of forward transition
        a - decision variables (-1, 0 or +1)
        """
        ############ Sample v_1 -- either +1 or -1 ############
        p = torch.sigmoid(self.logit)
        probs = self.direction_distr.sample((z_old.shape[0], ))
        v_1 = torch.where(probs < p, self.device_one, -self.device_one)

        ############ Then we compute new points and densities ############
        # pdb.set_trace()
        z_f, log_jac_f = self._forward_step(z_old=z_old, k=k)
        z_b, log_jac_b = self._backward_step(z_old=z_old, k=k)

        target_log_density_f = target_distr.get_logdensity(z=z_f, x=x)
        target_log_density_b = target_distr.get_logdensity(z=z_b, x=x)
        target_log_density_old = target_distr.get_logdensity(z=z_old, x=x)
        ############ Then we select only those which correspond to selected direction ############
        target_log_density_new = torch.where(v_1 == 1., target_log_density_f, target_log_density_b)
        current_probs = torch.where(v_1 == 1., p, 1 - p)
        new_log_jacobian = torch.where(v_1 == 1., log_jac_f, log_jac_b)
        ############### Compute acceptance ratio ##############
        log_t = target_log_density_new + torch.log(1. - current_probs) + new_log_jacobian\
                - target_log_density_old - torch.log(current_probs)
        ############### Two expressions for performing transition  ##############
        log_1_t = torch.logsumexp(torch.cat([torch.zeros_like(log_t).view(-1, 1),
                                                                        log_t.view(-1, 1)], dim=-1), dim=-1)
        if self.use_barker:
            current_log_alphas_pre = log_t - log_1_t
        else:
            current_log_alphas_pre = torch.min(self.device_zero, log_t)
        log_probs = torch.log(self.direction_distr.sample((z_old.shape[0],)))
        a = torch.where(log_probs < current_log_alphas_pre, v_1, self.device_zero)

        if self.use_barker:
            current_log_alphas = torch.where((a == 0), -log_1_t, current_log_alphas_pre)
        else:
            expression = 1. - torch.exp(log_t)
            expression = torch.where(expression <= self.device_zero, self.device_one * 1e-8, expression)
            corr_expression = torch.log(expression)
            current_log_alphas = torch.where((a == 0), corr_expression, current_log_alphas_pre)

        current_log_probs = torch.log(current_probs)

        z_new = torch.where((a == -self.device_one)[:, None], z_b,
                            torch.where((a == self.device_zero)[:, None], z_old, z_f))

        same_log_jac = torch.zeros_like(new_log_jacobian)
        log_jac = torch.where((a == self.device_zero), same_log_jac, new_log_jacobian)

        return z_new, log_jac, current_log_alphas, current_log_probs, a



class RealNVP_new(Transition_new):
    def __init__(self, kwargs, device):
        super(RealNVP_new, self).__init__(kwargs, device)
        if "n_layers" in kwargs.keys():
            self.n_layers = kwargs["n_layers"]
        else:
            self.n_layers = 1

        if "n_flows" in kwargs.keys(): # number of RNVP steps
            self.n_flows = kwargs["n_flows"]
        elif "masks" in kwargs.keys():
            self.n_flows = len(kwargs["masks"])
        else:
            self.n_flows = 1


        self.hidden_dim = kwargs["hidden_dim"]
        self.z_dim = kwargs["z_dim"]

        if "masks" in kwargs.keys():
            self.masks = torch.tensor(kwargs['masks'], device=self.device).to(device)
        else:
            self.masks = []
            for i in range(self.n_flows):
                if i % 2:
                    self.masks.append(torch.cat((torch.ones((self.z_dim//2), device=device),
                                                 torch.zeros((self.z_dim - self.z_dim//2), device=device))))
                else:
                    self.masks.append(
                        torch.cat((torch.zeros((self.z_dim // 2), device=device),
                                   torch.ones((self.z_dim - self.z_dim // 2), device=device))))


        if len(self.masks) != self.n_flows:
            message = "mask list length : {}, n_flows : {}".format(len(self.masks), self.n_flows)
            print("WARNING : {}".format(message))


        nett, nets = kwargs["nett"], kwargs["nets"] # custom nets for t and s. Overwritten if not

        if nets == "linear": # if nets is not custom, we build a linear network with the right number of layers (n_layers)
            nets = lambda: nn.Sequential(
                *[nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU()] + [nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()] * self.n_layers + [nn.Linear(self.hidden_dim, self.z_dim), nn.Tanh()] )
        if nett == "linear":
            if kwargs['step_conditioning']:
                if kwargs['noise_aggregation'] in ['addition',  None]:
                    nett = lambda : nn.Sequential(*[nn.Linear(self.z_dim, self.hidden_dim), nn.LeakyReLU()] + [nn.Linear(self.hidden_dim, self.hidden_dim), nn.LeakyReLU()] * self.n_layers + [nn.Linear(self.hidden_dim, self.z_dim)])
                elif kwargs['noise_aggregation'] == 'stacking':
                    nett = lambda : nn.Sequential(*[nn.Linear(2 * self.z_dim, 2 * self.hidden_dim), nn.LeakyReLU()] + [nn.Linear(2 * self.hidden_dim, 2 * self.hidden_dim), nn.LeakyReLU()] * (self.n_layers - 1) + [nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.Linear(self.hidden_dim, self.z_dim)]) # n_layers - 1


        if nets == "convolutional":
            raise NotImplementedError

        if nett == "convolutional":
            raise NotImplementedError


        self.t = nn.ModuleList([nett() for _ in range(self.n_flows)]).to(device)
        self.s = nn.ModuleList([nets() for _ in range(self.n_flows)]).to(device)

    def _forward_step(self, z_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional) k[0] - value, k[1] - method
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        log_det_J = z_old.new_zeros(z_old.shape[0])
        if k is not None:
            noise = torch.ones_like(z_old) * k[0]
        for i in range(self.n_flows):
            z_old_ = z_old * self.masks[i]
            if k is not None:
                if k[1] == 'stacking':
                    stacked_vector = torch.cat([z_old_, noise], dim=1)
                    s = self.s[i](stacked_vector) * (self.device_one - self.masks[i])
                    t = self.t[i](stacked_vector) * (self.device_one - self.masks[i])
                elif k[1] == 'addition':
                    added_vector = z_old_ + noise
                    s = self.s[i](added_vector) * (self.device_one - self.masks[i])
                    t = self.t[i](added_vector) * (self.device_one - self.masks[i])
            else:
                s = self.s[i](z_old_) * (self.device_one - self.masks[i])
                t = self.t[i](z_old_) * (self.device_one - self.masks[i])
            z_old = z_old_ + (self.device_one - self.masks[i]) * (z_old * torch.exp(s) + t)
            if len(s.squeeze()) == 2:
                log_det_J += s.squeeze().sum(dim=1)
            else:
                log_det_J += s.sum(dim=1)
        z_new = z_old
        return z_new, log_det_J

    def _backward_step(self, z_old, x=None, k=None, target=None, p_old=None):
        """
        The function makes backward step
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        target - target class (optional)
        p_old - auxilary variables for some types of transitions (like momentum for HMC)
        Output:
        z_new - new position T^(-1)(z_old)
        log_jac - log_jacobian of the transformation
        """
        log_det_J, z = z_old.new_zeros(z_old.shape[0]), z_old
        if k is not None:
            noise = torch.ones_like(z_old) * k[0]
        for i in reversed(range(self.n_flows)):
            z_ = self.masks[i] * z
            if k is not None:
                if k[1] == 'stacking':
                    stacked_vector = torch.cat([z_, noise], dim=1)
                    t = self.t[i](stacked_vector) * (self.device_one - self.masks[i])
                    s = self.s[i](stacked_vector) * (self.device_one - self.masks[i])
                elif k[1] == 'addition':
                    added_vector = z_ + noise
                    t = self.t[i](added_vector) * (self.device_one - self.masks[i])
                    s = self.s[i](added_vector) * (self.device_one - self.masks[i])
            else:
                t = self.t[i](z_) * (self.device_one - self.masks[i])
                s = self.s[i](z_) * (self.device_one - self.masks[i])
            z = (self.device_one - self.masks[i]) * (z - t) * torch.exp(-s) + z_
            log_det_J -= s.sum(dim=1)
        z_new = z
        return z_new, log_det_J

    def log_prob(self, x):
        z, logp = self.f(x)
        return self.prior.log_prob(z) + logp

    def sample(self, batchSize):
        z = self.prior.sample((batchSize, 1))
        logp = self.prior.log_prob(z)
        x, log_det_J = self.g(z)
        return x

class BNAF(Transition_new):
    def __init__(self, kwargs, device):
        super(BNAF, self).__init__(kwargs, device)
        self.z_dim = kwargs["z_dim"]
        self.flow = BlockAutoregressive(self.z_dim)


    def _forward_step(self, z_old, x=None, k=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional) k[0] - value, k[1] - method
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        z_new = self.flow(z_old)
        log_det_J = self.flow.log_abs_det_jacobian(z_old, z_new)
        return z_new, log_det_J

    def _backward_step(self, z_old, x=None, k=None):
        """
        The function makes backward step
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        Output:
        z_new - new position T^(-1)(z_old)
        log_jac - log_jacobian of the transformation
        """
        raise NotImplementedError
    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, batchSize):
        raise NotImplementedError

class NAF(Transition_new):
    def __init__(self, kwargs, device):
        super(NAF, self).__init__(kwargs, device)
        self.z_dim = kwargs["z_dim"]
        self.flow = NeuralAutoregressive(
                    AutoRegressiveNN(self.z_dim, [2 * self.z_dim], param_dims=[self.z_dim] * 3),
                    hidden_units=self.z_dim)


    def _forward_step(self, z_old, x=None, k=None):
        """
        The function makes forward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional) k[0] - value, k[1] - method
        Output:
        z_new - new position = T(z_old)
        log_jac - log_jacobian of the transformation
        """
        z_new = self.flow(z_old)
        log_det_J = self.flow.log_abs_det_jacobian(z_old, z_new)
        return z_new, log_det_J

    def _backward_step(self, z_old, x=None, k=None):
        """
        The function makes backward step
        Also, this function computes log_jacobian of the transformation
        Input:
        z_old - current position
        x - data object (optional)
        k - number of transition (optional)
        Output:
        z_new - new position T^(-1)(z_old)
        log_jac - log_jacobian of the transformation
        """
        raise NotImplementedError
    def log_prob(self, x):
        raise NotImplementedError

    def sample(self, batchSize):
        raise NotImplementedError
