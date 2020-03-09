import math
import pdb
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import gamma, invgamma

torchType = torch.float32


class Target(nn.Module):
    """
    Base class for a custom target distribution
    """

    def __init__(self, kwargs, device):
        super(Target, self).__init__()
        self.device = device
        self.device_zero = torch.tensor(0., dtype=torchType, device=self.device)
        self.device_one = torch.tensor(1., dtype=torchType, device=self.device)

    def get_density(self, x, z):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def get_logdensity(self, x, z):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x)
        """
        # You should define the class for your custom distribution
        raise NotImplementedError

    def get_samples(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        # You should define the class for your custom distribution
        raise NotImplementedError


##################################################################################################################
########################################### Classes for custom targets ###########################################
##################################################################################################################

class GMM_target(Target):
    """
    Mixture of TWO gaussians (multivariate)
    """

    def __init__(self, kwargs, device):
        super(GMM_target, self).__init__(kwargs, device)

        self.p = kwargs['p_first_gaussian']  # probability of the first gaussian (1-p for the second)
        self.log_pis = [torch.tensor(np.log(self.p), dtype=torch.float32, device=device),
                        torch.tensor(np.log(1 - self.p), dtype=torch.float32,
                                     device=device)]  # LOGS! probabilities of Gaussians
        self.locs = kwargs['locs_single_gmm']  # list of locations for each of these gaussians
        self.covs = kwargs['covs_single_gmm']  # list of covariance matrices for each of these gaussians
        self.dists = [torch.distributions.MultivariateNormal(loc=self.locs[0], covariance_matrix=self.covs[0]),
                      torch.distributions.MultivariateNormal(loc=self.locs[1], covariance_matrix=self.covs[
                          1])]  # list of distributions for each of them

    def get_density(self, z, x=None):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.get_logdensity(z).exp()
        return density

    def get_logdensity(self, z, x=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x)
        """
        log_p_1 = (self.log_pis[0] + self.dists[0].log_prob(z)).view(-1, 1)
        log_p_2 = (self.log_pis[1] + self.dists[1].log_prob(z)).view(-1, 1)
        log_p_1_2 = torch.cat([log_p_1, log_p_2], dim=-1)
        log_density = torch.logsumexp(log_p_1_2, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density

    def get_samples(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        n_first = int(n * self.p)
        n_second = n - n_first
        samples_1 = self.dists[0].sample((n_first,))
        samples_2 = self.dists[1].sample((n_second,))
        samples = torch.cat([samples_1, samples_2])
        return samples


class GMM_target2(Target):
    """
    Mixture of n gaussians (multivariate)
    """

    def __init__(self, kwargs, device):
        super(GMM_target2, self).__init__(kwargs, device)
        self.device = device
        self.num = kwargs['num_gauss']
        self.pis = kwargs['p_gaussians']
        self.locs = kwargs['locs']  # list of locations for each of these gaussians
        self.covs = kwargs['covs']  # list of covariance matrices for each of these gaussians
        self.peak = [None] * self.num
        for i in range(self.num):
            self.peak[i] = torch.distributions.MultivariateNormal(loc=self.locs[i], covariance_matrix=self.covs[i])

    def get_density(self, z, x=None):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x)
        """
        density = self.get_logdensity(z).exp()
        return density

    def get_logdensity(self, z, x=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x)
        """
        log_p = torch.tensor([], device=self.device)
        for i in range(self.num):
            log_paux = (torch.log(self.pis[i]) + self.peak[i].log_prob(z)).view(-1, 1)
            log_p = torch.cat([log_p, log_paux], dim=-1)
        log_density = torch.logsumexp(log_p, dim=1)  # + torch.tensor(1337., device=self.device)
        return log_density

    def get_samples(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        p = np.random.choice(a=self.num, size=n)
        samples = torch.tensor([], device=self.device)

        for idx in p:
            z = self.peak[idx].sample((1,))
            samples = torch.cat([samples, z])
        return samples


class Funnel(Target):
    """
    Funnel distribution
    """

    def __init__(self, kwargs, device):
        super(Funnel, self).__init__(kwargs, device)
        self.d = kwargs['z_dim']
        self.std_normal = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)

    def get_density(self, z, x=None):
        """
        The method returns target density, estimated at point x
        Input:
        x - datapoint
        z - latent variable
        Output:
        density - p(x)
        """
        density = self.distr.log_prob(z).exp()
        return density

    def get_logdensity(self, z, x=None):
        """
        The method returns target logdensity, estimated at point x
        Input:
        x - datapoint
        z - latent variable
        Output:
        log_density - log p(x)
        """
        d = z.shape[1]
        fst_component = z[:, 0]
        log_density = -fst_component ** 2 / 2 - torch.sum(z[:, 1:] ** 2., dim=1) * torch.exp(
            -2. * fst_component) / 2. - (d - 1) * fst_component
        return log_density

    def get_samples(self, n):
        """
        The method returns samples from the distribution
        Input:
        n - amount of samples
        Output:
        samples - samples from the distribution
        """
        samples = torch.zeros((n, self.d), device=self.device)
        for i in range(n):
            samples[i][0] = self.std_normal.sample()
            component_normal = torch.distributions.Normal(loc=torch.zeros(self.d - 1, device=self.device),
                                                          scale=torch.exp(samples[i][0]) * torch.ones(self.d - 1,
                                                                                                      device=self.device))
            samples[i][1:] = component_normal.sample()
        return samples


class NN_bernoulli(Target):
    """
    Density for NN with Bernoulli output
    """

    def __init__(self, kwargs, model, device):
        super(NN_bernoulli, self).__init__(kwargs, device)
        self.decoder = model
        self.prior = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)

    def get_density(self, x, z):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x, z)
        """
        density = self.get_logdensity(x).exp()
        return density

    def get_logdensity(self, x, z):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x, z)
        """
        p_x_given_z_logits = self.decoder(z)
        p_x_given_z = torch.distributions.Bernoulli(logits=p_x_given_z_logits[0])
        expected_log_likelihood = torch.sum(p_x_given_z.log_prob(x), [1, 2, 3])
        log_density = expected_log_likelihood + self.prior.log_prob(z).sum(1)
        return log_density


class MF_target(Target):
    """
    Density for NN with Bernoulli output
    """

    def __init__(self, kwargs, model, device):
        super(MF_target, self).__init__(kwargs, device)
        self.decoder = model
        self.prior = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)

    def get_density(self, x, z):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x, z)
        """
        density = self.get_logdensity(x).exp()
        return density

    def get_logdensity(self, x, z):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x, z)
        """
        probs = self.decoder(z)
        p_x_given_z = torch.distributions.Bernoulli(probs=probs)
        expected_log_likelihood = torch.sum(p_x_given_z.log_prob(x.view(z.shape[0], -1)), 1)
        log_density = expected_log_likelihood + self.prior.log_prob(z).sum(1)
        return log_density


class NN_Gaussian(Target):
    """
    Density for NN with Gaussian output
    """

    def __init__(self, kwargs, model, device):
        super(NN_Gaussian, self).__init__(kwargs, device)
        self.decoder = model
        self.data_c = kwargs['data_c']
        self.prior = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)

    def get_density(self, x, z):
        """
        The method returns target density
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        density - p(x, z)
        """
        density = self.get_logdensity(x, z).exp()
        return density

    def get_logdensity(self, x, z):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent variable
        Output:
        log_density - log p(x, z)
        """
        mu, scale = self.decoder(z)
        p_x_given_z = torch.distributions.Normal(loc=mu, scale=scale)
        expected_log_likelihood = torch.sum(p_x_given_z.log_prob(x), [1, 2, 3])
        log_density = expected_log_likelihood + self.prior.log_prob(z).sum(1)
        return log_density


class BNAF_examples(Target):

    def __init__(self, kwargs, device):
        super(BNAF_examples, self).__init__(kwargs, device)
        self.data = kwargs.bnaf_data

    def get_logdensity(self, z, x=None):
        if self.data == 't1':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            z_norm = torch.norm(z, 2, 1)
            add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
            add2 = - torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + \
                               torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2) + 1e-9)
            return -add1 - add2

        elif self.data == 't2':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            return -0.5 * ((z[:, 1] - w1) / 0.4) ** 2
        elif self.data == 't3':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            w2 = 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
            in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.35) ** 2)
            in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w2) / 0.35) ** 2)
            return torch.log(in1 + in2 + 1e-9)
        elif self.data == 't4':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            w3 = 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)
            in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.4) ** 2)
            in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w3) / 0.35) ** 2)
            return torch.log(in1 + in2 + 1e-9)
        else:
            raise RuntimeError

    def get_density(self, z, x=None):
        density = self.distr.log_prob(z).exp()
        return density

    def get_samples(self, n):
        return torch.stack(hamiltorch.sample(log_prob_func=self.get_logdensity, params_init=torch.zeros(2),
                                             num_samples=n, step_size=.3, num_steps_per_sample=5))


class MNIST_target(Target):
    def __init__(self, kwargs, device):
        super(MNIST_target, self).__init__(kwargs, device)
        self.true_x = kwargs["true_x"]
        self.decoder = kwargs[
            "decoder"]
        self.prior = torch.distributions.Normal(loc=self.device_zero, scale=self.device_one)
        self.decoder.eval()

    def get_logdensity(self, z, x=None):
        """
        The method returns target logdensity
        Input:
        x - datapoint
        z - latent vaiable
        Output:
        log_density - log p(x, z)
        """
        if x is not None:
            if not torch.all(torch.eq(x, self.true_x)):
                raise AttributeError
            x = self.true_x
        else:
            pdb.set_trace()
        p_x_given_z_logits = self.decoder(z)
        p_x_given_z = torch.distributions.Bernoulli(logits=p_x_given_z_logits[0])
        expected_log_likelihood = 0
        for i in range(len(x)):
            expected_log_likelihood += torch.sum(p_x_given_z.log_prob(x[i][None]), [1, 2, 3])
        log_density = expected_log_likelihood + self.prior.log_prob(z).sum(1)
        return log_density

    def get_density(self, z, x=None):
        return self.get_logdensity(z).exp()

    def get_samples(self, n=1):
        raise NotImplementedError

    def show_x(self, x=None):
        if not x:
            x = self.true_x
        np_image = x.cpu().numpy()
        plt.imshow(np_image[0, 0, :, :])
        plt.show()

    def get_pictures(self, z, nrow=5):
        x_logit = self.decoder(z)[0]
        out = torchvision.utils.make_grid(x_logit, nrow)
        out = out.cpu().numpy()[0]
        plt.imshow(out)
        plt.show()
