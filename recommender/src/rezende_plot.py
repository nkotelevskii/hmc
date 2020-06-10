import torch
import numpy as np
import torch.nn as nn
import pdb
from tqdm import tqdm

import pyro
from pyro.distributions import AffineCoupling
from pyro.infer.mcmc import HMC, MCMC, NUTS
from kernels import HMC_vanilla

class Target():
    def __init__(self, cur_dat, energy=False):
        self.data = cur_dat
        self.energy = energy

    def get_logdensity(self, z):
        if self.energy:
            z = z['points']
        if self.data == 't1':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            z_norm = torch.norm(z, 2, 1)
            add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
            add2 = - torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + \
                               torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2) + 1e-9)
            if self.energy:
                return add1 + add2
            else:
                return -add1 - add2

        elif self.data == 't2':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            if self.energy:
                return 0.5 * ((z[:, 1] - w1) / 0.4) ** 2
            else:
                return -0.5 * ((z[:, 1] - w1) / 0.4) ** 2
        elif self.data == 't3':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            w2 = 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
            in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.35) ** 2)
            in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w2) / 0.35) ** 2)
            if self.energy:
                return -torch.log(in1 + in2 + 1e-9)
            else:
                return torch.log(in1 + in2 + 1e-9)
        elif self.data == 't4':
            if len(z.shape) == 1:
                z = z.view(1, 2)
            w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
            w3 = 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)
            in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.4) ** 2)
            in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w3) / 0.35) ** 2)
            if self.energy:
                return -torch.log(in1 + in2 + 1e-9)
            else:
                return torch.log(in1 + in2 + 1e-9)
        else:
            raise RuntimeError

    def get_density(self, z, x=None):
        density = self.distr.log_prob(z).exp()
        return density

    def get_samples(self, n):
        raise NotImplementedError



def run_rezende(args):
    pdb.set_trace()
    prior = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                       scale=torch.tensor(0., dtype=args.torchType, device=args.device))
    # # rnvp
    # all_samples = run_rezende_rnvp(args, prior)
    # for i, samples in enumerate(all_samples):
    #     np.savetxt('./rnvp_{}.txt'.format(i), samples.cpu().detach().numpy())

    # nuts
    all_samples = run_rezende_nuts(args, prior, n_chains=1)
    for i, samples in enumerate(all_samples):
        np.savetxt('./nuts_{}.txt'.format(i), samples.cpu().detach().numpy())

    # hoffman
    all_samples = run_rezende_hoffman(args, prior)
    for i, samples in enumerate(all_samples):
        np.savetxt('./hoffman_{}.txt'.format(i), samples.cpu().detach().numpy())

def run_rezende_hoffman(args, prior):
    # hoffman
    all_samples = []
    for cur_dat in ['t1', 't2', 't3', 't4']:
        target = Target(cur_dat).get_logdensity
        ##### Minimize KL first
        mu_init_hoff = nn.Parameter(torch.zeros(args.data_dim, device=args.device, dtype=args.torchType))
        sigma_init_hoff = nn.Parameter(torch.ones(args.data_dim, device=args.device, dtype=args.torchType))
        optimizer = torch.optim.Adam(params=[mu_init_hoff, sigma_init_hoff])

        for i in tqdm(range(10000)):
            u_init = args.std_normal.sample((500, 2))
            q_init = mu_init_hoff + nn.functional.softplus(sigma_init_hoff) * u_init

            current_kl = prior.log_prob(u_init).sum(1) - torch.sum(
                nn.functional.softplus(sigma_init_hoff).log()) - target(z=q_init)
            torch.mean(current_kl).backward()  ## minimize KL
            optimizer.step()
            optimizer.zero_grad()
            if i % 2000 == 0:
                print(current_kl.mean().cpu().detach().numpy())

        samples = prior.sample((10000, 2))

        q_new = mu_init_hoff + samples * nn.functional.softplus(
            sigma_init_hoff)
        p_new = prior.sample(q_new.shape)

        samples_hoffman = torch.empty((args.n_steps, args.z_dim), device=args.device,
                                      dtype=args.torchType)

        vanilla_kernel = HMC_vanilla(args)
        print("Now we are sampling!")
        for i in tqdm(range(args.K)):
            q_new, p_new, _, _, a, _ = vanilla_kernel.make_transition(q_old=q_new, p_old=p_new, target_distr=target)
            samples_hoffman[i] = q_new
        all_samples.append(samples_hoffman.cpu().detach().numpy())

    return all_samples

def run_rezende_nuts(args, prior, n_chains=1):
    # nuts
    # Init NUTS sampler
    all_samples = []
    for cur_dat in ['t1', 't2', 't3', 't4']:
        targ_log_dens = Target(cur_dat, True).get_logdensity
        kernel = NUTS(potential_fn=targ_log_dens)
        target_samples = torch.tensor([], device=args.device)
        for _ in range(n_chains):
            init_samples = prior.sample((args.z_dim,)).view(1, args.z_dim)
            init_params = {'points': init_samples}
            mcmc = MCMC(kernel=kernel, num_samples=10000 // n_chains,
                        initial_params=init_params,
                        num_chains=1, warmup_steps=2000)
            mcmc.run()
            samples = torch.cat([target_samples, mcmc.get_samples()['points']], dim=0)
        samples = samples.squeeze().cpu().numpy()
        all_samples.append(samples)
    return all_samples

def run_rezende_rnvp(args, prior):
    # rnvp
    args['z_dim'] = 2
    args['hidden_dim'] = 2
    num_batches = 5000

    args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']), nn.Tanh())
    args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))
    all_samples = []
    ################################################################################################
    for cur_dat in ['t1', 't2', 't3', 't4']:
        target = Target(cur_dat).get_logdensity

        transitions_rnvp = nn.ModuleList([RealNVP_new(kwargs=args, device=args.device).to(args.device) for _ in range(args['K'])])
        optimizer_rnvp = torch.optim.Adam(params=transitions_rnvp.parameters())

        for _ in tqdm(range(num_batches)):
            optimizer_rnvp.zero_grad()
            u = prior.sample((args['batch_size_train'], args.z_dim))
            sum_log_jacobian = torch.zeros(u.shape[0], dtype=args.torchType, device=args.device)  # for log_jacobian accumulation
            z = u
            for k in range(args.K):
                z_upd, log_jac = transitions_rnvp[k]._forward_step(z)
                sum_log_jacobian = sum_log_jacobian + log_jac  # refresh log jacobian
                z = z_upd
            log_p = target(z)
            log_q = prior.log_prob(u).sum(1) - sum_log_jacobian
            elbo = (log_p - log_q).mean()
            (-elbo).backward()
            optimizer_rnvp.step()
            optimizer_rnvp.zero_grad()

        samples = prior.sample((10000, 2))
        with torch.no_grad():
            for rnvp in transitions_rnvp:
                samples = rnvp(samples)
        all_samples.append(samples.cpu().detach().numpy())
        return all_samples