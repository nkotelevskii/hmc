import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import matplotlib.pyplot as plt
import sys
import numpy as np
import pdb

sys.path.insert(0, './src/')
from kernels import HMC_our, HMC_vanilla, Reverse_kernel

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
class Target(nn.Module):
    def __init__(self, dec, args):
        super(Target, self).__init__()
        self.decoder = dec(args.data_dim, args.z_dim)
        self.std_normal = args.std_normal
    def get_logdensity(self, z, x, prior=None, args=None, prior_flow=None):
        probs = torch.sigmoid(self.decoder(z))
        return torch.distributions.Bernoulli(probs=probs).log_prob(x).sum(1) + self.std_normal.log_prob(z).sum(1)

def run_training_our_var(args, enc, dec, transition_kernel, dataloader, reverse=None, accept=None, get_prior=None, prior_flow=None):
    #########################################################################################################################
    encoder = enc(args.data_dim, args.z_dim).to(args.device)
    target = Target(dec, args).to(args.device)
    if args.learnable_reverse:
        reverse_kernel = reverse(args).to(args.device)
        reverse_params = reverse_kernel.parameters()
    else:
        reverse_params = list([])
    if args.amortize:
        transitions = transition_kernel(kwargs=args).to(args.device)
    else:
        transitions = nn.ModuleList([transition_kernel(kwargs=args).to(args.device) for _ in range(args['K'])])
        
    if args.fix_transition_params:
        for p in transitions.parameters():
            transitions.requires_grad_(False)
            
    params = list(encoder.parameters()) + list(target.parameters()) + list(transitions.parameters()) + list(reverse_params)
    optimizer = torch.optim.Adam(params=params, lr=args.learning_rate)
    #########################################################################################################################
    
    print_info_ = args.print_info_
#     best_elbo = -float("inf")
#     current_elbo_val = -float("inf")
#     current_tolerance = 0
    for ep in tqdm(range(args.num_epoches)): # cycle over epoches
        for b_num, batch_train in enumerate(dataloader): # cycle over batches
            
            enc_out = encoder(batch_train)
            if len(enc_out) == 2:
                mu, sigma = enc_out
                h = mu
            else:
                mu, sigma, h = enc_out
                
            u = args.std_normal.sample(mu.shape) # sample random tensor for reparametrization trick
            z = mu + sigma * u # reperametrization trick

            p_old = args.std_normal.sample(z.shape)
            p = p_old
            #############################
            sum_log_sigma = torch.sum(torch.log(sigma), 1)
            sum_log_alpha = torch.zeros(z.shape[0], dtype=args.torchType, device=args.device) # for grad log alpha accumulation
            sum_log_jacobian = torch.zeros(z.shape[0], dtype=args.torchType, device=args.device) # for log_jacobian accumulation

            if args.learnable_reverse:
                all_directions = torch.tensor([], device=args.device)
            else:
                all_directions = None
                
#             pdb.set_trace()
            for k in range(args.K):
                cond_vectors = args.std_normal.sample(p.shape)
                if args.amortize:
                    z, p, log_jac, current_log_alphas, directions, _ = transitions.make_transition(q_old=z, x=batch_train,
                                                        p_old=p, k=cond_vectors, target_distr=target, args=args, get_prior=get_prior, prior_flow=prior_flow)
                else:
                    z, p, log_jac, current_log_alphas, directions, _ = transitions[k].make_transition(q_old=z, x=batch_train,
                                                                        p_old=p, k=cond_vectors, target_distr=target, args=args, get_prior=get_prior, prior_flow=prior_flow)
                if args.learnable_reverse:
                    all_directions = torch.cat([all_directions, directions.view(-1, 1)], dim=1)
                sum_log_alpha = sum_log_alpha + current_log_alphas
                sum_log_jacobian = sum_log_jacobian + log_jac
            #############################
            
            ##############################################
            #### Objective computation and optimization steps
            if args.learnable_reverse:
                log_r = reverse_kernel(z_fin=z.detach(), h=h.detach(), a=all_directions)
                log_m = args.std_normal.log_prob(u).sum(1) + args.std_normal.log_prob(p_old).sum(1) - sum_log_jacobian - sum_log_sigma + sum_log_alpha
            else:
                torch_log_2 = torch.tensor(np.log(2.), device=args.device, dtype=args.torchType)
                log_r = -args.K * torch_log_2
                log_m = args.std_normal.log_prob(u).sum(1) + args.std_normal.log_prob(p_old).sum(1) - sum_log_jacobian - sum_log_sigma + sum_log_alpha
                
            log_p = target.get_logdensity(z=z, x=batch_train, prior=get_prior, args=args, prior_flow=prior_flow) + args.std_normal.log_prob(p).sum(1)
            elbo_full = log_p + log_r - log_m
            ### Gradient of the first objective:
            obj_1 = torch.mean(elbo_full + elbo_full.detach() * sum_log_alpha)
            (-obj_1).backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if ((ep % print_info_) == 0) and (b_num == len(dataloader) - 1):
                print('elbo_full:', elbo_full.cpu().detach().mean().item())
                print(target.decoder.W.weight.detach().T)
                if args.amortize:
                    print('Stepsize {}'.format(np.exp(transitions.gamma.cpu().detach().item())))
                    print('Autoregression coeff {}'.format(torch.sigmoid(transitions.alpha_logit).cpu().detach().item()))
                else:
                    for k_num in range(args.K):
                        print(k_num)
                        print('Stepsize {}'.format(np.exp(transitions[k_num].gamma.cpu().detach().item())))
                        print('Autoregression coeff {}'.format(torch.sigmoid(transitions[k_num].alpha_logit).cpu().detach().item()))
    return encoder, target
                
#             (-obj_1).backward(retain_graph=True)
#             optimizer_inference.step()
#             optimizer_inference.zero_grad()
#             optimizer_decoder.zero_grad() 

#             ### Gradient of the second objective:
#             log_p = target.get_logdensity(z=z.detach(), x=batch_train, prior=get_prior, args=args, prior_flow=prior_flow) + args.std_normal.log_prob(p.detach()).sum(1)
#             elbo_full = log_p - log_m
#             obj_2 = torch.mean(elbo_full + elbo_full.detach() * sum_log_alpha)
#             (-obj_2).backward()
#             optimizer_decoder.step()
#             optimizer_inference.zero_grad()
#             optimizer_decoder.zero_grad()
    
    
def validate_vae(args, encoder, target, transitions, dataset, get_prior, prior_flow):
    elbo_list = []
    for batch_num, batch_val in enumerate(dataset.next_val_batch()):
        if args.learnable_reverse:
            all_directions = torch.tensor([], device=args.device)
        else:
            all_directions = None
        mu, sigma = encoder(batch_val)
        
        sum_log_alpha = torch.zeros(mu.shape[0], dtype=args.torchType, device=args.device) # for grad log alpha accumulation
        sum_log_jacobian = torch.zeros(mu.shape[0], dtype=args.torchType, device=args.device) # for log_jacobian accumulation
        sum_log_sigma = torch.sum(torch.log(sigma), 1)

        u = args.std_normal.sample(mu.shape)
        z = mu + sigma * u
        
        p_old = args.std_normal.sample(z.shape)
        cond_vectors = [args.std_normal.sample(p_old.shape) for k in range(args.K)]
        p = p_old
        
        for k in range(args.K):
            if args.amortize:
                z, p, log_jac, current_log_alphas, directions, _ = transitions.make_transition(q_old=z, x=batch_val,
                                                    p_old=p, k=cond_vectors[k], target_distr=target, args=args, get_prior=get_prior, prior_flow=prior_flow)
            else:
                z, p, log_jac, current_log_alphas, directions, _ = transitions[k].make_transition(q_old=z, x=batch_val,
                                                                    p_old=p, k=cond_vectors[k], target_distr=target, args=args, get_prior=get_prior, prior_flow=prior_flow) # sample a_i -- directions
            if args.learnable_reverse:
                all_directions = torch.cat([all_directions, directions.view(-1, 1)], dim=1)
            sum_log_alpha = sum_log_alpha + current_log_alphas
            sum_log_jacobian = sum_log_jacobian + log_jac  # refresh log jacobian
        
        elbo_current, _ = compute_loss(z_new=z, p_new=p, u=u, p_old=p_old, x=batch_val, sum_log_alpha=sum_log_alpha,
                                    sum_log_jac=sum_log_jacobian, sum_log_sigma=sum_log_sigma, mu=mu, all_directions=all_directions,
                                      get_prior=get_prior, args=args, prior_flow=prior_flow)
        
        elbo_list.append(elbo_current.cpu().detach().numpy())
    mean_val_elbo = np.mean(elbo_list)
    return mean_val_elbo