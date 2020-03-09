import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, './src')
from data import Dataset
from models import Gen_network, Inf_network
from plotting import plot_figures, plot_figures_separated, plot_gibbs, plot_mixture_plots, \
    plot_densities
from target import GMM_target, Funnel, BNAF_examples, GMM_target2, NN_bernoulli, \
    MNIST_target
from tqdm import tqdm
from transitions import RealNVP_new

warnings.simplefilter("ignore", UserWarning)
import torch
import torch.nn as nn

tolerance_for_early_stopping_epoches = 25
tolerance_for_early_stopping_batches = 500


def train_gibbs(args):
    current_tol = 0

    best_elbo_val = -np.float("inf")
    dataset = Dataset(args, args.device)
    num_epoches = args['num_epoches']
    K = args['K']

    encoder = Inf_network(kwargs=args).to(args.device)
    target = NN_bernoulli(kwargs=args, model=Gen_network(args.z_dim, args), device=args.device).to(args.device)

    target.decoder.load_state_dict(torch.load("./models/VAE_NAF_decoder_state_dict.pt"))
    target.decoder.eval()
    for p in target.decoder.parameters():
        p.requires_grad = False

    encoder.load_state_dict(torch.load("./models/VAE_NAF_encoder_state_dict.pt"))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    flow = torch.load('./models/VAE_NAF_flow.pt', map_location=args.device)

    if args.step_conditioning is None:
        transitions = nn.ModuleList([RealNVP_new(kwargs=args,
                                                 device=args.device).to(args.device) for _ in range(args['K'])])
    else:
        transitions = RealNVP_new(kwargs=args, device=args.device).to(args.device)

    optimizer_var_distr = torch.optim.Adam(params=transitions.parameters(), lr=1e-4)

    print_info_ = args.print_info

    torch_log_2 = torch.tensor(np.log(2.), device=args.device, dtype=args.torchType)
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                            scale=torch.tensor(1., dtype=args.torchType, device=args.device))

    if args.step_conditioning == 'fixed':
        cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(args.K)]
    else:
        cond_vectors = [None] * args.K

    def compute_loss(z, x, u, sum_log_jacobian, sum_log_alpha, sum_log_probs, sum_log_sigma):
        log_r = -args.K * torch_log_2
        log_q = std_normal.log_prob(u).sum(1) - sum_log_sigma - sum_log_jacobian + sum_log_alpha
        log_p = target.get_logdensity(z=z, x=x)
        elbo_full = log_p + log_r - log_q
        grad_elbo = torch.mean(elbo_full + elbo_full.detach() * (sum_log_alpha + sum_log_probs))
        return elbo_full.detach().mean().item(), grad_elbo

    prob_tran = 0
    #################################### Training ##########################################
    for ep in tqdm(range(num_epoches)):
        for b_num, batch_train in enumerate(dataset.next_train_batch()):
            if args.step_conditioning == 'free':
                cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]

            sum_log_alpha = torch.zeros(batch_train.shape[0], dtype=args.torchType,
                                        device=args.device)
            sum_log_jacobian = torch.zeros(batch_train.shape[0], dtype=args.torchType,
                                           device=args.device)
            sum_log_probs = torch.zeros(batch_train.shape[0], dtype=args.torchType,
                                        device=args.device)
            mu, sigma = encoder(batch_train)
            u = std_normal.sample(mu.shape)
            z = mu + sigma * u
            sum_log_sigma = torch.sum(torch.log(sigma), 1)

            for k in range(K):
                if args.step_conditioning is None:
                    z, log_jac, current_log_alphas, current_log_probs, directions = transitions[k].make_transition(
                        z_old=z, k=
                        cond_vectors[k], target_distr=target, x=batch_train)
                else:
                    z, log_jac, current_log_alphas, current_log_probs, directions = transitions.make_transition(
                        z_old=z, k=
                        cond_vectors[k], target_distr=target, x=batch_train)
                if ep % print_info_ == 0:
                    if b_num % 100 == 0:
                        print(
                            'For batch {} (epoch {}) for k = {} we have for -1: {}, for 0: {} and for +1: {}'.format(
                                b_num + 1, ep + 1, k + 1,
                                (directions == -1.).to(float).mean(), (directions == 0.).to(float).mean(),
                                (directions == 1.).to(float).mean()))
                sum_log_probs = sum_log_probs + current_log_probs
                sum_log_alpha = sum_log_alpha + current_log_alphas
                sum_log_jacobian = sum_log_jacobian + log_jac

            elbo_full, grad_elbo = compute_loss(z=z, x=batch_train, u=u, sum_log_jacobian=sum_log_jacobian,
                                                sum_log_alpha=sum_log_alpha, sum_log_probs=sum_log_probs,
                                                sum_log_sigma=sum_log_sigma)
            (-grad_elbo).backward()
            optimizer_var_distr.step()
            optimizer_var_distr.zero_grad()

        ########################### Validation: ####################################
        with torch.no_grad():
            mean_val_elbo = []
            for val_batch in dataset.next_val_batch():
                mu, sigma = encoder(val_batch)
                u = std_normal.sample(mu.shape)
                z = mu + sigma * u
                sum_log_sigma = torch.sum(torch.log(sigma), 1)
                sum_log_alpha = torch.zeros(val_batch.shape[0], dtype=args.torchType,
                                            device=args.device)
                sum_log_jacobian = torch.zeros(val_batch.shape[0], dtype=args.torchType,
                                               device=args.device)
                sum_log_probs = torch.zeros(val_batch.shape[0], dtype=args.torchType,
                                            device=args.device)

                if args.step_conditioning == 'free':
                    cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in
                                    range(K)]
                for k in range(K):
                    if args.step_conditioning is None:
                        z, log_jac, current_log_alphas, current_log_probs, directions = transitions[
                            k].make_transition(
                            z_old=z, k=cond_vectors[k], target_distr=target, x=val_batch)
                    else:
                        z, log_jac, current_log_alphas, current_log_probs, directions = transitions.make_transition(
                            z_old=z, k=cond_vectors[k], target_distr=target, x=val_batch)

                elbo_full_val, _ = compute_loss(z=z, x=val_batch, u=u, sum_log_jacobian=sum_log_jacobian,
                                                sum_log_alpha=sum_log_alpha, sum_log_probs=sum_log_probs,
                                                sum_log_sigma=sum_log_sigma)
                mean_val_elbo.append(elbo_full_val)
            mean_val_elbo = torch.mean(torch.tensor(mean_val_elbo))
            if ((mean_val_elbo != mean_val_elbo).sum()) or np.isnan(prob_tran):
                print('NAN appeared!')
                raise ValueError

            if mean_val_elbo > best_elbo_val:
                current_tol = 0
                best_elbo_val = mean_val_elbo
                torch.save(transitions,
                           './models/best_flows_{}_{}_{}_{}.pt'.format(args.problem, args.data, args.step_conditioning,
                                                                       args.use_barker))
            else:
                current_tol += 1
                if current_tol >= tolerance_for_early_stopping_epoches:
                    print('Early stopping on {}'.format(ep))
                    break

        if ep % print_info_ == 0:
            print('Current epoch:', ep, '\t', 'Current ELBO val:', mean_val_elbo, '\t', 'Best ELBO val:',
                  best_elbo_val)
            for kk in range(K):
                if args.step_conditioning is None:
                    print('For {}-th flow the prob of forward transition is {}'.format(kk + 1, torch.sigmoid(
                        transitions[kk].logit.cpu().data)))
                    prob_tran = torch.sigmoid(transitions[kk].logit.cpu().detach()).item()
                else:
                    print('For {}-th flow the prob of forward transition is {}'.format(kk + 1, torch.sigmoid(
                        transitions.logit.cpu().data)))
                    prob_tran = torch.sigmoid(transitions.logit.cpu().detach()).item()
                    break

    #### Save
    torch.save(transitions, './models/vae_flows_{}_{}_{}.pt'.format(args.data, args.step_conditioning, args.use_barker))
    if args.step_conditioning == 'fixed':
        cond_v = torch.cat([cv[0] for cv in cond_vectors])
        torch.save(cond_v, './models/vae_cond_vectors_{}_{}.pt'.format(args.K, args.use_barker))

    #### Plotting ###
    transitions = torch.load(
        './models/best_flows_{}_{}_{}_{}.pt'.format(args.problem, args.data, args.step_conditioning,
                                                    args.use_barker))

    transitions.eval()
    for p in transitions.parameters():
        p.requires_grad = False

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    target.decoder.eval()
    for p in target.decoder.parameters():
        p.requires_grad = False

    plot_gibbs(encoder=encoder, target=target, metflow=transitions, flow=flow, args=args, dataset=dataset,
               cond_vectors=cond_vectors)


def train_sampling(args):
    current_tol = 0
    best_elbo = -np.float("inf")
    args["true_x"] = None
    if args.data == 'gmm2':
        target = GMM_target2(kwargs=args, device=args.device)
    elif args.data == 'gmm':
        target = GMM_target(kwargs=args, device=args.device)
    elif args.data == 'funnel':
        target = Funnel(kwargs=args, device=args.device)
    elif args.data == 'mnist':
        txt_files = [p for p in os.listdir('./data/') if
                     p.endswith('.txt') and str.isdigit(p[-5]) and p.startswith('fixed_image')]
        args["true_x"] = torch.tensor([], device=args.device)
        for txt in txt_files:
            args["true_x"] = torch.cat(
                [args["true_x"], torch.tensor(np.loadtxt('./data/{}'.format(txt)), device=args.device,
                                              dtype=args.torchType)[None, None, :, :]], dim=0)

        decoder_loaded = Gen_network(args.z_dim, args)
        decoder_path = "./models/VAE_NAF_decoder_state_dict.pt"
        decoder_loaded.load_state_dict(torch.load(decoder_path))
        for p in decoder_loaded.parameters():
            p.requires_grad = False
        args["decoder"] = decoder_loaded
        target = MNIST_target(args, args.device).to(args.device)
    else:
        raise NotImplementedError
    num_batches = args['num_batches']
    K = args['K']

    if args.step_conditioning is None:
        transitions = nn.ModuleList([RealNVP_new(kwargs=args,
                                                 device=args.device).to(args.device) for _ in range(args['K'])])
    else:
        transitions = RealNVP_new(kwargs=args, device=args.device).to(args.device)

    params = transitions.parameters()
    optimizer = torch.optim.Adam(params=params)

    print_info_ = args.print_info

    torch_log_2 = torch.tensor(np.log(2.), device=args.device, dtype=args.torchType)
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                            scale=torch.tensor(1., dtype=args.torchType, device=args.device))

    if args.step_conditioning == 'fixed':
        cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]
    else:
        cond_vectors = [None] * K

    def compute_loss(z, x, u, sum_log_jacobian, sum_log_alpha, sum_log_probs):
        log_p = target.get_logdensity(z=z, x=x)
        log_r = -args.K * torch_log_2
        log_m = std_normal.log_prob(u).sum(1) - sum_log_jacobian + sum_log_alpha
        elbo_full = log_p + log_r - log_m
        grad_elbo = torch.mean(elbo_full + elbo_full.detach() * (sum_log_alpha + sum_log_probs))
        return elbo_full.detach().mean().item(), grad_elbo

    iterator = tqdm(range(num_batches))
    for batch_num in iterator:
        plt.close()
        if args.step_conditioning == 'free':
            cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]
        u = std_normal.sample(
            (args['batch_size_train'], args.z_dim))
        sum_log_alpha = torch.zeros(u.shape[0], dtype=args.torchType,
                                    device=args.device)
        sum_log_probs = torch.zeros(u.shape[0], dtype=args.torchType,
                                    device=args.device)
        sum_log_jacobian = torch.zeros(u.shape[0], dtype=args.torchType,
                                       device=args.device)
        z = u
        if (batch_num) % print_info_ == 0:
            array_z = []
            array_directions = []
            array_alpha = []
        for k in range(K):
            if args.step_conditioning is None:
                z, log_jac, current_log_alphas, \
                current_log_probs, directions = transitions[k].make_transition(z_old=z, k=cond_vectors[k],
                                                                               target_distr=target,
                                                                               x=args["true_x"])
            else:
                z, log_jac, current_log_alphas, \
                current_log_probs, directions = transitions.make_transition(z_old=z, k=cond_vectors[k],
                                                                            target_distr=target, x=args["true_x"])
            if (batch_num) % print_info_ == 0:
                print('On epoch number {} and on k = {} we have for -1: {}, for 0: {} and for +1: {}'.format(
                    batch_num + 1, k + 1,
                    (directions == -1.).to(float).mean(), (directions == 0.).to(float).mean(),
                    (directions == 1.).to(float).mean()))
            sum_log_alpha = sum_log_alpha + current_log_alphas
            sum_log_probs = sum_log_probs + current_log_probs
            sum_log_jacobian = sum_log_jacobian + log_jac
            if (batch_num) % print_info_ == 0 and args.problem != 'mixture':
                array_z.append(z.detach())
                array_directions.append(directions.detach())
                array_alpha.append(current_log_alphas.detach())

        elbo_full, grad_elbo = compute_loss(z=z, x=args["true_x"], u=u, sum_log_jacobian=sum_log_jacobian,
                                            sum_log_alpha=sum_log_alpha, sum_log_probs=sum_log_probs)
        (-grad_elbo).backward()
        optimizer.step()
        optimizer.zero_grad()

        if np.isnan(elbo_full):
            print('NAN appeared!')
            raise ValueError

        if elbo_full > best_elbo:
            best_elbo = elbo_full
            current_tol = 0
            torch.save(transitions,
                       './models/best_flows_{}_{}_{}_{}.pt'.format(args.problem, args.data, args.step_conditioning,
                                                                   args.use_barker))
        else:
            current_tol += 1
            if current_tol >= tolerance_for_early_stopping_batches:
                print('Early stopping on {}'.format(batch_num))
                iterator.close()
                break

        if (batch_num) % print_info_ == 0:
            print('Current epoch:', (batch_num + 1), '\t', 'Current ELBO:', elbo_full, '\t',
                  'Best ELBO:', best_elbo)
            for kk in range(K):
                if args.step_conditioning is None:
                    print('For {}-th flow the prob of forward transition is {}'.format(kk + 1, torch.sigmoid(
                        transitions[kk].logit.cpu().data)))
                else:
                    print('For {}-th flow the prob of forward transition is {}'.format(kk + 1, torch.sigmoid(
                        transitions.logit.cpu().data)))
                    break
            # if args.problem != 'mixture':
            #     fig, ax = plt.subplots(ncols=K, figsize=(30, 5))
            #     label = ['Backward', 'Same', 'Forward']
            #     for kk in range(K):
            #         for d in [0., -1., 1.]:
            #             z_c = array_z[kk][array_directions[kk] == d]
            #             alpha_c = array_alpha[kk][array_directions[kk] == d].cpu().exp().numpy()
            #             color = np.zeros((z_c.shape[0], 4))
            #             color[:, 3] = alpha_c
            #             color[:, int(d + 1)] = 1.
            #             ax[kk].scatter(z_c[:, 0].cpu().numpy(), z_c[:, 1].cpu().numpy(), color=color,
            #                            label=label[int(d + 1)])
            #             ax[kk].legend()
            #     if not os.path.exists('./pics/{}/'.format(args.problem)):
            #         os.makedirs('./pics/{}/'.format(args.problem))
            #     plt.tight_layout()
            #     plt.savefig('./pics/{}/{}_{}_batch_{}_{}_{}_{}.png'.format(args.problem, args.problem,
            #                                                                args.data, batch_num, args.step_conditioning,
            #                                                                args.noise_aggregation,
            #                                                                args.use_barker), format='png')

    path_for_saving = './models/{}_{}_{}_{}_{}.pt'.format(args.problem, args.data,
                                                          args.step_conditioning, args.noise_aggregation,
                                                          args.use_barker)
    torch.save(transitions, path_for_saving)
    print('\n')
    print('Model for MetFlow was saved: {}'.format(path_for_saving))
    print('\n')

    transitions = torch.load(
        './models/best_flows_{}_{}_{}_{}.pt'.format(args.problem, args.data, args.step_conditioning,
                                                    args.use_barker))
    for p in transitions.parameters():
        p.requires_grad = False

    if args.problem != 'mixture':
        ##################################################################################################
        ########################################### RNVP alone ###########################################
        ##################################################################################################
        rnvp_models = []
        print('\n')
        for rnvp_run in range(2):
            best_elbo = -np.float("inf")
            current_tol = 0
            print('Now we are training {}-run of RNVP'.format(rnvp_run + 1))

            args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                                 nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                                 nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']),
                                                 nn.Tanh())
            args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                                 nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                                 nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))

            transitions_rnvp = nn.ModuleList([RealNVP_new(kwargs=args,
                                                          device=args.device).to(args.device) for _ in
                                              range(args['K'])])
            best_transitions_rnvp = nn.ModuleList([RealNVP_new(kwargs=args,
                                                               device=args.device).to(args.device) for _ in
                                                   range(args['K'])])
            params = transitions_rnvp.parameters()
            optimizer_rnvp = torch.optim.Adam(params=params)

            iterator = tqdm(range(num_batches))
            for batch_num in iterator:  # cycle over batches
                u = std_normal.sample((args['batch_size_train'], args.z_dim))
                sum_log_jacobian = torch.zeros(u.shape[0], dtype=args.torchType, device=args.device)
                z = u
                for k in range(K):
                    z_upd, log_jac = transitions_rnvp[k]._forward_step(z)
                    sum_log_jacobian = sum_log_jacobian + log_jac
                    z = z_upd
                log_p = target.get_logdensity(z)
                log_q = std_normal.log_prob(u).sum(1) - sum_log_jacobian
                elbo = (log_p - log_q).mean()
                (-elbo).backward()
                if elbo > best_elbo:
                    best_elbo = elbo
                    current_tol = 0
                else:
                    current_tol += 1
                    if current_tol >= tolerance_for_early_stopping_batches:
                        best_transitions_rnvp.load_state_dict(transitions_rnvp.state_dict())
                        print('NAF early stopping on batch ', batch_num)
                        iterator.close()
                        break
                optimizer_rnvp.step()
                optimizer_rnvp.zero_grad()
                if (batch_num + 1) % 1000 == 0:
                    print('Current epoch:', (batch_num + 1), '\t', 'Current ELBO:', elbo.item())
            rnvp_models.append(best_transitions_rnvp)

    if args['plot_all_pics'] and args.problem != 'mixture':
        # plot_repetitions(transitions=transitions, target=target, cond_vectors=cond_vectors,
        #                  args=args, rnvp_models=rnvp_models)
        plot_figures(transitions=transitions, target=target, cond_vectors=cond_vectors,
                     args=args, rnvp_models=rnvp_models)
        plot_figures_separated(transitions=transitions, target=target, cond_vectors=cond_vectors,
                               args=args, rnvp_models=rnvp_models)
    if args.problem == 'mixture':
        plot_mixture_plots(args=args, target=target, transitions=transitions, cond_vectors=cond_vectors,
                           tol=tolerance_for_early_stopping_batches)
    return z


def train_densities(args):
    if args.data != 'rezende':
        return NotImplementedError

    num_batches = args['num_batches']
    K = args['K']

    print_info_ = args.print_info

    torch_log_2 = torch.tensor(np.log(2.), device=args.device, dtype=args.torchType)
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                            scale=torch.tensor(1., dtype=args.torchType, device=args.device))

    models_for_experiments_ours = []
    cond_vectors_ours = []

    early_stopping = tolerance_for_early_stopping_batches * 2
    ################################################################################################
    for cur_dat in ['t1', 't2', 't3', 't4']:
        best_elbo = -float('inf')
        current_tol = 0
        args['bnaf_data'] = cur_dat
        target = BNAF_examples(kwargs=args, device=args.device)

        if args.step_conditioning is None:
            transitions = nn.ModuleList([RealNVP_new(kwargs=args,
                                                     device=args.device).to(args.device) for _ in range(args['K'])])
            best_transitions = nn.ModuleList([RealNVP_new(kwargs=args,
                                                          device=args.device).to(args.device) for _ in
                                              range(args['K'])])
        else:
            transitions = RealNVP_new(kwargs=args, device=args.device).to(args.device)
            best_transitions = RealNVP_new(kwargs=args, device=args.device).to(args.device)

        optimizer = torch.optim.Adam(params=transitions.parameters())

        if args.step_conditioning == 'fixed':
            if args.noise_aggregation in ['addition', 'stacking']:
                cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for k in range(K)]
        else:
            cond_vectors = [None] * K
        cond_vectors_ours.append(cond_vectors)

        def compute_loss(z, u, sum_log_jacobian, sum_log_alpha, sum_log_probs):
            log_p = target.get_logdensity(z)
            log_r = -args.K * torch_log_2
            log_m_tilde = std_normal.log_prob(u).sum(1) - sum_log_jacobian
            log_m = log_m_tilde + sum_log_alpha
            elbo_full = log_p + log_r - log_m
            grad_elbo = torch.mean(elbo_full + elbo_full.detach() * (sum_log_alpha + sum_log_probs))
            return elbo_full.cpu().detach().mean(), grad_elbo

        iterator = tqdm(range(num_batches))
        for batch_num in iterator:
            if args.step_conditioning == 'free':
                cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]
            optimizer.zero_grad()
            u = std_normal.sample(
                (args['batch_size_train'], args.z_dim))
            sum_log_alpha = torch.zeros(u.shape[0], dtype=args.torchType, device=args.device)
            sum_log_probs = torch.zeros(u.shape[0], dtype=args.torchType, device=args.device)
            sum_log_jacobian = torch.zeros(u.shape[0], dtype=args.torchType, device=args.device)
            z = u
            for k in range(K):
                if args.step_conditioning is None:
                    z, log_jac, current_log_alphas, current_log_probs, directions = transitions[k].make_transition(
                        z_old=z, k=
                        cond_vectors[k], target_distr=target)
                else:
                    z, log_jac, current_log_alphas, current_log_probs, directions = transitions.make_transition(z_old=z,
                                                                                                                k=
                                                                                                                cond_vectors[
                                                                                                                    k],
                                                                                                                target_distr=target)
                if (batch_num) % print_info_ == 0:
                    print('On epoch number {} and on k = {} we have for -1: {}, for 0: {} and for +1: {}'.format(
                        batch_num + 1, k + 1,
                        (directions == -1.).to(float).mean(), (directions == 0.).to(float).mean(),
                        (directions == 1.).to(float).mean()))
                sum_log_alpha = sum_log_alpha + current_log_alphas
                sum_log_probs = sum_log_probs + current_log_probs
                sum_log_jacobian = sum_log_jacobian + log_jac

            elbo_full, grad_elbo = compute_loss(z, u, sum_log_jacobian, sum_log_alpha, sum_log_probs)
            (-grad_elbo).backward()
            optimizer.step()
            optimizer.zero_grad()
            if elbo_full > best_elbo:
                best_elbo = elbo_full
                current_tol = 0
            else:
                current_tol += 1
                if current_tol >= early_stopping:
                    best_transitions.load_state_dict(transitions.state_dict())
                    print('Early stopping on batch ', batch_num)
                    iterator.close()
                    break
            if (batch_num) % print_info_ == 0:
                print('Current epoch:', (batch_num + 1), '\t', 'Current ELBO:', elbo_full.item(), '\t', 'Best ELBO:',
                      best_elbo.item())
                for kk in range(K):
                    if args.step_conditioning is None:
                        print('For {}-th flow the prob of forward transition is {}'.format(kk + 1, torch.sigmoid(
                            transitions[kk].logit.cpu().data)))
                    else:
                        print('For {}-th flow the prob of forward transition is {}'.format(kk + 1, torch.sigmoid(
                            transitions.logit.cpu().data)))
                        break
        transitions = best_transitions
        models_for_experiments_ours.append(transitions)

    ##################################################################################################
    ########################################### RNVP alone ###########################################
    ##################################################################################################
    args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']), nn.Tanh())
    args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))

    models_for_experiments_rnvp = []
    ################################################################################################
    for cur_dat in ['t1', 't2', 't3', 't4']:
        best_elbo = -float('inf')
        current_tol = 0
        args['bnaf_data'] = cur_dat
        target = BNAF_examples(kwargs=args, device=args.device)

        transitions_rnvp = nn.ModuleList(
            [RealNVP_new(kwargs=args, device=args.device).to(args.device) for _ in range(args['K'])])
        best_transitions_rnvp = nn.ModuleList(
            [RealNVP_new(kwargs=args, device=args.device).to(args.device) for _ in range(args['K'])])

        optimizer_rnvp = torch.optim.Adam(params=transitions_rnvp.parameters())

        iterator = tqdm(range(num_batches))
        for batch_num in iterator:  # cycle over batches
            optimizer_rnvp.zero_grad()
            u = std_normal.sample((args['batch_size_train'], args.z_dim))
            sum_log_jacobian = torch.zeros(u.shape[0], dtype=args.torchType,
                                           device=args.device)  # for log_jacobian accumulation
            z = u
            for k in range(K):
                z, log_jac = transitions_rnvp[k]._forward_step(z)
                sum_log_jacobian = sum_log_jacobian + log_jac  # refresh log jacobian
            log_p = target.get_logdensity(z)
            log_q = std_normal.log_prob(u).sum(1) - sum_log_jacobian
            elbo = (log_p - log_q).mean()
            (-elbo).backward()
            optimizer_rnvp.step()
            optimizer_rnvp.zero_grad()
            if elbo > best_elbo:
                current_tol = 0
                best_elbo = elbo
            else:
                current_tol += 1
                if current_tol >= tolerance_for_early_stopping_batches:
                    best_transitions_rnvp.load_state_dict(transitions_rnvp.state_dict())  # copy weights and stuff
                    print('Early stopping on batch ', batch_num)
                    iterator.close()
                    break

            if (batch_num + 1) % 1000 == 0:
                print('Current epoch:', (batch_num + 1), '\t', 'Current ELBO:', elbo.item())

        transitions_rnvp = best_transitions_rnvp
        models_for_experiments_rnvp.append(transitions_rnvp)

    plot_densities(args, models_for_experiments_rnvp, models_for_experiments_ours, cond_vectors_ours)
