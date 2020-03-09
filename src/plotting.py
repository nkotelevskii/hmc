import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pyro.distributions.transforms import NeuralAutoregressive
from pyro.nn import AutoRegressiveNN
import sys
sys.path.insert(0, './src')
from target import GMM_target, BNAF_examples
from tqdm import tqdm
from utils import get_samples, plot_digit_samples

matplotlib.rcParams.update({'font.size': 20})
NUM_DIGITS_TO_PLOT = 64

NUM_REPETITIONS = 101
INCREMENT = 50
NUM_PICS = 1
NUM_SAMPLES_SAMPLING = 50000


def plot_repetitions(transitions, target, cond_vectors, args, rnvp_models):
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                            scale=torch.tensor(1., dtype=args.torchType, device=args.device))
    u = std_normal.sample((NUM_SAMPLES_SAMPLING, args.z_dim))  # sample random tensor for reparametrization trick
    z = u

    increment = INCREMENT
    K = args.K
    repetitions = NUM_REPETITIONS

    plt.close()
    fig, ax_our = plt.subplots(ncols=repetitions // increment + 2, figsize=(14, 4), dpi=400)
    z_prev = z
    ax_our[0].scatter(z[:, 0].cpu().detach().numpy(), z[:, 1].cpu().detach().numpy(), s=10,
                      c=z_prev[:, 0].cpu().detach().numpy(), cmap='jet', label='Pushforward')
    ax_our[0].set_title('Initial samples')

    with torch.no_grad():
        for rep in range(repetitions):
            for k in range(K):
                if args.step_conditioning == 'free':
                    cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]
                if args.step_conditioning is None:
                    z, _, _, _, _ = transitions[k].make_transition(z_old=z, k=cond_vectors[k],
                                                                   target_distr=target, x=args["true_x"])
                else:
                    z, _, _, _, _ = transitions.make_transition(z_old=z, k=cond_vectors[k],
                                                                target_distr=target, x=args["true_x"])
            if rep % increment == 0:
                ax_our[rep // increment + 1].scatter(z[:, 0].cpu().detach().numpy(), z[:, 1].cpu().detach().numpy(),
                                                     s=10,
                                                     c=z_prev[:, 0].cpu().detach().numpy(), cmap='jet',
                                                     label='Pushforward')
                ax_our[rep // increment + 1].set_xlim((-15, 15))
                ax_our[rep // increment + 1].set_ylim((-15, 15))
                ax_our[rep // increment + 1].set_title('{} repetitions'.format(rep))
                z_prev = z
    plt.tight_layout()
    if not os.path.exists('./pics/{}/'.format(args.problem)):
        os.makedirs('./pics/{}/'.format(args.problem))
    plt.savefig('./pics/{}/{}_repetitions_{}_{}_{}_{}.png'.format(args.problem, args.data, args.K,
                                                                  args.step_conditioning,
                                                                  args.noise_aggregation, args.use_barker),
                format='png')
    plt.close()


def plot_figures(transitions, target, cond_vectors, args, rnvp_models):
    ########### PLOTTING ############
    plt.close()
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                            scale=torch.tensor(1., dtype=args.torchType, device=args.device))

    print('\n')
    print("Now we are plotting Figure 1 from the main paper")
    print('\n')

    ##### Plotting of Figure 1 from the main paper #####
    K = args.K
    s = 10
    repetitions = NUM_REPETITIONS
    # pdb.set_trace()
    npts = 400
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    x_lim = 15
    y_lim = 15

    z_rnvp_samples = []

    ###### Target ########
    xside = np.linspace(-x_lim - 1, x_lim + 1, npts)
    yside = np.linspace(-y_lim - 1, y_lim + 1, npts)
    xx, yy = np.meshgrid(xside, yside)
    z = torch.tensor(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]), device=args.device, dtype=args.torchType)
    logdens = target.get_logdensity(z)
    p = np.exp(logdens.cpu().detach().numpy()).reshape(npts, npts)
    ax[0, 0].set_title('Target')
    ax[0, 0].pcolormesh(xx, yy, p)
    ax[0, 0].set_xlim(-x_lim, x_lim)
    ax[0, 0].set_ylim(-y_lim, y_lim)
    ax[0, 0].set_xticks(np.arange(-x_lim, x_lim + 1, 5))
    ax[0, 0].set_yticks(np.arange(-y_lim, y_lim + 1, 5))
    ax[0, 0].set_aspect('equal', 'box')

    u = std_normal.sample((NUM_SAMPLES_SAMPLING, args.z_dim))

    #### Prior samples
    ax[1, 0].set_title('Prior')
    ax[1, 0].scatter(u.cpu().detach().numpy()[:, 0], u.cpu().detach().numpy()[:, 1], s=s, cmap='jet',
                     c=u.cpu().detach().numpy()[:, 0])
    ax[1, 0].set_aspect('equal', 'box')
    ax[1, 0].set_xlim(-4, 4)
    ax[1, 0].set_ylim(-4, 4)
    ax[1, 0].set_xticks(np.arange(-4, 5, 2))
    ax[1, 0].set_yticks(np.arange(-4, 5, 2))
    z = u

    #### MetFlow
    ax[0, 1].set_title('MetFlow')
    ax[0, 2].set_title('MetFlow ({})'.format(repetitions - 1))

    with torch.no_grad():
        for rep in range(repetitions):
            for k in range(K):
                if args.step_conditioning == 'free':
                    cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for k in
                                    range(K)]
                if args.step_conditioning is None:
                    z, _, _, _, _ = transitions[k].make_transition(z_old=z, k=cond_vectors[k],
                                                                   target_distr=target, x=args["true_x"])
                else:
                    z, _, _, _, _ = transitions.make_transition(z_old=z, k=cond_vectors[k],
                                                                target_distr=target, x=args["true_x"])
            if rep == 0:
                ax[0, 1].scatter(z[:, 0].cpu().detach().numpy(), z[:, 1].cpu().detach().numpy(), s=s,
                                 c=u[:, 0].cpu().detach().numpy(), cmap='jet', label='Pushforward')
                ax[0, 1].set_xlim((-x_lim, x_lim))
                ax[0, 1].set_ylim((-y_lim, y_lim))
                ax[0, 1].set_xticks(np.arange(-x_lim, x_lim + 1, 5))
                ax[0, 1].set_yticks(np.arange(-y_lim, y_lim + 1, 5))
                ax[0, 1].set_aspect('equal', 'box')
            elif rep == (repetitions - 1):
                ax[0, 2].scatter(z[:, 0].cpu().detach().numpy(), z[:, 1].cpu().detach().numpy(), s=s,
                                 c=u[:, 0].cpu().detach().numpy(), cmap='jet', label='Pushforward')
                ax[0, 2].set_xlim((-x_lim, x_lim))
                ax[0, 2].set_ylim((-y_lim, y_lim))
                ax[0, 2].set_xticks(np.arange(-x_lim, x_lim + 1, 5))
                ax[0, 2].set_yticks(np.arange(-y_lim, y_lim + 1, 5))
                ax[0, 2].set_aspect('equal', 'box')
    ###### RNVP ########
    for rnvp_num in range(len(rnvp_models)):
        ax[1, rnvp_num + 1].set_title('RNVP {}'.format(rnvp_num + 1))
        z_rnvp = u
        with torch.no_grad():
            for k in range(K):
                z_rnvp, _ = rnvp_models[rnvp_num][k]._forward_step(z_rnvp)
            z_rnvp_samples.append(z_rnvp.cpu().detach().numpy())
            ax[1, rnvp_num + 1].scatter(z_rnvp[:, 0].cpu().detach().numpy(), z_rnvp[:, 1].cpu().detach().numpy(),
                                        s=s, c=u[:, 0].cpu().detach().numpy(), cmap='jet')
            ax[1, rnvp_num + 1].set_aspect('equal', 'box')
            ax[1, rnvp_num + 1].set_xlim((-x_lim, x_lim))
            ax[1, rnvp_num + 1].set_ylim((-y_lim, y_lim))
            ax[1, rnvp_num + 1].set_xticks(np.arange(-x_lim, x_lim + 1, 5))
            ax[1, rnvp_num + 1].set_yticks(np.arange(-y_lim, y_lim + 1, 5))

    if not os.path.exists('./pics/{}/'.format(args.problem)):
        os.makedirs('./pics/{}/'.format(args.problem))
    path_for_saving = './pics/{}/figure_1_{}_{}_{}_{}_{}.png'.format(args.problem, args.data, args.K,
                                                                     args.step_conditioning, args.noise_aggregation,
                                                                     args.use_barker)
    plt.savefig(path_for_saving, dpi=400, format='png')
    print('Figure 1 from the main paper was saved: {}'.format(path_for_saving))
    plt.close()

    if args.data not in ['gmm2']:
        return
    ####### Plotting supplementary #######

    init_samples_prior = std_normal.sample((NUM_SAMPLES_SAMPLING, args.z_dim))
    cmap = 'jet'

    ######## S1 ########

    fig, ax = plt.subplots(ncols=args.K + 1, nrows=1, figsize=(25, 6), dpi=400)
    fig.suptitle('Our method')
    ax[0].set_title('Initial samples (K=0)')
    ax[0].scatter(init_samples_prior[:, 0].cpu().detach().numpy(), init_samples_prior[:, 1].cpu().detach().numpy(),
                  c=init_samples_prior[:, 0].cpu().detach().numpy(), cmap=cmap, s=0.5)
    ax[0].axis('equal')

    previous_samples = init_samples_prior

    with torch.no_grad():
        for k in range(args.K):
            if args.step_conditioning == 'free':
                cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]

            if args.step_conditioning is None:
                current_samples, _, _, _, _ = transitions[k].make_transition(z_old=previous_samples, k=cond_vectors[k],
                                                                             target_distr=target, x=args["true_x"])
            else:
                current_samples, _, _, _, _ = transitions.make_transition(z_old=previous_samples, k=cond_vectors[k],
                                                                          target_distr=target, x=args["true_x"])
            ax[k + 1].set_title('Samples after {}-th'.format(k + 1))
            ax[k + 1].scatter(current_samples[:, 0].cpu().detach().numpy(),
                              current_samples[:, 1].cpu().detach().numpy(),
                              c=previous_samples[:, 0].cpu().detach().numpy(), cmap=cmap,
                              s=0.5)
            ax[k + 1].set_aspect('equal', adjustable='datalim')
            previous_samples = current_samples
    if not os.path.exists('./pics/supplementary'):
        os.makedirs('./pics/supplementary')
    plt.tight_layout()
    plt.savefig("./pics/supplementary/S1_{}_{}_{}_{}_{}.png".format(args.step_conditioning,
                                                                    args.noise_aggregation, args.K,
                                                                    args.data, args.use_barker), format='png')
    plt.close()

    ######## S2 ########
    fig, ax = plt.subplots(ncols=args.K + 1, nrows=1, figsize=(25, 6), dpi=400)
    fig.suptitle('RNVP')
    ax[0].set_title('Initial samples (K=0)')
    ax[0].scatter(init_samples_prior[:, 0].cpu().detach().numpy(), init_samples_prior[:, 1].cpu().detach().numpy(),
                  c=init_samples_prior[:, 0].cpu().detach().numpy(), cmap=cmap, s=0.5)
    ax[0].axis('equal')

    previous_samples = init_samples_prior

    with torch.no_grad():
        for k in range(args.K):
            z_upd, _ = rnvp_models[0][k]._forward_step(previous_samples)
            current_samples = z_upd
            ax[k + 1].set_title('Samples after {}-th'.format(k + 1))
            ax[k + 1].scatter(current_samples[:, 0].cpu().detach().numpy(),
                              current_samples[:, 1].cpu().detach().numpy(),
                              c=init_samples_prior[:, 0].cpu().detach().numpy(), cmap=cmap,
                              s=0.5)
            ax[k + 1].set_aspect('equal', adjustable='datalim')
            previous_samples = current_samples
    if not os.path.exists('./pics/supplementary'):
        os.makedirs('./pics/supplementary')
    plt.tight_layout()
    plt.savefig("./pics/supplementary/S2_{}_{}_{}_{}_{}.png".format(args.step_conditioning,
                                                                    args.noise_aggregation, args.K,
                                                                    args.data, args.use_barker), format='png')
    plt.close()

    ######## S3 ########
    init_samples_subst = GMM_target(kwargs=args, device=args.device).get_samples(NUM_SAMPLES_SAMPLING)
    fig, ax = plt.subplots(ncols=args.K + 1, nrows=1, figsize=(25, 6), dpi=400)
    fig.suptitle('Our method')
    ax[0].set_title('Initial samples (K=0)')
    ax[0].scatter(init_samples_subst[:, 0].cpu().detach().numpy(), init_samples_subst[:, 1].cpu().detach().numpy(),
                  # c=init_samples_subst[:, 0].cpu().detach().numpy(),
                  cmap=cmap, s=0.5)
    ax[0].axis('equal')

    previous_samples = init_samples_subst
    current_samples = previous_samples
    with torch.no_grad():
        for rep in range(NUM_REPETITIONS + 1):
            for k in range(args.K):
                if args.step_conditioning == 'free':
                    cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for k in range(K)]

                if args.step_conditioning is None:
                    current_samples, _, _, _, _ = transitions[k].make_transition(z_old=current_samples,
                                                                                 k=cond_vectors[k],
                                                                                 target_distr=target, x=args["true_x"])
                else:
                    current_samples, _, _, _, _ = transitions.make_transition(z_old=current_samples, k=cond_vectors[k],
                                                                              target_distr=target, x=args["true_x"])
            if rep % 25 == 0:
                ax[rep // 25 + 1].set_title('Repeated {}'.format(rep))
                ax[rep // 25 + 1].scatter(current_samples[:, 0].cpu().detach().numpy(),
                                          current_samples[:, 1].cpu().detach().numpy(),
                                          # c=init_samples_subst[:, 0].cpu().detach().numpy(), cmap=cmap,
                                          s=0.5)
                ax[rep // 25 + 1].set_aspect('equal', adjustable='datalim')
                previous_samples = current_samples
    if not os.path.exists('./pics/supplementary'):
        os.makedirs('./pics/supplementary')
    plt.tight_layout()
    plt.savefig("./pics/supplementary/S3_{}_{}_{}_{}_{}.png".format(args.step_conditioning,
                                                                    args.noise_aggregation, args.K,
                                                                    args.data, args.use_barker), format='png')
    plt.close()

    ######## S4 ########

    fig, ax = plt.subplots(ncols=args.K + 1, nrows=1, figsize=(25, 6), dpi=400)
    fig.suptitle('RNVP')
    ax[0].set_title('Initial samples (K=0)')
    ax[0].scatter(init_samples_subst[:, 0].cpu().detach().numpy(), init_samples_subst[:, 1].cpu().detach().numpy(),
                  # c=init_samples_subst[:, 0].cpu().detach().numpy(),
                  cmap=cmap, s=0.5)
    ax[0].axis('equal')

    previous_samples = init_samples_subst
    current_samples = previous_samples
    with torch.no_grad():
        for rep in range(NUM_REPETITIONS + 1):
            for k in range(args.K):
                current_samples, _ = rnvp_models[0][k]._forward_step(current_samples)
            if rep % 25 == 0:
                ax[rep // 25 + 1].set_title('Repeated {}'.format(rep))
                ax[rep // 25 + 1].scatter(current_samples[:, 0].cpu().detach().numpy(),
                                          current_samples[:, 1].cpu().detach().numpy(),
                                          # c=init_samples_subst[:, 0].cpu().detach().numpy(), cmap=cmap,
                                          s=0.5)
                ax[rep // 25 + 1].set_aspect('equal', adjustable='datalim')
                previous_samples = current_samples
    if not os.path.exists('./pics/supplementary'):
        os.makedirs('./pics/supplementary')
    plt.tight_layout()
    plt.savefig("./pics/supplementary/S4_{}_{}_{}_{}_{}.png".format(args.step_conditioning,
                                                                    args.noise_aggregation, args.K,
                                                                    args.data, args.use_barker), format='png')
    plt.close()


def plot_figures_separated(transitions, target, cond_vectors, args, rnvp_models):
    plt.close()
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                            scale=torch.tensor(1., dtype=args.torchType, device=args.device))
    K = args.K
    PATH_ROOT = './pics/{}/'.format(args.problem)
    if not os.path.exists('./pics/{}/'.format(args.problem)):
        os.makedirs('./pics/{}/'.format(args.problem))

    # npts = 300
    # bins = 100
    # x_lim = 15
    # y_lim = 15
    #
    # z_ours = None
    # z_ours_rep = None
    # z_rnvp_samples = []
    #
    # ###### Target ########
    # xside = np.linspace(-x_lim - 1, x_lim + 1, npts)
    # yside = np.linspace(-y_lim - 1, y_lim + 1, npts)
    # xx, yy = np.meshgrid(xside, yside)
    # z = torch.tensor(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]), device=args.device, dtype=args.torchType)
    # logdens = target.get_logdensity(z)
    # p = np.exp(logdens.cpu().detach().numpy()).reshape(npts, npts)
    #
    # plt.pcolormesh(xx, yy, p)
    # plt.xlim(-x_lim, x_lim)
    # plt.ylim(-y_lim, y_lim)
    # plt.xticks(np.arange(-x_lim, x_lim + 1, 5))
    # plt.yticks(np.arange(-y_lim, y_lim + 1, 5))
    # plt.axis('equal')
    # plt.axis('off')
    # plt.tight_layout()
    # plt.savefig(PATH_ROOT + 'Target_{}_{}_{}_{}_{}.png'.format(args.data, args.K,
    #                                                            args.step_conditioning, args.noise_aggregation,
    #                                                            args.use_barker), dpi=400, format='png')
    # plt.close()
    #
    # repetitions = NUM_REPETITIONS
    # s = 1
    # u = std_normal.sample((NUM_SAMPLES_SAMPLING, args.z_dim))  # sample random tensor for reparametrization trick
    #
    # #### Prior samples
    # plt.scatter(u.cpu().detach().numpy()[:, 0], u.cpu().detach().numpy()[:, 1], s=s, cmap='jet',
    #             c=u.cpu().detach().numpy()[:, 0])
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    # plt.xticks(np.arange(-4, 5, 2))
    # plt.yticks(np.arange(-4, 5, 2))
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.savefig(PATH_ROOT + 'Prior_{}_{}_{}_{}_{}.png'.format(args.data, args.K,
    #                                                           args.step_conditioning, args.noise_aggregation,
    #                                                           args.use_barker), dpi=400, format='png')
    # plt.close()
    # z = u
    #
    # #### MetFlow
    #
    # with torch.no_grad():
    #     for rep in range(repetitions):
    #         for k in range(K):
    #             if args.step_conditioning == 'free':
    #                 cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for k in
    #                                 range(K)]
    #             if args.step_conditioning is None:
    #                 z, _, _, _, _ = transitions[k].make_transition(z_old=z, k=cond_vectors[k],
    #                                                                target_distr=target, x=args["true_x"])
    #             else:
    #                 z, _, _, _, _ = transitions.make_transition(z_old=z, k=cond_vectors[k],
    #                                                             target_distr=target, x=args["true_x"])
    #         if rep == 0:
    #             plt.scatter(z[:, 0].cpu().detach().numpy(), z[:, 1].cpu().detach().numpy(), s=s,
    #                         c=u[:, 0].cpu().detach().numpy(), cmap='jet', label='Pushforward')
    #             plt.xlim((-x_lim, x_lim))
    #             plt.ylim((-y_lim, y_lim))
    #             plt.xticks(np.arange(-x_lim, x_lim + 1, 5))
    #             plt.yticks(np.arange(-y_lim, y_lim + 1, 5))
    #             plt.axis('equal')
    #             plt.tight_layout()
    #             plt.savefig(PATH_ROOT + 'MetFlow_{}_{}_{}_{}_{}.png'.format(args.data, args.K,
    #                                                                         args.step_conditioning,
    #                                                                         args.noise_aggregation, args.use_barker),
    #                         dpi=400, format='png')
    #             plt.close()
    #             z_ours = z.cpu().detach().numpy()
    #         elif rep == (repetitions - 1):
    #             plt.scatter(z[:, 0].cpu().detach().numpy(), z[:, 1].cpu().detach().numpy(), s=s,
    #                         c=u[:, 0].cpu().detach().numpy(), cmap='jet', label='Pushforward')
    #             plt.xlim((-x_lim, x_lim))
    #             plt.ylim((-y_lim, y_lim))
    #             plt.xticks(np.arange(-x_lim, x_lim + 1, 5))
    #             plt.yticks(np.arange(-y_lim, y_lim + 1, 5))
    #             plt.axis('equal')
    #             plt.tight_layout()
    #             plt.savefig(PATH_ROOT + 'MetFlow_{}_({})_{}_{}_{}_{}.png'.format(args.data, repetitions - 1, args.K,
    #                                                                              args.step_conditioning,
    #                                                                              args.noise_aggregation,
    #                                                                              args.use_barker), dpi=400,
    #                         format='png')
    #             plt.close()
    #             z_ours_rep = z.cpu().detach().numpy()
    # ###### RNVP ########
    # for rnvp_num in range(len(rnvp_models)):
    #     z_rnvp = u
    #     with torch.no_grad():
    #         for k in range(K):
    #             z_rnvp, _ = rnvp_models[rnvp_num][k]._forward_step(z_rnvp)
    #         z_rnvp_samples.append(z_rnvp.cpu().detach().numpy())
    #         plt.scatter(z_rnvp[:, 0].cpu().detach().numpy(), z_rnvp[:, 1].cpu().detach().numpy(),
    #                     s=s, c=u[:, 0].cpu().detach().numpy(), cmap='jet')
    #         plt.xlim((-x_lim, x_lim))
    #         plt.ylim((-y_lim, y_lim))
    #         plt.xticks(np.arange(-x_lim, x_lim + 1, 5))
    #         plt.yticks(np.arange(-y_lim, y_lim + 1, 5))
    #         plt.axis('equal')
    #         plt.tight_layout()
    #         plt.savefig(PATH_ROOT + 'RNVP_run_{}_{}_{}_{}_{}_{}.png'.format(rnvp_num + 1, args.data, args.K,
    #                                                                         args.step_conditioning,
    #                                                                         args.noise_aggregation, args.use_barker),
    #                     dpi=400, format='png')
    #         plt.close()
    #
    # ########################################## Same 2d hist plot ##########################################
    # print('\n')
    # print("Now we are plotting Figure 1 with hist2d from the main paper")
    # print('\n')
    # for gamma in np.arange(0.1, 1., 0.1):
    #     norm = matplotlib.colors.PowerNorm(gamma)
    #     #### Target ####
    #
    #     plt.pcolormesh(xx, yy, p, norm=norm)
    #     plt.xlim(-x_lim, x_lim)
    #     plt.ylim(-y_lim, y_lim)
    #     plt.xticks(np.arange(-x_lim, x_lim + 1, 5))
    #     plt.yticks(np.arange(-y_lim, y_lim + 1, 5))
    #     plt.axis('equal')
    #     plt.axis('off')
    #     plt.tight_layout()
    #     plt.savefig(PATH_ROOT + 'Target_hist2d_gamma_{}_{}_{}_{}_{}_{}.png'.format(gamma, args.data, args.K,
    #                                                                                args.step_conditioning,
    #                                                                                args.noise_aggregation,
    #                                                                                args.use_barker), dpi=400,
    #                 format='png')
    #     plt.close()
    #     #### Prior samples
    #     h, _, _ = np.histogram2d(u.cpu().detach().numpy()[:, 0], u.cpu().detach().numpy()[:, 1],
    #                              bins=bins, density=True, range=[[-x_lim, x_lim], [-y_lim, y_lim]])
    #     plt.imshow(h.T, interpolation='lanczos', norm=norm)
    #     plt.axis('off')
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     plt.savefig(PATH_ROOT + 'Prior_hist2d_gamma_{}_{}_{}_{}_{}_{}.png'.format(gamma, args.data, args.K,
    #                                                                               args.step_conditioning,
    #                                                                               args.noise_aggregation,
    #                                                                               args.use_barker), dpi=400,
    #                 format='png')
    #     plt.close()
    #     #### MetFlow
    #     h, _, _ = np.histogram2d(z_ours[:, 0], z_ours[:, 1],
    #                              bins=bins, density=True, range=[[-x_lim, x_lim], [-y_lim, y_lim]])
    #     plt.imshow(h.T, interpolation='lanczos', norm=norm)
    #     plt.axis('off')
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     plt.savefig(PATH_ROOT + 'MetFlow_hist2d_gamma_{}_{}_{}_{}_{}_{}.png'.format(gamma, args.data, args.K,
    #                                                                                 args.step_conditioning,
    #                                                                                 args.noise_aggregation,
    #                                                                                 args.use_barker), dpi=400,
    #                 format='png')
    #     plt.close()
    #
    #     h, _, _ = np.histogram2d(z_ours_rep[:, 0], z_ours_rep[:, 1],
    #                              bins=bins, density=True, range=[[-x_lim, x_lim], [-y_lim, y_lim]])
    #     plt.imshow(h.T, interpolation='lanczos', norm=norm)
    #     plt.axis('off')
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     plt.savefig(
    #         PATH_ROOT + 'MetFlow_({})_hist2d_gamma_{}_{}_{}_{}_{}_{}.png'.format(repetitions - 1, gamma, args.data,
    #                                                                              args.K,
    #                                                                              args.step_conditioning,
    #                                                                              args.noise_aggregation,
    #                                                                              args.use_barker),
    #         dpi=400, format='png')
    #     plt.close()
    #     ###### RNVP ########
    #     h, _, _ = np.histogram2d(z_rnvp_samples[0][:, 0], z_rnvp_samples[0][:, 1],
    #                              bins=bins, density=True, range=[[-x_lim, x_lim], [-y_lim, y_lim]])
    #     plt.imshow(h.T, interpolation='lanczos', norm=norm)
    #     plt.axis('off')
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     plt.savefig(PATH_ROOT + 'RNVP_1_hist2d_gamma_{}_{}_{}_{}_{}_{}.png'.format(gamma, args.data, args.K,
    #                                                                                args.step_conditioning,
    #                                                                                args.noise_aggregation,
    #                                                                                args.use_barker), dpi=400,
    #                 format='png')
    #     plt.close()
    #     h, _, _ = np.histogram2d(z_rnvp_samples[1][:, 0], z_rnvp_samples[1][:, 1],
    #                              bins=bins, density=True, range=[[-x_lim, x_lim], [-y_lim, y_lim]])
    #     plt.imshow(h.T, interpolation='lanczos', norm=norm)
    #     plt.axis('off')
    #     plt.axis('equal')
    #     plt.tight_layout()
    #     plt.savefig(PATH_ROOT + 'RNVP_2_hist2d_gamma_{}_{}_{}_{}_{}_{}.png'.format(gamma, args.data, args.K,
    #                                                                                args.step_conditioning,
    #                                                                                args.noise_aggregation,
    #                                                                                args.use_barker), dpi=400,
    #                 format='png')
    #     plt.close()

    ####### Plotting supplementary #######

    init_samples_prior = std_normal.sample((NUM_SAMPLES_SAMPLING, args.z_dim))
    cmap = 'jet'

    ######## S1 ########
    plt.scatter(init_samples_prior.cpu().detach().numpy()[:, 0],
                init_samples_prior.cpu().detach().numpy()[:, 1], cmap=cmap,
                c=init_samples_prior[:, 0].cpu().detach().numpy())
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("./pics/supplementary/prior.png", format='png', dpi=400)
    plt.close()

    previous_samples = init_samples_prior

    with torch.no_grad():
        for k in range(args.K):
            if args.step_conditioning == 'free':
                cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]

            if args.step_conditioning is None:
                current_samples, _, _, _, _ = transitions[k].make_transition(z_old=previous_samples,
                                                                             k=cond_vectors[k],
                                                                             target_distr=target, x=args["true_x"])
            else:
                current_samples, _, _, _, _ = transitions.make_transition(z_old=previous_samples, k=cond_vectors[k],
                                                                          target_distr=target, x=args["true_x"])
            plt.scatter(current_samples[:, 0].cpu().detach().numpy(),
                        current_samples[:, 1].cpu().detach().numpy(),
                        c=previous_samples[:, 0].cpu().detach().numpy(), cmap=cmap,
                        s=0.5)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig("./pics/supplementary/S1_{}_{}_{}_{}_{}_{}.png".format(args.step_conditioning, k,
                                                                               args.noise_aggregation, args.K,
                                                                               args.data, args.use_barker),
                        format='png', dpi=400)
            plt.close()
            previous_samples = current_samples

    ######## S2 ########

    previous_samples = init_samples_prior

    with torch.no_grad():
        for k in range(args.K):
            z_upd, _ = rnvp_models[0][k]._forward_step(previous_samples)
            current_samples = z_upd
            plt.scatter(current_samples[:, 0].cpu().detach().numpy(),
                        current_samples[:, 1].cpu().detach().numpy(),
                        c=init_samples_prior[:, 0].cpu().detach().numpy(), cmap=cmap,
                        s=0.5)
            plt.axis('equal')
            plt.tight_layout()
            plt.tight_layout()
            plt.savefig("./pics/supplementary/S2_{}_{}_{}_{}_{}_{}.png".format(args.step_conditioning, k,
                                                                               args.noise_aggregation, args.K,
                                                                               args.data, args.use_barker),
                        format='png', dpi=400)
            plt.close()
            previous_samples = current_samples

    ######## S3 ########
    init_samples_subst = GMM_target(kwargs=args, device=args.device).get_samples(NUM_SAMPLES_SAMPLING)
    plt.scatter(init_samples_subst.cpu().detach().numpy()[:, 0],
                init_samples_subst.cpu().detach().numpy()[:, 1], cmap=cmap,
                c=init_samples_subst[:, 0].cpu().detach().numpy())
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("./pics/supplementary/prior_subst.png", format='png', dpi=400)
    plt.close()

    previous_samples = init_samples_subst
    current_samples = previous_samples
    with torch.no_grad():
        for rep in range(NUM_REPETITIONS + 1):
            for k in range(args.K):
                if args.step_conditioning == 'free':
                    cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for k in range(K)]

                if args.step_conditioning is None:
                    current_samples, _, _, _, _ = transitions[k].make_transition(z_old=current_samples,
                                                                                 k=cond_vectors[k],
                                                                                 target_distr=target,
                                                                                 x=args["true_x"])
                else:
                    current_samples, _, _, _, _ = transitions.make_transition(z_old=current_samples,
                                                                              k=cond_vectors[k],
                                                                              target_distr=target, x=args["true_x"])
            if rep % 25 == 0:
                plt.scatter(current_samples[:, 0].cpu().detach().numpy(),
                            current_samples[:, 1].cpu().detach().numpy(),
                            #  c=init_samples_subst[:, 0].cpu().detach().numpy(),
                            cmap=cmap,
                            s=0.5)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig("./pics/supplementary/S3_{}_{}_{}_{}_{}_{}.png".format(args.step_conditioning, rep,
                                                                                   args.noise_aggregation, args.K,
                                                                                   args.data, args.use_barker),
                            format='png')
                plt.close()
                previous_samples = current_samples

    ######## S4 ########

    previous_samples = init_samples_subst
    current_samples = previous_samples
    with torch.no_grad():
        for rep in range(NUM_REPETITIONS + 1):
            for k in range(args.K):
                current_samples, _ = rnvp_models[0][k]._forward_step(current_samples)
            if rep % 25 == 0:
                plt.scatter(current_samples[:, 0].cpu().detach().numpy(),
                            current_samples[:, 1].cpu().detach().numpy(),
                            # c=init_samples_subst[:, 0].cpu().detach().numpy(),
                            cmap=cmap,
                            s=0.5)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig("./pics/supplementary/S4_{}_{}_{}_{}_{}_{}.png".format(args.step_conditioning, rep,
                                                                                   args.noise_aggregation, args.K,
                                                                                   args.data, args.use_barker),
                            format='png')
                plt.close()
                previous_samples = current_samples


def plot_mixture_plots(args, target, transitions, cond_vectors, tol):
    # pdb.set_trace()
    current_tol = 0
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                            scale=torch.tensor(1., dtype=args.torchType, device=args.device))
    num_batches = args['num_batches']
    K = args.K
    input_dim = args['z_dim']
    hidden_dim = args['z_dim'] * 2
    hidden_units = 16

    if args.step_conditioning is None:
        num_nafs = len(transitions)
    else:
        num_nafs = 1

    naf = []
    for i in range(num_nafs):
        one_arn = AutoRegressiveNN(input_dim, [hidden_dim], param_dims=[hidden_units] * 3).to(args.device)
        one_naf = NeuralAutoregressive(one_arn, hidden_units=hidden_units)
        naf.append(one_naf)
    naf = nn.ModuleList(naf)
    optimizer = torch.optim.Adam(naf.parameters())

    best_naf = []
    for i in range(num_nafs):
        one_arn = AutoRegressiveNN(input_dim, [hidden_dim], param_dims=[hidden_units] * 3).to(args.device)
        one_naf = NeuralAutoregressive(one_arn, hidden_units=hidden_units)
        best_naf.append(one_naf)
    best_naf = nn.ModuleList(best_naf)

    print('\n')
    print('Now we are training an independent NAF')
    print('\n')

    best_kl = float("inf")
    fin_batch = 0
    iterator = tqdm(range(num_batches))
    for i in iterator:
        z = std_normal.sample((args.batch_size_train, args['z_dim']))
        log_jac = torch.zeros(z.shape[0], device=args.device, dtype=args.torchType)
        z_prev = z
        for k in range(num_nafs):
            z_new = naf[k](z_prev)
            log_jac += naf[k].log_abs_det_jacobian(z_prev, z_new)
            z_prev = z_new
        true_log_density = target.get_logdensity(z=z_new, x=args["true_x"])
        kl = torch.mean(std_normal.log_prob(z).sum(1) - log_jac - true_log_density)
        kl.backward()
        if (i % 1000 == 0):
            print('Current KL for NAF: {}'.format(kl.item()))
        optimizer.step()
        optimizer.zero_grad()
        if kl < best_kl:
            best_kl = kl
            current_tol = 0
            fin_batch = i
        else:
            current_tol += 1
            if current_tol >= tol:
                best_naf.load_state_dict(naf.state_dict())  # copy weights and stuff
                print('NAF early stopping on batch ', fin_batch)
                iterator.close()
                break

    naf = best_naf
    n_samples = NUM_DIGITS_TO_PLOT
    repetitions = NUM_REPETITIONS
    z = std_normal.sample((n_samples, args.z_dim))

    with torch.no_grad():
        for rep in range(repetitions):
            for k in range(num_nafs):
                if args.step_conditioning == 'free':
                    cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for _ in range(K)]
                if args.step_conditioning is None:
                    z, log_jac, current_log_alphas, \
                    current_log_probs, directions = transitions[k].make_transition(z_old=z, k=cond_vectors[k],
                                                                                   target_distr=target,
                                                                                   x=args["true_x"])
                else:
                    z, log_jac, current_log_alphas, \
                    current_log_probs, directions = transitions.make_transition(z_old=z, k=cond_vectors[k],
                                                                                target_distr=target,
                                                                                x=args["true_x"])
                if rep == 0:
                    MNF_samples = z
    MNF_samples_repeated = z

    ### After one pushforward
    for k in range(NUM_PICS):
        plot_digit_samples(samples=get_samples(target.decoder, random_code=MNF_samples[64 * k: 64 * (k + 1)]),
                           args=args,
                           method='MetFlow_{}'.format(k))

    z = std_normal.sample((n_samples, args['z_dim']))
    for k in range(NUM_PICS):
        z = naf[k](z)
    samples_naf = z

    for k in range(NUM_PICS):
        plot_digit_samples(samples=get_samples(target.decoder, random_code=samples_naf[64 * k: 64 * (k + 1)]),
                           args=args,
                           method='Independant_NAF_{}'.format(k))

    ### After many pushforwards
    for k in range(NUM_PICS):
        plot_digit_samples(
            samples=get_samples(target.decoder, random_code=MNF_samples_repeated[64 * k: 64 * (k + 1)]), args=args,
            method='MetFlow_repeated_{}'.format(k))


def plot_gibbs(encoder, target, metflow, flow, args, dataset, cond_vectors=None):
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                            scale=torch.tensor(1., dtype=args.torchType, device=args.device))
    K = args.K
    for batch in dataset.next_train_batch():  # cycle over batches
        batch_train = batch
        break
    indices = np.random.randint(low=0, high=batch_train.shape[0], size=5)
    for ind in indices:
        fixed_image = batch_train[ind][None, ...]

        top_row = 14
        top_mask = torch.tensor([[1] * 28 * top_row + [0] * 28 * (28 - top_row)], device=args.device).view(28, 28)
        bottom_mask = 1. - top_mask

        # initial_image = torch.distributions.Bernoulli(0.5).sample((1, 1, 28, 28)).to(device) * top_mask + fixed_image * bottom_mask
        initial_image = fixed_image * top_mask + fixed_image * bottom_mask

        num_pics = 16
        repetitions = 30

        fig, ax = plt.subplots(nrows=3, ncols=num_pics, dpi=400, figsize=(21, 6))
        ax[0, num_pics // 2].set_title('MetFlow')
        ax[1, num_pics // 2].set_title('Encoder')
        ax[2, num_pics // 2].set_title('Encoder and NAF')
        ax[0, 0].set_title('Initial sample', size=20)
        ax[1, 0].set_title('Initial sample', size=20)
        ax[2, 0].set_title('Initial sample', size=20)

        current_image_enc_our_dec = initial_image
        current_image_enc_dec = initial_image
        current_image_enc_flow_dec = initial_image

        for i in range(num_pics):
            ax[0, i].imshow((1. - current_image_enc_our_dec[0]).permute(1, 2, 0)[:, :, 0].cpu(), 'gray')
            ax[0, i].axis('off')
            ax[1, i].imshow((1. - current_image_enc_dec[0]).permute(1, 2, 0)[:, :, 0].cpu(), 'gray')
            ax[1, i].axis('off')
            ax[2, i].imshow((1. - current_image_enc_flow_dec[0]).permute(1, 2, 0)[:, :, 0].cpu(), 'gray')
            ax[2, i].axis('off')

            ####### ENCODER + OUR APPROACH + DECODER ######
            for r in range(repetitions):
                mu, sigma = encoder(current_image_enc_our_dec)
                u = std_normal.sample(mu.shape)
                z = mu + sigma * u
                for k in range(K):
                    if args.step_conditioning == 'free':
                        cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for k in range(K)]
                    if args.step_conditioning is None:
                        z, _, _, _, _ = metflow[k].make_transition(z_old=z, k=cond_vectors[k],
                                                                   target_distr=target, x=current_image_enc_our_dec)
                    else:
                        z, _, _, _, _ = metflow.make_transition(z_old=z, k=cond_vectors[k],
                                                                target_distr=target, x=current_image_enc_our_dec)
                bernoulli_params = nn.Sigmoid()(target.decoder(z)[0][0])
                generated_pic = (torch.distributions.Bernoulli(probs=bernoulli_params).sample()).view(1, 28, 28)
                current_image_enc_our_dec = generated_pic * top_mask + fixed_image * bottom_mask

            ####### ENCODER + DECODER #######
            for r in range(repetitions):
                mu, sigma = encoder(current_image_enc_dec)
                u = std_normal.sample(mu.shape)
                z = mu + sigma * u
                bernoulli_params = nn.Sigmoid()(target.decoder(z)[0][0])
                generated_pic = (torch.distributions.Bernoulli(probs=bernoulli_params).sample()).view(1, 28, 28)
                current_image_enc_dec = generated_pic * top_mask + fixed_image * bottom_mask

            ####### ENCODER + FLOW + DECODER #######
            for r in range(repetitions):
                mu, sigma = encoder(current_image_enc_flow_dec)
                u = std_normal.sample(mu.shape)
                z = flow(mu + sigma * u)
                bernoulli_params = nn.Sigmoid()(target.decoder(z)[0][0])
                generated_pic = (torch.distributions.Bernoulli(probs=bernoulli_params).sample()).view(1, 28, 28)
                current_image_enc_flow_dec = generated_pic * top_mask + fixed_image * bottom_mask

        plt.tight_layout()
        plt.subplots_adjust(left=0.025, right=1., bottom=0., top=0.94, wspace=0.1, hspace=0.25)
        if not os.path.exists('./pics/{}/'.format(args.problem)):
            os.makedirs('./pics/{}/'.format(args.problem))
        saving_path = './pics/{}/{}_{}_{}_{}_{}_{}.pdf'.format(args.problem, args.problem, ind, args.K,
                                                               args.step_conditioning, args.noise_aggregation,
                                                               args.use_barker)
        plt.savefig(saving_path, format='pdf')
        print('\n')
        print('Picture was saved: {}'.format(saving_path))


def plot_densities(args, models_for_experiments_rnvp, models_for_experiments_ours, cond_vectors_ours=None):
    print('\n')
    print('Now we are plotting densities from Rezende, Mohamed 2015')
    print('\n')

    def d1(z):
        z_norm = torch.norm(z, 2, 1)
        add1 = 0.5 * ((z_norm - 2) / 0.4) ** 2
        add2 = - torch.log(torch.exp(-0.5 * ((z[:, 0] - 2) / 0.6) ** 2) + \
                           torch.exp(-0.5 * ((z[:, 0] + 2) / 0.6) ** 2) + 1e-9)
        return add1 + add2

    def d2(z):
        w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
        return 0.5 * ((z[:, 1] - w1) / 0.4) ** 2

    def d3(z):
        w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
        w2 = 3 * torch.exp(-0.5 * ((z[:, 0] - 1) / 0.6) ** 2)
        in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.35) ** 2)
        in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w2) / 0.35) ** 2)
        return -torch.log(in1 + in2 + 1e-9)

    def d4(z):
        w1 = torch.sin(2 * np.pi * z[:, 0] / 4)
        w3 = 3 * torch.sigmoid((z[:, 0] - 1) / 0.3)
        in1 = torch.exp(-0.5 * ((z[:, 1] - w1) / 0.4) ** 2)
        in2 = torch.exp(-0.5 * ((z[:, 1] - w1 + w3) / 0.35) ** 2)
        return -torch.log(in1 + in2 + 1e-9)

    def target_energy(cur_dat):
        if cur_dat == 't1':
            return d1
        elif cur_dat == 't2':
            return d2
        elif cur_dat == 't3':
            return d3
        elif cur_dat == 't4':
            return d4

    target_samples = []
    rnvp_samples = []
    our_samples = []
    our_samples_repeated = []

    ##### Plotting the big picture for Rezende and Mohamed 2015 #####
    plt.close()

    repetitions = NUM_REPETITIONS
    increment = NUM_REPETITIONS - 1
    bins = 200
    limit = 4
    num_samples = 5000 #000
    K = args.K

    npts = int(num_samples ** 0.5)
    xside = np.linspace(-limit, limit, npts)
    yside = np.linspace(-limit, limit, npts)
    xx, yy = np.meshgrid(xside, yside)

    for i, cur_dat in tqdm(enumerate(['t1', 't2', 't3', 't4'])):
        args['bnaf_data'] = cur_dat
        target = BNAF_examples(kwargs=args, device=args.device)
        targ_log_dens = target_energy(cur_dat)

        z = torch.tensor(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]))
        u = targ_log_dens(z)
        p = np.exp(-u).reshape(npts, npts)
        target_samples.append(p)
        std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                                scale=torch.tensor(1., dtype=args.torchType, device=args.device))

        u = std_normal.sample((num_samples, 2))
        z_rnvp = u
        with torch.no_grad():
            for k in range(K):
                z_upd, _ = models_for_experiments_rnvp[i][k]._forward_step(z_rnvp)
                z_rnvp = z_upd
            a = z_rnvp.cpu().detach().numpy()
            rnvp_samples.append(a)

        u = std_normal.sample((num_samples, 2))
        z = u
        with torch.no_grad():
            for rep in range(repetitions):
                for k in range(K):
                    if args.step_conditioning == 'free':
                        cond_vectors = [[std_normal.sample((args.z_dim,)), args.noise_aggregation] for k in
                                        range(K)]
                    else:
                        cond_vectors = cond_vectors_ours[i]
                    if args.step_conditioning is None:
                        z, _, _, _, _ = models_for_experiments_ours[i][k].make_transition(z_old=z, k=
                        cond_vectors[k], target_distr=target)
                    else:
                        z, _, _, _, _ = models_for_experiments_ours[i].make_transition(z_old=z, k=
                        cond_vectors[k], target_distr=target)
                if rep % increment == 0:
                    a = z.cpu().data.numpy()
                    if rep == 0:
                        our_samples.append(a)
                    else:
                        our_samples_repeated.append(a)

    stacked_target_samples = [[], [], [], []]
    stacked_rnvp_samples = [[], [], [], []]
    stacked_our_samples = [[], [], [], []]
    stacked_our_repeated_samples = [[], [], [], []]
    for i in range(len(target_samples)):
        stacked_target_samples[i % 4].append(target_samples[i])
        stacked_rnvp_samples[i % 4].append(rnvp_samples[i])
        stacked_our_samples[i % 4].append(our_samples[i])
        stacked_our_repeated_samples[i % 4].append(our_samples_repeated[i])

    fig, ax = plt.subplots(nrows=len(models_for_experiments_ours),
                           ncols=2 + repetitions // increment + 1, figsize=(8, 7), dpi=400)

    interpolation = 'gaussian'

    ax[0, 0].set_title('Target')
    ax[0, 1].set_title('RNVP')
    ax[0, 2].set_title('MetFlow')
    ax[0, 3].set_title('MetFlow ({})'.format(repetitions - 1))
    for i in range(4):
        ax[i, 0].pcolormesh(xx, yy, stacked_target_samples[i][0])
        ax[i, 0].invert_yaxis()
        ax[i, 0].get_xaxis().set_ticks([])
        ax[i, 0].get_yaxis().set_ticks([])
        ax[i, 0].axis('off')

        h, _, _ = np.histogram2d(np.concatenate(stacked_rnvp_samples[i])[:, 0],
                                 np.concatenate(stacked_rnvp_samples[i])[:, 1],
                                 bins=bins, density=True, range=[[-limit, limit], [-limit, limit]])
        ax[i, 1].imshow(h.T, interpolation=interpolation)
        ax[i, 1].axis('off')

        h, _, _ = np.histogram2d(np.concatenate(stacked_our_samples[i])[:, 0],
                                 np.concatenate(stacked_our_samples[i])[:, 1],
                                 bins=bins, density=True, range=[[-limit, limit], [-limit, limit]])
        ax[i, 2].imshow(h.T, interpolation=interpolation)
        ax[i, 2].axis('off')

        h, _, _ = np.histogram2d(np.concatenate(stacked_our_repeated_samples[i])[:, 0],
                                 np.concatenate(stacked_our_repeated_samples[i])[:, 1],
                                 bins=bins, density=True, range=[[-limit, limit], [-limit, limit]])
        ax[i, 3].imshow(h.T, interpolation=interpolation)
        ax[i, 3].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(left=0., right=1., bottom=0., top=0.95, wspace=0.01, hspace=0.01)
    if not os.path.exists('./pics/{}/'.format(args.problem)):
        os.makedirs('./pics/{}/'.format(args.problem))
    saving_path = './pics/{}/{}_{}_{}_{}.png'.format(args.problem, args.problem, args.K, args.step_conditioning,
                                                     args.use_barker)
    plt.savefig(saving_path, format='png')
    print('Picture was saved: {}'.format(saving_path))
    plt.close()
