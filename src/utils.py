import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_args(args, torchType):
    print('\nParameters: \n', args, '\n')
#     args = dotdict(vars(args))
    args = dotdict(args)

    device = args.gpu
    device = ('cpu' if device == -1 else 'cuda:{}'.format(args.gpu))

    args['device'] = device
    args['torchType'] = torchType
    args['std_normal'] = torch.distributions.Normal(loc=torch.tensor(0., device=device, dtype=torchType),
                                                   scale=torch.tensor(1., device=device, dtype=torchType))
    args['use_barker'] = True if args['use_barker'] == 'True' else False
    args['plot_all_pics'] = True if args['plot_all_pics'] == 'True' else False
    if args.step_conditioning == 'None':
        args.step_conditioning = None
    ############################################## RNVP parameters ##############################################
    # pdb.set_trace()
    args['hidden_dim'] = 2 * args.z_dim
    args['masks'] = np.array([[i % 2 for i in range(args['z_dim'])],
                              [(i + 1) % 2 for i in range(args['z_dim'])]]).astype(np.float32)

    args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']), nn.Tanh())
    args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'], args['hidden_dim']), nn.LeakyReLU(),
                                         nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                         nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))

    # if we are using stacking, we multiply input layers by 2
    if args['noise_aggregation'] == 'stacking' and args['step_conditioning']:
        args['nets'] = lambda: nn.Sequential(nn.Linear(args['z_dim'] * 2, args['hidden_dim']), nn.LeakyReLU(),
                                             nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                             nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']), nn.Tanh())
        args['nett'] = lambda: nn.Sequential(nn.Linear(args['z_dim'] * 2, args['hidden_dim']), nn.LeakyReLU(),
                                             nn.Linear(args['hidden_dim'], args['hidden_dim']),
                                             nn.LeakyReLU(), nn.Linear(args['hidden_dim'], args['z_dim']))

    # if we are using step conditioning, we use more elementary rnvp's
    if args['step_conditioning']:
        args['masks'] = np.array([[i % 2 for i in range(args['z_dim'])],
                                  [(i + 1) % 2 for i in range(args['z_dim'])]] * 3).astype(np.float32)

    ############################################ Target distribution ############################################
    # Multivariate normal
    args['loc'] = torch.tensor([0., 0.], dtype=torchType, device=device)  # loc for Gaussian target
    args['cov_matrix'] = torch.tensor([[1., 0.7], [0.7, 1.]], dtype=torchType,
                                      device=device)  # cov_matrix for Gaussian target
    args['true_mean'] = torch.zeros(args['z_dim'], device=device, dtype=torchType)

    # GMM (two gaussians)
    args['p_first_gaussian'] = 0.5  # Probability (weight) of the first gaussian
    gaussian_centers = [-50., 50.]
    args['locs_single_gmm'] = [torch.tensor([gaussian_centers[0], 0.], dtype=torch.float32, device=device),
                               torch.tensor([gaussian_centers[1], 0.], dtype=torch.float32, device=device)]  # locs
    args['covs_single_gmm'] = [torch.eye(2, dtype=torch.float32, device=device),
                               torch.eye(2, dtype=torch.float32, device=device)]  # covariances

    # Banana
    args['banana_cov_matrix'] = torch.tensor([[1., .9], [.9, 1.]], dtype=torch.float32, device=device)  # covariance
    args['banana_a'] = 1.15
    args['banana_b'] = 0.5

    # Rough Well
    args['rw_eps'] = 1e-2
    args['rw_easy'] = True

    # Examples from BNAF paper
    args['bnaf_data'] = 't4'  # t1, t2, t3, t4

    # GMM with arbitraty many components
    args['num_gauss'] = 8
    args['p_gaussians'] = [torch.tensor(1. / args['num_gauss'], device=device, dtype=torchType)] * args['num_gauss']
    args['locs'] = [torch.tensor([0., 10.], dtype=torch.float32, device=device),
                    torch.tensor([7., 7.], dtype=torch.float32, device=device),
                    torch.tensor([10., 0.], dtype=torch.float32, device=device),
                    torch.tensor([7., -7.], dtype=torch.float32, device=device),
                    torch.tensor([0., -10.], dtype=torch.float32, device=device),
                    torch.tensor([-7., -7.], dtype=torch.float32, device=device),
                    torch.tensor([-10., 0.], dtype=torch.float32, device=device),
                    torch.tensor([-7., 7.], dtype=torch.float32,
                                 device=device)]  # list of locations for each of these gaussians
    args['covs'] = [torch.eye(2, dtype=torch.float32, device=device)] * args[
        'num_gauss']  # list of covariance matrices for each of these gaussians

    return args


def plot_digit_samples(samples, args, epoch=None):
    """
    Plot samples from the generative network in a grid
    """

    grid_h = 8
    grid_w = 8
    data_h = 28
    data_w = 28
    data_c = 1

    # Turn the samples into one large image
    tiled_img = np.zeros((data_h * grid_h, data_w * grid_w))

    for idx, image in enumerate(samples):
        i = idx % grid_w
        j = idx // grid_w

        top = j * data_h
        bottom = (j + 1) * data_h
        left = i * data_w
        right = (i + 1) * data_w
        tiled_img[top:bottom, left:right] = image

    # Save the new image
    plt.close()
    plt.axis('off')

    plt.imshow(tiled_img, cmap='gray')

    if not os.path.exists('./pics/'):
        os.makedirs('./pics/')
    plt.tight_layout()
    img_path = './pics/mnist_epoch_{}_K_{}_N_{}_amortize_{}.png'.format(epoch, args.K, args.N, args.amortize)
    plt.savefig(img_path)
    print('Saved samples to {}'.format(img_path))
    plt.show()


def get_samples(gen_network, random_code):
    samples = nn.Sigmoid()(gen_network(random_code)[0]).view(random_code.shape[0], 28, 28)
    out = samples.cpu().detach().numpy()
    return out
