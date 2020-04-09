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
