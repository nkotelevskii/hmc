import torch
import torch.nn as nn
import numpy as np
import pdb
from scipy.stats import truncnorm

def truncated_normal(size, std=1):
    values = truncnorm.rvs(-2.*std, 2.*std, size=size)
    return values

def make_linear_network(dims, encoder=False):
    layer_list = nn.ModuleList([])
    for i in range(len(dims) - 1):
        if i == len(dims) - 2 and encoder:
            layer_list.append(nn.Linear(dims[i], 2 * dims[i + 1]))
        else:
            layer_list.append(nn.Linear(dims[i], dims[i + 1]))
        layer_list[-1].weight = nn.init.xavier_uniform_(layer_list[-1].weight)
        layer_list[-1].bias = nn.Parameter(
            torch.tensor(truncated_normal(layer_list[-1].bias.shape, 0.001), dtype=torch.float32))
        layer_list.append(nn.Tanh())
    layer_list = layer_list[:-1]
    model = nn.Sequential(*layer_list)
    return model


class MultiDAE(nn.Module):
    def __init__(self, p_dims, q_dims=None):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims
        dims = self.q_dims + self.p_dims[1:]

        self.model = make_linear_network(dims)
        self.dropout = nn.Dropout()
        print(self.model)

    def forward(self, x_initial, is_training_ph=1.):
        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)
        logits = self.model(x)
        return logits, 0.


class MultiVAE(nn.Module):
    def __init__(self, p_dims, q_dims=None, device="cpu"):
        super(MultiVAE, self).__init__()

        self.p_dims = p_dims
        if q_dims is None:
            self.q_dims = p_dims[::-1]
            q_dims = self.q_dims
        else:
            assert q_dims[0] == p_dims[-1], "Input and output dimension must equal each other for autoencoders."
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q-network mismatches."
            self.q_dims = q_dims

        self.encoder = make_linear_network(q_dims, encoder=True)
        print(self.encoder)

        self.decoder = make_linear_network(p_dims)
        print(self.decoder)

        self.dropout = nn.Dropout()

        device_zero = torch.tensor(0., dtype=torch.float32, device=device)
        device_one = torch.tensor(1., dtype=torch.float32, device=device)
        self.std_normal = torch.distributions.Normal(loc=device_zero, scale=device_one)

    def forward(self, x_initial, is_training_ph=1.):
        l2 = torch.sum(x_initial ** 2, 1)[..., None]
        x_normed = x_initial / torch.sqrt(torch.max(l2, torch.ones_like(l2) * 1e-12))
        x = self.dropout(x_normed)

        enc_out = self.encoder(x)
        mu, logvar = enc_out[:, :self.q_dims[-1]], enc_out[:, self.q_dims[-1]:]
        std = torch.exp(0.5 * logvar)
        KL = torch.mean(torch.sum(0.5 * (-logvar + torch.exp(logvar) + mu ** 2 - 1), dim=1))

        u = self.std_normal.sample(mu.shape)
        z = mu + is_training_ph * u * std

        logits = self.decoder(z)

        return logits, KL