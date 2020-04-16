
import torch
import torch.nn as nn
import torch.nn.functional as F

torchType = torch.float32

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class Inf_network(nn.Module):
    def __init__(self, kwargs):
        super(Inf_network, self).__init__()
        args = kwargs
        self.z_dim = args.z_dim
        self.size_h = 28
        self.size_w = 28
        self.size_c = 1
        
        self.conv1 = nn.Conv2d(in_channels=self.size_c, out_channels=16, kernel_size=5,
                               stride=2, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5,
                               stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5,
                               stride=2, padding=2)
        self.linear = nn.Linear(in_features=512, out_features=450)
        self.mu = nn.Linear(in_features=450, out_features=self.z_dim)
        self.sigma = nn.Linear(in_features=450, out_features=self.z_dim)
        self.h = nn.Linear(in_features=450, out_features=self.z_dim)

    def forward(self, x):
        h1 = F.softplus(self.conv1(x))
        h2 = F.softplus(self.conv2(h1))
        h3 = F.softplus(self.conv3(h2))
        h3_flat = h3.view(h3.shape[0], -1)
        h4 = F.softplus(self.linear(h3_flat))
        mu = self.mu(h4)
        sigma = F.softplus(self.sigma(h4))
        h = self.h(h4)
        return mu, sigma,h


class Gen_network(nn.Module):
    def __init__(self, z_dim, args):
        super(Gen_network, self).__init__()
        self.z_dim = z_dim
        self.linear1 = nn.Linear(in_features=self.z_dim, out_features=450)
        self.linear2 = nn.Linear(in_features=450, out_features=512)
        self.size_h = 28
        self.size_w = 28
        self.size_c = 1
        self.use_batchnorm = args.use_batchnorm
        if self.use_batchnorm:
            self.bn = nn.BatchNorm1d(450)
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(16)
        else:
            self.bn = Identity()
            self.bn1 = Identity()
            self.bn2 = Identity()

        if args.decoder == "deconv":
            self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=5,
                                              stride=2, padding=2)
            self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5,
                                              stride=2, padding=2, output_padding=1)
            self.deconv3 = nn.ConvTranspose2d(in_channels=16, out_channels=self.size_c, kernel_size=5,
                                              stride=2, padding=2, output_padding=1)
        elif args.decoder == "bilinear":
            self.deconv1 = nn.Sequential(nn.UpsamplingBilinear2d(size=(7, 7)),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1))
            self.deconv2 = nn.Sequential(nn.UpsamplingBilinear2d(size=(14, 14)),
                                         nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1))
            self.deconv3 = nn.Sequential(nn.UpsamplingBilinear2d(size=(self.size_h, self.size_w)),
                                         nn.Conv2d(in_channels=16, out_channels=self.size_c, kernel_size=1))

    def forward(self, x):
        h1 = F.softplus(self.bn(self.linear1(x)))
        h2_flatten = F.softplus(self.linear2(h1))
        h2 = h2_flatten.view(-1, 32, 4, 4)
        h3 = F.softplus(self.bn1(self.deconv1(h2)))
        h4 = F.softplus(self.bn2(self.deconv2(h3)))
        bernoulli_logits = self.deconv3(h4)
        return [bernoulli_logits, None]


class Inf_network_simple(nn.Module):
    def __init__(self, kwargs):
        super(Inf_network_simple, self).__init__()
        args = kwargs
        self.z_dim = args.z_dim
        self.input_dim = args.data_dim

        self.linear = nn.Linear(in_features=self.input_dim, out_features=20 * args.z_dim)
        self.mu = nn.Linear(in_features=20 * args.z_dim, out_features=self.z_dim)
        self.sigma = nn.Linear(in_features=20 * args.z_dim, out_features=self.z_dim)

    def forward(self, x):
        h4 = F.softplus(self.linear(x))
        mu = self.mu(h4)
        sigma = F.softplus(self.sigma(h4))
        return mu, sigma


class Gen_network_simple(nn.Module):
    def __init__(self, z_dim, kwargs):
        super(Gen_network_simple, self).__init__()
        args = kwargs
        self.z_dim = args.z_dim
        self.output_dim = args.data_dim

        self.linear = nn.Linear(in_features=self.z_dim, out_features=5 * args.z_dim)
        self.mu = nn.Linear(in_features=5 * args.z_dim, out_features=self.output_dim)
        self.sigma = nn.Linear(in_features=5 * args.z_dim, out_features=self.output_dim)

    def forward(self, x):
        h4 = F.softplus(self.linear(x))
        mu = self.mu(h4)
        sigma = F.softplus(self.sigma(h4))
        return mu, sigma



