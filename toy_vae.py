from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt 
import random
# from ipdb import set_trace as st
import pdb

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()

# torch.manual_seed(args.seed)

device = torch.device("cuda:1")

K = 1024

def data_gen(BATCH_SIZE):
    #8 gaussians
    while 1:
        theta = (np.pi/4) * torch.randint(0, 8, (BATCH_SIZE,)).float().to(device)
        centers = torch.stack((torch.cos(theta), torch.sin(theta)), dim = -1)
        noise = torch.randn_like(centers) * 0.1
        yield centers + noise

test_loader = train_loader = data_gen(args.batch_size)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc0 = nn.Linear(2, K)
        self.fc1 = nn.Linear(K, K)
        self.fc21 = nn.Linear(K, K)
        self.fc22 = nn.Linear(K, K)
        self.fc3 = nn.Linear(K, K)
        self.fc4 = nn.Linear(K, K)
        self.fc5 = nn.Linear(K, 2)

    def encode(self, x):
        h1 = F.selu(self.fc1(F.selu(self.fc0(x))))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.fc5(F.selu(self.fc4(F.selu(self.fc3(z)))))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 2))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose = True, threshold = 1E-2, eps=1e-6)

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
#     #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 2), reduction='sum')
    L2 = torch.mean((recon_x-x)**2)
    
#     # see Appendix B from VAE paper:
#     # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
#     # https://arxiv.org/abs/1312.6114
#     # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
#     KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
#     return L2 + KLD
#     L2 = -torch.distributions.Normal(loc=recon_x, scale=torch.ones_like(recon_x)).log_prob(x).mean()
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return L2 + KLD


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        if batch_idx > 100:
            break #100 batches per epoch
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss /= 100
    scheduler.step(train_loss)
    print (train_loss)

if __name__ == "__main__":
#     pdb.set_trace()
    for epoch in range(1, args.epochs + 1):
        train(epoch)

    gt = next(train_loader)
        
    with torch.no_grad():
        sample = torch.randn(2048, K).to(device)
        out = model.decode(sample).cpu().numpy()
        recon = model(gt)[0].cpu().numpy()

    rx,ry = recon[:,0], recon[:,1]
        
    gt = gt.cpu().numpy()
    gx, gy = gt[:,0], gt[:,1]
        
    xs, ys = out[:,0], out[:,1]

    plt.scatter(gx, gy, c = 'red', s=3)
    plt.scatter(xs, ys, c = 'blue', s=3)
    plt.axes().set_aspect('equal')
    plt.show()
    plt.tight_layout()
    plt.savefig('./toy_vae.png', format='png', dpi=300)

#     st()