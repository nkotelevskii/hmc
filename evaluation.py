import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
from tqdm import tqdm
from kernels import HMC_vanilla
import logging
 
# add filemode="w" to overwrite
logging.basicConfig(filename="./logging.txt", level=logging.INFO)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
class Encoder_vae(nn.Module):
    def __init__(self, data_dim, z_dim, K=8):
        super(Encoder_vae, self).__init__()
        self.h1 = nn.Linear(in_features=data_dim, out_features=K)
        self.h2 = nn.Linear(in_features=K, out_features=K)
        self.mu = nn.Linear(in_features=K, out_features=z_dim)
        self.sigma = nn.Linear(in_features=K, out_features=z_dim)
    def forward(self, x):
        h1 = F.selu(self.h1(x))
        h2 = F.selu(self.h2(h1))
        return self.mu(h2), F.softplus(self.sigma(h2))
    
class Decoder(nn.Module):
    def __init__(self, data_dim, z_dim):
        super(Decoder, self).__init__()
        self.W = nn.Linear(in_features=z_dim, out_features=data_dim, bias=False)
    def forward(self, z):
        return self.W(z)

class Target(nn.Module):
    def __init__(self, dec, args):
        super(Target, self).__init__()
        self.decoder = dec
        self.std_normal = args.std_normal
    def get_logdensity(self, z, x, prior=None, args=None, prior_flow=None):
        probs = torch.sigmoid(self.decoder(z))
        return torch.distributions.Bernoulli(probs=probs).log_prob(x).sum(1) + self.std_normal.log_prob(z).sum(1)



def evaluation(encoder, decoder, data):
    dataloader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=False)

    args = dotdict({})
    args.device = data.device
    args.torchType = torch.float32
    args.std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                                scale=torch.tensor(1., dtype=args.torchType, device=args.device))
    n_IS = 10000

    print('Method 1: Overdispersed encoder is running')
    means = []
    stds = []
    for n in tqdm(range(3)):
        nll = []
        for batch_raw in dataloader:
            batch = batch_raw.repeat(n_IS, 1)
            mu, sigma_ = encoder(batch)
            sigma = 1.2 * sigma_

            u = args.std_normal.sample(mu.shape)
            z = mu + sigma * u

            probs = torch.sigmoid(decoder(z))
            log_p = torch.distributions.Bernoulli(probs=probs).log_prob(batch).sum(1)
            KLD = (-0.5 * (1 + 2 * sigma.log() - mu.pow(2) - sigma.pow(2))).sum(1)
            nll_lse = torch.logsumexp((log_p - KLD).view([n_IS, -1]), 0)
            nll_current = -np.log(n_IS) + torch.mean(nll_lse)
            nll.append(nll_current.cpu().detach().numpy())
        means.append(np.array(nll).mean())
        stds.append(np.array(nll).std())
    print('Method 1: Overdisperced encoder:', np.array(means).mean(), '+-', np.array(stds).mean())
    print('\n')
    print('-' * 100)
    print('\n')
    
    logging.info('Method 1: Overdisperced encoder: {} +- {}'.format(np.array(means).mean(), np.array(stds).mean()))
    logging.info('-' * 100)

    print('Method 2: Overdispersed encoder with HMC is running')

    args.N = 2
    args.alpha = 0.5
    args.gamma = 0.1
    args.use_partialref = False
    args.use_barker = False
    transitions = HMC_vanilla(args)

    n_warmup = 300
    n_samples = 300

    target = Target(decoder, args)

    means = []
    stds = []
    for n in tqdm(range(3)):
        nll = []
        for batch_raw in dataloader:
            batch = batch_raw.repeat(n_IS, 1)
            mu_, sigma_ = encoder(batch)
            
            u = args.std_normal.sample(mu_.shape)
            z_  = mu_ + sigma_ * u

            momentum = args.std_normal.sample(z_.shape)
            for i in range(n_warmup):
                z_, momentum, _, _, _, _ = transitions.make_transition(z_, momentum, target, x=batch)

            samples = torch.tensor([], device=args.device, dtype=args.torchType)
            
            for i in range(n_samples):
                z_, momentum, _, _, _, _ = transitions.make_transition(z_, momentum, target, x=batch)
                samples = torch.cat([samples, z_[None]], 0)
                            
            sigma = 1.2 * sigma_
            mu = samples.mean(0)
            
            z = mu + sigma * std_normal.sample(mu.shape)

            probs = torch.sigmoid(decoder(z))
            log_p = torch.distributions.Bernoulli(probs=probs).log_prob(batch).sum(1)
            KLD = (-0.5 * (1 + 2 * sigma.log() - mu.pow(2) - sigma.pow(2))).sum(1)
            nll_lse = torch.logsumexp((log_p - KLD).view([n_IS, -1]), 0)
            nll_current = -np.log(n_IS) + torch.mean(nll_lse)
            nll.append(nll_current.cpu().detach().numpy())
        means.append(np.array(nll).mean())
        stds.append(np.array(nll).std())
    print('Method 2: Overdispersed encoder with HMC', np.array(means).mean(), '+-', np.array(stds).mean())
    print('\n')
    print('-' * 100)
    print('\n')
    
    logging.info('Method 2: Overdispersed encoder with HMC: {} +- {}'.format(np.array(means).mean(), np.array(stds).mean()))
    logging.info('-' * 100)


    print('Method 3: Overdispersed variance of final HMC samples is running')

    target = Target(decoder, args)

    means = []
    stds = []
    for n in tqdm(range(3)):
        nll = []
        for batch_raw in dataloader:
            batch = batch_raw.repeat(n_IS, 1)
            mu_, sigma_ = encoder(batch)
            
            u = args.std_normal.sample(mu_.shape)
            z_  = mu_ + sigma_ * u

            momentum = args.std_normal.sample(z_.shape)
            for i in range(n_warmup):
                z_, momentum, _, _, _, _ = transitions.make_transition(z_, momentum, target, x=batch)

            samples = torch.tensor([], device=args.device, dtype=args.torchType)
            for i in range(n_samples):
                z_, momentum, _, _, _, _ = transitions.make_transition(z_, momentum, target, x=batch)
                samples = torch.cat([samples, z_[None]], 0)

            mu = samples.mean(0)
            sigma = samples.std(0) * 1.2
            
            z = mu + sigma * std_normal.sample(mu.shape)

            probs = torch.sigmoid(decoder(z))
            log_p = torch.distributions.Bernoulli(probs=probs).log_prob(batch).sum(1)
            KLD = (-0.5 * (1 + 2 * sigma.log() - mu.pow(2) - sigma.pow(2))).sum(1)
            nll_lse = torch.logsumexp((log_p - KLD).view([n_IS, -1]), 0)
            nll_current = -np.log(n_IS) + torch.mean(nll_lse)
            nll.append(nll_current.cpu().detach().numpy())
        means.append(np.array(nll).mean())
        stds.append(np.array(nll).std())
    print('Method 3: Overdispersed variance of final HMC samples', np.array(means).mean(), '+-', np.array(stds).mean())
    
    logging.info('Method 3: Overdispersed variance of final HMC samples: {} +- {}'.format(np.array(means).mean(), np.array(stds).mean()))
    logging.info('-' * 100)
    
    print('\n')
    print('-' * 100)
    print('\n')
    
    print('Finish!')


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    true_theta = torch.tensor([[-0.2963,  2.6764, -0.1408, -0.8441,  0.2905, -0.2838, -1.4535,  2.3737,
         -0.0177, -2.7884],
        [-0.3788,  0.7046, -1.3956, -0.1248, -0.9259, -1.5463, -0.4902,  0.0244,
         -1.5992, -0.8469]], dtype=torch.float32, device=device)
    std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=torch.float32, device=device),
                                                scale=torch.tensor(1., dtype=torch.float32, device=device))
    data = torch.distributions.Bernoulli(probs=torch.sigmoid(std_normal.sample((5000, 2)) @ true_theta)).sample()


    names = [
        ('./models_to_eval/encoder_metflow_loss.pth', './models_to_eval/decoder_metflow_loss.pth'),
        ('./models_to_eval/encoder_hoffman_loss.pth', './models_to_eval/decoder_hoffman_loss.pth'),
        ('./models_to_eval/encoder_hoffman_metflow_loss.pth', './models_to_eval/decoder_hoffman_metflow_loss.pth'),
        ('./models_to_eval/encoder_lagging.pth', './models_to_eval/decoder_lagging.pth'),
    ]
    for name in names:
        print('-' * 100)
        print(name[0])
        print(name[1])
        logging.info('-' * 100)
        logging.info('-' * 100)
        logging.info(name[0])
        logging.info(name[1])
        logging.info('-' * 100)
        logging.info('-' * 100)
        encoder = torch.load(name[0], map_location=device)
        decoder = torch.load(name[1], map_location=device)
        if 'decoder' in decoder._modules:
            decoder = decoder.decoder
        for p in encoder.parameters():
            p.requires_grad_(False)
        for p in decoder.parameters():
            p.requires_grad_(False)
        evaluation(encoder, decoder, data)
        print('\n')
        print('-' * 100)
        print('-' * 100)
        print('-' * 100)
        print('\n')
        
