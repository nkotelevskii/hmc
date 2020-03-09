import torch
import numpy as np
import torch.nn as nn
from pyro.distributions.transforms import NeuralAutoregressive
from pyro.nn import AutoRegressiveNN


def get_args():
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args = {}
    args = dotdict(args)

    args.device = device
    args.torchType = torch.float32
    
    args.data = 'mnist'
    args.decoder = 'bilinear'
    ###############################
    ####### Model Params ##########
    ###############################
    
    args.z_dim = 64 # Data dimensionality
    args.K = 1 # How many different kernels to train
    
    args.N = 1 ## Number of Leapfrogs
    args.gamma = 0.15 ## Stepsize
    args.alpha = 0.5  ## For partial momentum refresh
    
    args.amortize = True
    
    ###############################
    ######## Data Params ##########
    ###############################
    args.n_data = 0
    
    args.vds = 10000 ## Validation data set
    
    args.train_batch_size = 250
    args.test_batch_size = 10 ## Batch size test
    args.val_batch_size = 1000 ## batch size validation
    
    args.num_batches = 1000
    args.num_epoches = 400
    args.early_stopping_tolerance = 10000
    
    
    args.neutralizing_idea = False  # if we want to perform HMC in warped space
    args.num_neutralizing_flows = 5 # how many neutralizing flows (NAFs) to use
    args.use_barker = True
    args.use_partialref = True
    
    if args.neutralizing_idea:
        naf = []
        for i in range(args.num_neutralizing_flows):
            hidden_units = args.z_dim * 3
            one_arn = AutoRegressiveNN(args.z_dim, [args.z_dim * 2], param_dims=[hidden_units] * 3).to(args.device)
            one_naf = NeuralAutoregressive(one_arn, hidden_units=hidden_units)
            naf.append(one_naf)
        args.naf = nn.ModuleList(naf)
    
    args.std_normal = torch.distributions.Normal(loc=torch.tensor(0., dtype=args.torchType, device=args.device),
                                                scale=torch.tensor(1., dtype=args.torchType, device=args.device))

    ########################################### Data Params ###########################################
    # Multivariate normal
    args['loc'] = torch.tensor([0., 0.], dtype=args.torchType, device=args.device) # loc for Gaussian target
    args['cov_matrix'] = torch.tensor([[50.05, -49.95], [-49.95, 50.05]], dtype=args.torchType, device=args.device) # cov_matrix for Gaussian target
    args['true_mean'] = torch.zeros(args['z_dim'], device=args.device, dtype=args.torchType)

    # GMM (two gaussians)
    args['p_first_gaussian'] = 0.5 # Probability (weight) of the first gaussian
    gaussian_centers = [-5., 5.]
    args['locs_single_gmm'] = [torch.tensor([gaussian_centers[0], 0.], dtype=args.torchType, device=args.device),
               torch.tensor([gaussian_centers[1], 0.], dtype=args.torchType, device=args.device)] # locs
    args['covs_single_gmm'] = [torch.eye(2, dtype=args.torchType, device=args.device),
           torch.eye(2, dtype=args.torchType, device=args.device)] # covariances

    # Banana
    args['banana_cov_matrix'] = torch.tensor([[1., .9], [.9, 1.]], dtype=args.torchType, device=args.device) # covariance
    args['banana_a'] = 1.15
    args['banana_b'] = 0.5

    # Rough Well
    args['rw_eps'] = 1e-2
    args['rw_easy'] = True

    # Examples from BNAF paper
    args['bnaf_data'] = 't4' # t1, t2, t3, t4

    # GMM with arbitraty many components
    comp_1 = 100
    comp_2 = 70

    args['num_gauss'] = 8
    args['p_gaussians'] = [torch.tensor(1. / args['num_gauss'], device=args.device, dtype=args.torchType)] * args['num_gauss']
    args['locs'] = [torch.tensor([0., comp_1], dtype=args.torchType, device=args.device),
                   torch.tensor([comp_2, comp_2], dtype=args.torchType, device=args.device),
                   torch.tensor([comp_1, 0.], dtype=args.torchType, device=args.device),
                   torch.tensor([comp_2, -comp_2], dtype=args.torchType, device=args.device),
                   torch.tensor([0., -comp_1], dtype=args.torchType, device=args.device),
                   torch.tensor([-comp_2, -comp_2], dtype=args.torchType, device=args.device),
                   torch.tensor([-comp_1, 0.], dtype=args.torchType, device=args.device),
                   torch.tensor([-comp_2, comp_2], dtype=args.torchType, device=args.device)]  # list of locations for each of these gaussians
    args['covs'] = [torch.eye(2, dtype=args.torchType, device=args.device)] * args['num_gauss']   # list of covariance matrices for each of these gaussians
    
    return args