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
        
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    args = {}
    args = dotdict(args)

    args.device = device
    args.torchType = torch.float32
    
    args.data = 'mnist'
    args.decoder = 'deconv'
    ###############################
    ####### Model Params ##########
    ###############################
    args.n_samples = 1 # how many samples for estimation to take
    args.n_alpha = None # None if itsnot needed
    
    
    args.learning_rate = 1e-5 # either common lr (if saparate params = False), or lr only for generative network
    args.learning_rate_vanilla = 1e-3
    args.learning_rate_inference = 1e-3
    
    args.vanilla_vae_epoches = 0
    args.z_dim = 64 # Data dimensionality
    args.K = 1 # How many different kernels to train
    args.N = 3 ## Number of Leapfrogs
    args.gamma = 0.1 ## Stepsize
    args.alpha = 0.5  ## For partial momentum refresh
    
    args.nf_prior = 'IAF'
    args.num_flows_prior = 2
    
    args.separate_params = True # Whether to separate params for training our alg or not
    args.hoffman_idea = True and args.separate_params ## Whether to use Hoffman's idea of separating objectives or not (note that usable only if separate params == True)

    args.use_batchnorm = True # whether to use batch norm layer in decoder or not
    args.train_only_inference_period = 10  # period
    args.train_only_inference_cutoff = 7  # how many times we train ONLY inference part
    
    args.fix_transition_params = False  # whether to freeze transition params
    args.amortize = False # whether to amortize transitions
    args.learnable_reverse = True  # whether to learn reverse
    args.clip_norm = False
    args.clip_value = 5.    
    
    
    ###############################
    ######## Data Params ##########
    ###############################
    args.n_data = 0
    
    args.vds = 1000 ## Validation data set
    args.train_data_size = 15000
    args.train_batch_size = 100
    args.test_batch_size = 10 ## Batch size test
    args.val_batch_size = 100 ## batch size validation
    args.n_IS = 1000
    
    args.num_batches = 20000
    args.num_epoches = 1000
    args.early_stopping_tolerance = 50
    
    
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
    args['loc'] = torch.tensor([10., 0.], dtype=args.torchType, device=args.device) # loc for Gaussian target
    args['cov_matrix'] = torch.tensor([[5, -2], [-2, 5]], dtype=args.torchType, device=args.device) # cov_matrix for Gaussian target
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
    comp_1 = 10
    comp_2 = 7

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