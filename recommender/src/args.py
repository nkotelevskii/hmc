import torch
import numpy as np
from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch


def get_args(args):
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    args = dotdict(vars(args))

    args.device = device
    args.torchType = torch.float32

    ## Data and training parameters
    args.train_batch_size = 500
    args.val_batch_size = 2000
    args.n_epoches = 200

    args.total_anneal_steps = 200000
    args.anneal_cap = 0.2

    args.learning_rate = 1e-3

    args.print_info_ = 1

    ## Transition parameters (only for our vae)
    args.gamma = 0.1  # Stepsize
    args.alpha = 0.5   # For partial momentum refresh
    args.use_barker = True
    args.use_partialref = True

    args.learnable_reverse = True if args.learnable_reverse == 'True' else False


    ## Metric
    args.metric = NDCG_binary_at_k_batch

    if args.model == 'MultiDAE':
        args.l2_coeff = 0.01 / args.train_batch_size
    else:
        args.l2_coeff = 0.

    return args