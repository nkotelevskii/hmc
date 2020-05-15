import torch
import numpy as np
from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch


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

    args.data = 'ml-20m'

    args.total_anneal_steps = 200000
    args.anneal_cap = 0.2

    ###############################
    ####### Model Params ##########
    ###############################

    args.learning_rate = 1e-3  # either common lr (if saparate params = False), or lr only for generative network

    args.print_info_ = 10

    args.n_epoches = 200
    args.train_batch_size = 500
    args.val_batch_size = 2000

    args.metric = NDCG_binary_at_k_batch

    args.model = 'MultiVAE'
    if args.model == 'MultiDAE':
        args.l2_coeff = 0.01 / args.train_batch_size
    elif args.model == 'MultiVAE':
        args.l2_coeff = 0.

    return args