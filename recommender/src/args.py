import torch
from metrics import NDCG_binary_at_k_batch


def get_args(args):
    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = dotdict(vars(args))

    args.device = 'cpu' if args.gpu == -1 else 'cuda:{}'.format(args.gpu)
    args.torchType = torch.float32

    ## Data and training parameters
    args.train_batch_size = 500
    args.val_batch_size = 2000
    args.n_epoches = 200

    args.print_info_ = 1

    ## Transition parameters (only for our vae)
    args.gamma = 0.1  # Stepsize
    args.alpha = 0.5  # For partial momentum refresh
    args.use_barker = True
    args.use_partialref = True

    args.annealing = True if args.annealing == 'True' else False
    args.learnable_reverse = True if args.learnable_reverse == 'True' else False

    if args.annealing:
        args.total_anneal_steps = 200000
        args.anneal_cap = 0.2
    else:
        args.total_anneal_steps = 0
        args.anneal_cap = 1.

    ## Metric
    args.metric = NDCG_binary_at_k_batch

    if args.model == 'MultiDAE':
        args.l2_coeff = 0.01 / args.train_batch_size
    else:
        args.l2_coeff = 0.

    return args
