import torch
from metrics import NDCG_binary_at_k_batch, Recall_at_k_batch


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
    args.val_batch_size = 85

    args.print_info_ = 1

    args.annealing = True if args.annealing == 'True' else False
    args.learnable_reverse = True if args.learnable_reverse == 'True' else False
    args.learntransitions = True if args.learntransitions == 'True' else False

    ## Transition parameters (only for our vae)
    if args.learntransitions:
        args.gamma = 0.01
        # args.alpha = 0.5
    else:
        args.gamma = 0.01  # Stepsize
    args.alpha = 0.9  #0.5  # For partial momentum refresh
    args.use_barker = True
    args.use_partialref = True


    if args.annealing:
        args.total_anneal_steps = 46000
    else:
        args.total_anneal_steps = 0
        args.anneal_cap = 1.

    ## Metric
    args.metric = NDCG_binary_at_k_batch #Recall_at_k_batch #

    if args.model == 'MultiDAE':
        args.l2_coeff = 0.01 / args.train_batch_size
    else:
        args.l2_coeff = 0.

    return args
