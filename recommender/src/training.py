import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

# TODO: Dataset
# TODO: args

def train_model(model, dataset, args):
    metric_vad = []
    best_ndcg = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            logits, KL = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            neg_ll = -torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # complete objective
            neg_ELBO = neg_ll + anneal * KL
            neg_ELBO.backward()

            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 100 == 0) and (epoch % print_info_ == 0):
                print(neg_ELBO.cpu().detach().numpy())

            update_count += 1
        # compute validation NDCG
        model.eval()
        with torch.no_grad():
            metric_dist = []
            for bnum, batch_val in enumerate(dataset.next_val_batch()):
                pred_val, _ = model(batch_val[0], is_training_ph=0.)
                X = batch_val[0].cpu().detach().numpy()
                pred_val = pred_val.cpu().detach().numpy()
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                metric_dist.append(args.metric(pred_val, batch_val[1]))

            metric_dist = np.concatenate(metric_dist)
            ndcg_ = metric_dist.mean()
            metric_vad.append(ndcg_)

            # update the best model (if necessary)
            if ndcg_ > best_ndcg:
                torch.save(model, './best_model_{}.pt'.format(args.model))
                best_ndcg = ndcg_
            if epoch % print_info_ == 0:
                print('Best NDCG:', best_ndcg)
                print('Current NDCG:', ndcg_)
    return metric_vad