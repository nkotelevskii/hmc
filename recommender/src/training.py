import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

def train_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
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
            current_metric = metric_dist.mean()
            metric_vad.append(current_metric)

            # update the best model (if necessary)
            if current_metric > best_metric:
                torch.save(model, './best_model_{}.pt'.format(args.model))
                best_metric = current_metric
            if epoch % print_info_ == 0:
                print('Best NDCG:', best_metric)
                print('Current NDCG:', current_metric)
    return metric_vad


def train_met_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
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

            logits, log_q, log_prior, log_r, sum_log_alpha = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            log_p = torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # complete objective
            elbo_full = log_p + (log_prior + log_r - log_q) * anneal
            grad_elbo = torch.mean(elbo_full + elbo_full.detach() * sum_log_alpha)
            (-grad_elbo).backward()

            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 100 == 0) and (epoch % print_info_ == 0):
                print(elbo_full.cpu().detach().mean().numpy())

            update_count += 1
        # compute validation NDCG
        model.eval()
        metric_dist = []
        for bnum, batch_val in enumerate(dataset.next_val_batch()):
            pred_val, _, _, _, _ = model(batch_val[0], is_training_ph=0.)
            X = batch_val[0].cpu().detach().numpy()
            pred_val = pred_val.cpu().detach().numpy()
            # exclude examples from training and validation (if any)
            pred_val[X.nonzero()] = -np.inf
            metric_dist.append(args.metric(pred_val, batch_val[1]))

        metric_dist = np.concatenate(metric_dist)
        current_metric = metric_dist.mean()
        metric_vad.append(current_metric)

        # update the best model (if necessary)
        if current_metric > best_metric:
            torch.save(model, './best_model_{}.pt'.format(args.model))
            best_metric = current_metric
        if epoch % print_info_ == 0:
            print('Best NDCG:', best_metric)
            print('Current NDCG:', current_metric)
    return metric_vad