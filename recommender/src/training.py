import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pdb


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

            # compute objective
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
                pdb.set_trace()
                metric_dist.append(args.metric(pred_val, batch_val[1]))

            metric_dist = np.concatenate(metric_dist)
            current_metric = metric_dist.mean()
            metric_vad.append(current_metric)

            # update the best model (if necessary)
            if current_metric > best_metric:
                torch.save(model,
                           '../models/best_model_{}_K_{}_N_{}_learnreverse_{}_anneal_{}.pt'.format(args.model, args.K,
                                                                                                   args.N,
                                                                                                   args.learnable_reverse,
                                                                                                   args.annealing))
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

            logits, log_q, log_priors, log_r, sum_log_alpha = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            log_p = torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # compute objective
            elbo_full = log_p + (log_priors + log_r - log_q) * anneal
            grad_elbo = torch.mean(elbo_full + elbo_full.detach() * sum_log_alpha)
            (-grad_elbo).backward()

            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 100 == 0) and (epoch % print_info_ == 0):
                print(elbo_full.cpu().detach().mean().numpy())

            update_count += 1
        if np.isnan(elbo_full.cpu().detach().mean().numpy()):
            break
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
            torch.save(model,
                       '../models/best_model_{}_K_{}_N_{}_learnreverse_{}_anneal_{}.pt'.format(args.model, args.K,
                                                                                               args.N,
                                                                                               args.learnable_reverse,
                                                                                               args.annealing))
            best_metric = current_metric
        if epoch % print_info_ == 0:
            print('Best NDCG:', best_metric)
            print('Current NDCG:', current_metric)
    return metric_vad


def train_hoffman_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0
    optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=args.learning_rate, weight_decay=args.l2_coeff)
    optimizer_decoder = torch.optim.Adam(model.target.decoder.parameters(), lr=args.learning_rate,
                                         weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            logits, KL, logits_pre = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits_pre)
            log_p = torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # compute the first objective
            obj_1 = log_p - KL * anneal
            (-obj_1).backward(retain_graph=True)
            optimizer_encoder.step()
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            # compute the second objective
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            log_p = torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))
            obj_2 = log_p
            (-obj_2).backward()
            optimizer_decoder.step()
            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()

            if (bnum % 100 == 0) and (epoch % print_info_ == 0):
                print('obj_1', obj_1.cpu().detach().numpy())
                print('obj_2', obj_2.cpu().detach().numpy())

            update_count += 1
        if np.isnan(obj_1.cpu().detach().numpy()):
            break
        # compute validation NDCG
        model.eval()
        metric_dist = []
        for bnum, batch_val in enumerate(dataset.next_val_batch()):
            pred_val, _, _ = model(batch_val[0], is_training_ph=0.)
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
            torch.save(model,
                       '../models/best_model_{}_K_{}_N_{}_learnreverse_{}_anneal_{}.pt'.format(args.model, args.K,
                                                                                               args.N,
                                                                                               args.learnable_reverse,
                                                                                               args.annealing))
            best_metric = current_metric
        if epoch % print_info_ == 0:
            print('Best NDCG:', best_metric)
            print('Current NDCG:', current_metric)
    return metric_vad


def train_methoffman_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0
    reverse_params = []
    if args.learnable_reverse:
        reverse_params = model.reverse_kernel.parameters()
    optimizer_inference = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.transitions.parameters()) + list(reverse_params),
        lr=args.learning_rate, weight_decay=args.l2_coeff)
    optimizer_decoder = torch.optim.Adam(model.target.decoder.parameters(), lr=args.learning_rate,
                                         weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            logits, log_q, log_priors, log_r, sum_log_alpha, logits_pre = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits_pre)
            log_p = torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # compute the first objective
            elbo_full = log_p + (log_priors + log_r - log_q) * anneal
            grad_elbo = torch.mean(elbo_full + elbo_full.detach() * sum_log_alpha)
            (-grad_elbo).backward(retain_graph=True)
            optimizer_inference.step()
            optimizer_inference.zero_grad()
            optimizer_decoder.zero_grad()

            # compute the second objective
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            obj_2 = torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))
            (-obj_2).backward()
            optimizer_decoder.step()
            optimizer_decoder.zero_grad()
            optimizer_inference.zero_grad()

            if (bnum % 100 == 0) and (epoch % print_info_ == 0):
                print(elbo_full.cpu().detach().mean().numpy())

            update_count += 1
        if np.isnan(elbo_full.cpu().detach().mean().numpy()):
            break
        # compute validation NDCG
        model.eval()
        metric_dist = []
        for bnum, batch_val in enumerate(dataset.next_val_batch()):
            pred_val, _, _, _, _, _ = model(batch_val[0], is_training_ph=0.)
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
            torch.save(model,
                       '../models/best_model_{}_K_{}_N_{}_learnreverse_{}_anneal_{}.pt'.format(args.model, args.K,
                                                                                               args.N,
                                                                                               args.learnable_reverse,
                                                                                               args.annealing))
            best_metric = current_metric
        if epoch % print_info_ == 0:
            print('Best NDCG:', best_metric)
            print('Current NDCG:', current_metric)
    return metric_vad
