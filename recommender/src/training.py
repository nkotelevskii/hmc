import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
import pdb

def train_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    if args.model == 'MultiDAE':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lrdec, weight_decay=args.l2_coeff)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()}
        ],
            lr=lrenc, weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):
            # pdb.set_trace()
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
                # pdb.set_trace()
                reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
                is_training_ph = int(args.n_val_samples > 1)
                pred_val, _ = model(reshaped_batch, is_training_ph=is_training_ph)
                pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
                X = batch_val[0].cpu().detach().numpy()
                pred_val = pred_val.cpu().detach().numpy()
                # exclude examples from training and validation (if any)
                pred_val[X.nonzero()] = -np.inf
                # pdb.set_trace()
                metric_dist.append(args.metric(pred_val, batch_val[1]))

            metric_dist = np.concatenate(metric_dist)
            current_metric = metric_dist.mean()
            metric_vad.append(current_metric)

            # update the best model (if necessary)
            if current_metric > best_metric:
                torch.save(model,
                           '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}.pt'.format(
                               args.model, args.data, args.K,
                               args.N,
                               args.learnable_reverse,
                               args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma))
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

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    if not args.learntransitions:
        for p in model.transitions.parameters():
            p.requires_grad_(False)
    else:
        for k in range(len(model.transitions)):
            model.transitions[k].alpha_logit.requires_grad_(False)

    if args.learnable_reverse:
        optimizer = torch.optim.Adam([
            {'params': model.target.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()},
            {'params': model.transitions.parameters()},
            {'params': model.reverse_kernel.parameters()},
            {'params': model.momentum_scale},
        ],
            lr=lrenc, weight_decay=args.l2_coeff)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.target.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()},
            {'params': model.transitions.parameters()},
            {'params': model.momentum_scale},
        ],
            lr=lrenc, weight_decay=args.l2_coeff)
    if args.data == 'ml20m':
        scheduler = MultiStepLR(optimizer, [10, 40, 60, 100], gamma=0.2)
        # scheduler = MultiStepLR(optimizer, [20, 40, 100], gamma=0.2)
    elif args.data == 'gowalla':
        scheduler = MultiStepLR(optimizer, [20, 50, 75, 100, 150], gamma=0.75)
    else:
        scheduler = MultiStepLR(optimizer, [20, 50, 100, 150], gamma=0.25)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            # pdb.set_trace()
            logits, log_q, log_aux, log_priors, log_r, sum_log_alpha, directions = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            log_likelihood = torch.sum(log_softmax_var * batch_train, 1).mean()
            # compute objective
            KLD = log_q.mean() + log_aux.mean() - log_r.mean() - log_priors.mean()
            elbo_full = log_likelihood - anneal * KLD

            grad_elbo = elbo_full + elbo_full.detach() * torch.mean(sum_log_alpha)
            (-grad_elbo).backward()

            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 200 == 0) and (epoch % print_info_ == 0):
                print('Current anneal coeff:', anneal)
                if args.learnscale:
                    print('Min scale', torch.exp(model.momentum_scale.detach()).min().item(), 'Max scale', torch.exp(model.momentum_scale.detach()).max().item())
                print(elbo_full.cpu().detach().mean().numpy())
                for k in range(args.K):
                    print('k =', k)
                    print('0: {} and for +1: {}'.format((directions[:, k] == 0.).to(float).mean(), (directions[:, k] == 1.).to(float).mean()))
                    print('autoreg:', torch.sigmoid(model.transitions[k].alpha_logit.detach()).item())
                    print('stepsize', torch.exp(model.transitions[k].gamma.detach()).item())
                    print('-' * 100)

            update_count += 1
        if np.isnan(elbo_full.cpu().detach().mean().numpy()):
            break

        if (args.data in ['ml20m', 'gowalla', 'foursquare']): # and not args.annealing:
            scheduler.step()
            # if epoch in [20, 30, 50, 100]:
            #     for pr_gr, g in enumerate(optimizer.param_groups):
            #         if pr_gr == 1:
            #             g['lr'] *= 2.5
        if epoch % print_info_ == 0:
            for param_group in optimizer.param_groups:
                print(param_group['lr'])

        # compute validation NDCG
        model.eval()
        metric_dist = []
        for bnum, batch_val in enumerate(dataset.next_val_batch()):
            reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
            is_training_ph = int(args.n_val_samples > 1)
            pred_val, _, _, _, _, _, _ = model(reshaped_batch, is_training_ph=is_training_ph)
            pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
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
                       '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}_learnscale_{}.pt'.format(
                           args.model, args.data, args.K,
                           args.N,
                           args.learnable_reverse,
                           args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma, args.learnscale))
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

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    optimizer_encoder = torch.optim.Adam(model.encoder.parameters(), lr=lrenc, weight_decay=args.l2_coeff)
    optimizer_decoder = torch.optim.Adam(model.target.decoder.parameters(), lr=args.lrdec,
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
            reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
            is_training_ph = int(args.n_val_samples > 1)
            pred_val, _, _ = model(reshaped_batch, is_training_ph=is_training_ph)
            pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
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
                       '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}.pt'.format(
                           args.model, args.data, args.K,
                           args.N,
                           args.learnable_reverse,
                           args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma))
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

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    if not args.learntransitions:
        for p in model.transitions.parameters():
            p.requires_grad_(False)
    else:
        for k in range(len(model.transitions)):
            model.transitions[k].alpha_logit.requires_grad_(False)

    reverse_params = []
    if args.learnable_reverse:
        reverse_params = model.reverse_kernel.parameters()
    optimizer_inference = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.transitions.parameters()) + list(reverse_params),
        lr=lrenc, weight_decay=args.l2_coeff)
    optimizer_decoder = torch.optim.Adam(model.target.decoder.parameters(), lr=args.lrdec,
                                         weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            logits, log_q, log_priors, log_r, sum_log_alpha, logits_pre, directions = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits_pre)
            log_p = torch.sum(log_softmax_var * batch_train, dim=1)

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

            if (bnum % 200 == 0) and (epoch % print_info_ == 0):
                print(elbo_full.cpu().detach().mean().numpy())
                for k in range(args.K):
                    print('k =', k)
                    print('0: {} and for +1: {}'.format((directions[:, k] == 0.).to(float).mean(), (directions[:, k] == 1.).to(float).mean()))
                    print('autoreg:', torch.sigmoid(model.transitions[k].alpha_logit.detach()).item())
                    print('stepsize', torch.exp(model.transitions[k].gamma.detach()).item())
                    print('-' * 100)

            update_count += 1
        if np.isnan(elbo_full.cpu().detach().mean().numpy()):
            break
        # compute validation NDCG
        model.eval()
        metric_dist = []
        for bnum, batch_val in enumerate(dataset.next_val_batch()):
            reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
            is_training_ph = int(args.n_val_samples > 1)
            pred_val, _, _, _, _, _, _ = model(reshaped_batch, is_training_ph=is_training_ph)
            pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
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
                       '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}.pt'.format(
                           args.model, args.data, args.K,
                           args.N,
                           args.learnable_reverse,
                           args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma))
            best_metric = current_metric
        if epoch % print_info_ == 0:
            print('Best NDCG:', best_metric)
            print('Current NDCG:', current_metric)
    return metric_vad


def train_neutramet_model(model, dataset, args):
    metric_vad = []
    best_metric = -np.inf
    print_info_ = args.print_info_
    update_count = 0.0

    if args.lrenc is None:
        lrenc = args.lrdec
    else:
        lrenc = args.lrenc

    if not args.learntransitions:
        for p in model.transitions.parameters():
            p.requires_grad_(False)
    else:
        for k in range(len(model.transitions)):
            model.transitions[k].alpha_logit.requires_grad_(False)

    if args.learnable_reverse:
        optimizer = torch.optim.Adam([
            {'params': model.target.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()},
            {'params': model.transitions.parameters()},
            {'params': model.reverse_kernel.parameters()},
        ],
            lr=lrenc, weight_decay=args.l2_coeff)
    else:
        optimizer = torch.optim.Adam([
            {'params': model.target.decoder.parameters(), 'lr': args.lrdec},
            {'params': model.encoder.parameters()},
            {'params': model.transitions.parameters()},
        ],
            lr=lrenc, weight_decay=args.l2_coeff)

    for epoch in tqdm(range(args.n_epoches)):
        model.train()
        for bnum, batch_train in enumerate(dataset.next_train_batch()):

            if args.total_anneal_steps > 0:
                anneal = min(args.anneal_cap, 1. * update_count / args.total_anneal_steps)
            else:
                anneal = args.anneal_cap

            logits, log_q, log_priors, log_r, sum_log_alpha, directions = model(batch_train)

            # loglikelihood part
            log_softmax_var = nn.LogSoftmax(dim=-1)(logits)
            log_p = torch.mean(torch.sum(log_softmax_var * batch_train, dim=1))

            # compute objective
            elbo_full = log_p + (log_priors + log_r - log_q) * anneal
            grad_elbo = torch.mean(elbo_full + elbo_full.detach() * sum_log_alpha)
            (-grad_elbo).backward()

            optimizer.step()
            optimizer.zero_grad()

            if (bnum % 200 == 0) and (epoch % print_info_ == 0):
                print(elbo_full.cpu().detach().mean().numpy())
                for k in range(args.K):
                    print('k =', k)
                    print('0: {} and for +1: {}'.format((directions[:, k] == 0.).to(float).mean(), (directions[:, k] == 1.).to(float).mean()))
                    print('autoreg:', torch.sigmoid(model.transitions[k].alpha_logit.detach()).item())
                    print('stepsize', torch.exp(model.transitions[k].gamma.detach()).item())
                    print('-' * 100)

            update_count += 1
        if np.isnan(elbo_full.cpu().detach().mean().numpy()):
            break
        # compute validation NDCG
        model.eval()
        metric_dist = []
        for bnum, batch_val in enumerate(dataset.next_val_batch()):
            reshaped_batch = batch_val[0].repeat((args.n_val_samples, 1))
            is_training_ph = int(args.n_val_samples > 1)
            pred_val, _, _, _, _, _ = model(reshaped_batch, is_training_ph=is_training_ph)
            pred_val = pred_val.view((args.n_val_samples, *batch_val[0].shape)).mean(0)
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
                       '../models/best_model_{}_data_{}_K_{}_N_{}_learnreverse_{}_anneal_{}_lrdec_{}_lrenc_{}_learntransitions_{}_initstepsize_{}.pt'.format(
                           args.model, args.data, args.K,
                           args.N,
                           args.learnable_reverse,
                           args.annealing, args.lrdec, args.lrenc, args.learntransitions, args.gamma))
            best_metric = current_metric
        if epoch % print_info_ == 0:
            print('Best NDCG:', best_metric)
            print('Current NDCG:', current_metric)
    return metric_vad
