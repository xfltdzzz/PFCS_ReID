# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta
import os
from sklearn.cluster import DBSCAN
from torch.distributions.multivariate_normal import MultivariateNormal
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from PFCS import datasets
from PFCS import models
from PFCS.models.cm import ClusterMemory
from PFCS.trainers import Trainer
from PFCS.utils.data import IterLoader
from PFCS.utils.data import transforms as T
from PFCS.utils.data.preprocessor import Preprocessor
from PFCS.utils.logging import Logger
from PFCS.utils.faiss_rerank import compute_jaccard_distance
from PFCS.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from PFCS.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from PFCS.evaluators import Evaluator, extract_features,conf_eval_gauss

start_epoch = best_mAP = 0


def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)

    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)


    model_teacher = models.create(args.arch, pretrained=False, num_features=args.features, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type)

    model_teacher.cuda()
    model_teacher = nn.DataParallel(model_teacher)

    # Create model
    model = create_model(args)
    criterion_CE = nn.CrossEntropyLoss(reduction='none')
    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # Trainer
    model_teacher.train()
    trainer = Trainer(model, model_teacher)

    for epoch in range(-1, args.epochs):
        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

            if epoch == -1:
                features, features_up, features_down,features_mix, _ = extract_features(model_teacher, cluster_loader, print_freq=50)
            else:
                features, features_up, features_down,features_mix,_ = extract_features(model, cluster_loader, print_freq=50)

            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            features_up = torch.cat([features_up[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            features_down = torch.cat([features_down[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            features_mix = torch.cat([features_mix[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            prob_full = None
            distribution_h_dict = {}
            distribution_g_dict = {}
            distribution_l_dict = {}

            if epoch > 0:
                ind = (pseudo_labels >= 0)
                prob_full = torch.ones(len(pseudo_labels), dtype=torch.float32) * 0.1
                p = pseudo_labels[ind]

                loss = criterion_CE(features_mix[ind], torch.from_numpy(p.astype(int)))

                loss = (loss - loss.min()) / (loss.max() - loss.min())
                loss_re = loss.reshape(-1, 1)

                prob = conf_eval_gauss(loss_re)
                prob_full[ind] = torch.from_numpy(prob).float()

                for p in set(pseudo_labels):
                    if p == -1:
                        continue

                    p_index = (pseudo_labels == p)
                    noise_index = torch.from_numpy(p_index) & (prob_full < args.noisy_threshold)
                    clean_index = torch.from_numpy(p_index) & (prob_full >= args.noisy_threshold)
                    sp = prob_full[noise_index]
                    dp = 1 - prob_full[noise_index]
                    f_h = features[clean_index]
                    f_g = features_up[clean_index]
                    f_l = features_down[clean_index]

                    if len(f_h) < 2:
                        continue

                    num = len(dp)
                    if num == 0:
                        continue

                    c_h = torch.cov(f_h.T)
                    c_g = torch.cov(f_g.T)
                    c_l = torch.cov(f_l.T)

                    distribution_h_dict[p] = (torch.mean(f_h, axis=0), torch.mm(c_h, c_h.T).add_(torch.eye(2048)))
                    distribution_g_dict[p] = (torch.mean(f_g, axis=0), torch.mm(c_g, c_g.T).add_(torch.eye(2048)))
                    distribution_l_dict[p] = (torch.mean(f_l, axis=0), torch.mm(c_l, c_l.T).add_(torch.eye(2048)))

                    features[noise_index] = torch.mul(features[noise_index].t(), sp).t() + torch.mul(MultivariateNormal(distribution_h_dict[p][0], distribution_h_dict[p][1]).sample().repeat(num).reshape(num, -1).t(), dp).t()
                    features_up[noise_index] = torch.mul(features_up[noise_index].t(), sp).t() + torch.mul(MultivariateNormal(distribution_g_dict[p][0], distribution_g_dict[p][1]).sample().repeat(num).reshape(num, -1).t(), dp).t()
                    features_down[noise_index] = torch.mul(features_down[noise_index].t(), sp).t() + torch.mul(MultivariateNormal(distribution_l_dict[p][0], distribution_l_dict[p][1]).sample().repeat(num).reshape(num, -1).t(), dp).t()


            rerank_dist = compute_jaccard_distance(features, k1=args.k1, k2=args.k2)
            rerank_dist_up = compute_jaccard_distance(features_up, k1=args.k1, k2=args.k2)
            rerank_dist_down = compute_jaccard_distance(features_down, k1=args.k1, k2=args.k2)

            rerank_dist = (1.0 - args.lambda1 * 2) * rerank_dist + args.lambda1 * rerank_dist_up + args.lambda1 * rerank_dist_down

            if epoch == -1:
                # DBSCAN cluster
                eps = args.eps
                print('Clustering criterion: eps: {:.3f}'.format(eps))
                cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)

                # select & cluster images as training set of this epochs
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)


        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features = generate_cluster_features(pseudo_labels, features)
        cluster_features_up = generate_cluster_features(pseudo_labels, features_up)
        cluster_features_down = generate_cluster_features(pseudo_labels, features_down)

        del cluster_loader, features, features_up, features_down,features_mix

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard, lambda2=args.lambda2, mu=args.mu).cuda()
        memory.features = F.normalize(cluster_features, dim=1).cuda()
        memory.features_up = F.normalize(cluster_features_up, dim=1).cuda()
        memory.features_down = F.normalize(cluster_features_down, dim=1).cuda()

        trainer.memory = memory

        pseudo_labeled_dataset = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))
        
        iters = args.iters * 2 if (epoch == -1) else args.iters

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, no_cam=args.no_cam)

        train_loader.new_epoch()

     
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader),lamb=args.lamb,lamb2=args.lamb2)

        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
            
        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best_student.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.55,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.3,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=20)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    ##### solve the problem that the python path is not true
    working_dir = osp.join(working_dir, '..')

    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, '/'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam',  action="store_true")
    parser.add_argument('--lambda1', type=float, default=0.2)
    parser.add_argument('--lambda2', type=float, default=0.2)

    parser.add_argument('--noisy-threshold', type=float, default=0.05)
    parser.add_argument('--mu', type=float, default=0.7)
    parser.add_argument('--lamb', type=float, default=0.5)
    parser.add_argument('--lamb2', type=float, default=0.3)
    main()

