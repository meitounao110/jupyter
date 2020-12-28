#!coding:utf-8
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from util import datasets, Trainer
from architectures.arch import arch as arch_n
from architectures.fpn import Fpn_n

from util.datasets import NO_LABEL


def data_loaders(
        train_transform,
        eval_transform,
        heat_transform,
        datadir,
        config):
    traindir = os.path.join(datadir, config.train_subdir)
    trainset = datasets.dataloader(root=traindir,
                                   list_path1=config.list_path1,
                                   list_path2=config.list_path2,
                                   transform=train_transform,
                                   target_transform=heat_transform,
                                   max_iters=None)  # 这里缺最大数据量
    if config.labels:  # 这里给出有标签列表，得到有标签的索引和无标签的索引
        with open(config.labels) as f:
            labels = f.read().splitlines()
    labeled_idxs, unlabeled_idxs = datasets.relabel_dataset(trainset, labels)
    # assert len(trainset.sample_files) == len(labeled_idxs) + len(unlabeled_idxs)
    if config.labeled_batch_size < config.batch_size:  # 如果有标签batchsize小于总batchsize
        assert len(unlabeled_idxs) > 0  # 无标签索引为0时报错
        batch_sampler = datasets.TwoStreamBatchSampler(
            unlabeled_idxs, labeled_idxs, config.batch_size, config.labeled_batch_size)
    else:
        sampler = SubsetRandomSampler(labeled_idxs)
        batch_sampler = BatchSampler(sampler, config.batch_size, drop_last=True)
    train_loader = torch.utils.data.DataLoader(trainset,
                                               batch_sampler=batch_sampler,
                                               num_workers=config.workers,
                                               pin_memory=True)
    evaldir = os.path.join(datadir, config.eval_subdir)
    # evalset = torchvision.datasets.ImageFolder(evaldir, eval_transform)
    evalset = datasets.dataloader(root=evaldir,
                                  list_path1=config.test_list_path,
                                  list_path2=None,
                                  transform=eval_transform,
                                  target_transform=heat_transform,
                                  max_iters=None)
    eval_loader = torch.utils.data.DataLoader(evalset,
                                              batch_size=config.batch_size,
                                              shuffle=False,
                                              num_workers=2 * config.workers,
                                              pin_memory=True,
                                              drop_last=False)  # 测试用的
    return train_loader, eval_loader


# def create_data_loaders(train_transform,
#                         eval_transform,
#                         datadir,
#                         config):
#     traindir = os.path.join(datadir, config.train_subdir)
#     trainset = torchvision.datasets.ImageFolder(traindir, train_transform)
#     if config.labels:
#         with open(config.labels) as f:
#             labels = dict(line.split(' ') for line in f.read().splitlines())
#         labeled_idxs, unlabeled_idxs = datasets.relabel_dataset(trainset, labels)
#     assert len(trainset.imgs) == len(labeled_idxs) + len(unlabeled_idxs)
#     # 这里用两个batchsize是因为，无标签和有标签的数据不一样，为了匹配，所以一个batch里需要一定比例的有标签和无标签
#     if config.labeled_batch_size < config.batch_size:  # 如果有标签batchsize小于总batchsize
#         assert len(unlabeled_idxs) > 0  # 无标签索引为0时报错
#         batch_sampler = datasets.TwoStreamBatchSampler(
#             unlabeled_idxs, labeled_idxs, config.batch_size, config.labeled_batch_size)
#     else:
#         sampler = SubsetRandomSampler(labeled_idxs)
#         batch_sampler = BatchSampler(sampler, config.batch_size, drop_last=True)
#     train_loader = torch.utils.data.DataLoader(trainset,
#                                                batch_sampler=batch_sampler,
#                                                num_workers=config.workers,
#                                                pin_memory=True)
#
#     evaldir = os.path.join(datadir, config.eval_subdir)
#     evalset = torchvision.datasets.ImageFolder(evaldir, eval_transform)
#     eval_loader = torch.utils.data.DataLoader(evalset,
#                                               batch_size=config.batch_size,
#                                               shuffle=False,
#                                               num_workers=2 * config.workers,
#                                               pin_memory=True,
#                                               drop_last=False)
#     return train_loader, eval_loader


def create_loss_fn(config):
    if config.loss == 'mse':
        # for pytorch 0.4.0
        # criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction=None)
        criterion = nn.MSELoss()
        # for pytorch 0.4.1
        # criterion = nn.CrossEntropyLoss(ignore_index=NO_LABEL, reduction='none')
    return criterion


def create_optim(params, config):
    if config.optim == 'sgd':
        optimizer = optim.SGD(params, config.lr,
                              momentum=config.momentum,
                              weight_decay=config.weight_decay,
                              nesterov=config.nesterov)
    elif config.optim == 'adam':
        optimizer = optim.Adam(params, config.lr)
    return optimizer


def create_lr_scheduler(optimizer, config):
    if config.lr_scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=config.epochs,
                                                   eta_min=config.min_lr)
    elif config.lr_scheduler == 'multistep':
        if config.steps == "":
            return None
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=config.steps,
                                             gamma=config.gamma)
    elif config.lr_scheduler == 'none':
        scheduler = None
    return scheduler


def main(config):
    # SummaryWriter画图用的
    with SummaryWriter(comment='_{}_{}'.format(config.arch, config.dataset)) as writer:
        # 选择datasets中的cifar10
        dataset_config = datasets.FPN(config) if config.dataset == 'FPN' else datasets.cifar10()
        # dataset_config = datasets.cifar10() if config.dataset == 'cifar10' else datasets.cifar100()
        # num_classes为类别数
        # num_classes = dataset_config.pop('num_classes')
        train_loader, eval_loader = data_loaders(**dataset_config, config=config)

        dummy_input = torch.randn(1, 1, 200, 200)  # 添加一个模型的图
        # fpnmodel = arch_n[config.arch]
        net = Fpn_n()
        writer.add_graph(net, dummy_input)

        device1 = 'cuda' if torch.cuda.is_available() else 'cpu'
        criterion = create_loss_fn(config)
        if config.is_parallel:
            net = torch.nn.DataParallel(net).to(device1)
        else:
            device1 = 'cuda:{}'.format(config.gpu) if torch.cuda.is_available() else 'cpu'
            net = net.to(device1)
        optimizer = create_optim(net.parameters(), config)
        trainer = Trainer.PseudoLabel(net, optimizer, criterion, device1, config, writer)
        scheduler = create_lr_scheduler(optimizer, config)
        trainer.loop(config.epochs, train_loader, eval_loader,
                     scheduler=scheduler, print_freq=config.print_freq)
