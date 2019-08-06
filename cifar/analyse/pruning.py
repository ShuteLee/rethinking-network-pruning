import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F

from models import *

class Pruner():
    cuda = torch.cuda.is_available()
    dataset = 'cifar10'
    batch_size = 64
    test_batch_size = 256
    save = '.'
    model_pth = './logs/checkpoint.pth.tar'
    filter_num_scale = [1, 2, 3, 3, 2]
    state = [64, 128, 192, 192, 128]

    lr = 0.001
    momentum = 0.9
    weight_decay = 0

    def __init__(self):
        if not os.path.exists(self.save):
            os.makedirs(self.save)
        
        # original model initialization
        self.model = alexnet(dataset=self.dataset)
        if self.cuda:
            self.model.cuda()
        if os.path.isfile(self.model_pth):
            print("=> loading checkpoint '{}'".format(self.model_pth))
            checkpoint = torch.load(self.model_pth)
            best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            # print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
            #     .format(self.model, checkpoint['epoch'], best_prec1))
        else:
            print("=> no checkpoint found at '{}'".format(self.model_pth))
        print('Pre-processing Successful!')

        kwself = {'num_workers': 0, 'pin_memory': True} if self.cuda else {}
        if self.dataset == 'cifar10':
            self.train_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.Pad(4),
                                transforms.RandomCrop(32),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                            ])),
                batch_size=self.batch_size, shuffle=True, **kwself)

    def test(self):
        kwself = {'num_workers': 0, 'pin_memory': True} if self.cuda else {}
        if self.dataset == 'cifar10':
            test_loader = torch.utils.data.DataLoader(
                datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
                batch_size=self.test_batch_size, shuffle=True, **kwself)
        
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = self.model(data)
                pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        print('Test set: Accuracy: {}/{} ({})\n'.format(
            correct, len(test_loader.dataset), float(correct)/len(test_loader.dataset)))
        return correct / float(len(test_loader.dataset))


    def pruning_list(self, num_list):
        for num in num_list:
            print(self.state)
            temp = np.array(self.state) - np.array(self.filter_num_scale) * num
            temp = temp.tolist()
            cfg = temp.copy()
            cfg.insert(1, 'M')
            cfg.insert(3, 'M')
            cfg.append('M')
            print('cfg is: {}'.format(cfg))
            self.pruning_step(cfg)
            self.state = temp.copy()
        self.fine_tuning(20)
        return self.test()

    def pruning_step(self, cfg):
        cfg_mask = []
        layer_id = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                out_channels = m.weight.data.shape[0]
                if out_channels == cfg[layer_id]:
                    cfg_mask.append(torch.ones(out_channels))
                    layer_id += 1
                    continue
                weight_copy = m.weight.data.abs().clone()
                weight_copy = weight_copy.cpu().numpy()
                L1_norm = np.sum(weight_copy, axis=(1, 2, 3))
                arg_max = np.argsort(L1_norm)
                arg_max_rev = arg_max[::-1][:cfg[layer_id]]
                assert arg_max_rev.size == cfg[layer_id], "size of arg_max_rev not correct"
                mask = torch.zeros(out_channels)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask.append(mask)
                layer_id += 1
            elif isinstance(m, nn.MaxPool2d):
                layer_id += 1

        newmodel = alexnet(dataset=self.dataset, cfg=cfg)
        if self.cuda:
            newmodel.cuda()

        start_mask = torch.ones(3)
        layer_id_in_cfg = 0
        end_mask = cfg_mask[layer_id_in_cfg]
        for [m0, m1] in zip(self.model.modules(), newmodel.modules()):
            if isinstance(m0, nn.BatchNorm2d):
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Conv2d):
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
                print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
                if idx0.size == 1:
                    idx0 = np.resize(idx0, (1,))
                if idx1.size == 1:
                    idx1 = np.resize(idx1, (1,))
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
                w1 = w1[idx1.tolist(), :, :, :].clone()
                m1.weight.data = w1.clone()

                layer_id_in_cfg += 1
                start_mask = end_mask
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
            elif isinstance(m0, nn.Linear):
                if layer_id_in_cfg == len(cfg_mask):
                    temp = np.asarray(cfg_mask[-1].cpu().numpy()).reshape(-1, 1)
                    idx0 = np.zeros((len(temp), 16), dtype=np.int)
                    idx0[:] = temp[:]
                    idx0 = idx0.reshape(1, -1)
                    idx0 = np.squeeze(np.argwhere(np.squeeze(idx0)))
                    if idx0.size == 1:
                        idx0 = np.resize(idx0, (1,))
                    m1.weight.data = m0.weight.data[:, idx0].clone()
                    m1.bias.data = m0.bias.data.clone()
                    layer_id_in_cfg += 1
                    continue
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
            elif isinstance(m0, nn.BatchNorm1d):
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

        self.model = newmodel
        acc = self.fine_tuning(1)
        
        return acc


    def fine_tuning(self, epochs):  
        if epochs == 0:
            acc = self.test()          
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        for epoch in range(epochs):
            avg_loss = 0.
            train_acc = 0.
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if self.cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                avg_loss += loss.item()
                pred = output.data.max(1, keepdim=True)[1]
                train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
                loss.backward()
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(self.train_loader.dataset),
                        100. * batch_idx / len(self.train_loader), loss.item()))

            acc =  self.test()
        return acc

