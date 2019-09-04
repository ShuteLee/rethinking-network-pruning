import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# import torchvision.models as models
from mobilenet import *
from mobilenet_v1 import *

# import models.imagenet as customized_models

# from utils import AverageMeter, accuracy, mkdir_p

from math import cos, pi
import shutil

import argparse
import os
import time
from utils import progress_bar

best_prec1 = 0

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR', default='../tiny-imagenet-200',
                    help='path to dataset')
parser.add_argument('--data-backend', metavar='BACKEND', default='pytorch')
parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenet',
                    help='model architecture(default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=240, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--lr-decay', type=str, default='step',
                    help='mode for learning rate decay')
parser.add_argument('--step', type=int, default=30,
                    help='interval for learning rate decay in step mode')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.01,
                    help='LR is multiplied by gamma on schedule.')
parser.add_argument('--warmup', action='store_true',
                    help='set lower initial learning rate to warm up the training')

parser.add_argument('-c', '--checkpoint', default='checkpoints', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoints)')

parser.add_argument('--width-mult', type=float, default=1.0, help='MobileNet model width multiplier.')
parser.add_argument('--weight', default='', type=str, metavar='WEIGHT',
                    help='path to pretrained weight (default: none)')

args = parser.parse_args()

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = MobileNetV2()
    model = torch.nn.DataParallel(model).cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if not os.path.isdir(args.checkpoint):
        os.makedirs(args.checkpoint)

    if args.resume:
        if os.path.isfile(args.resume):
    
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            args.checkpoint = os.path.dirname(args.resume)
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # training
    best_prec1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        print('\nEpoch: [%d | %d]' % (epoch + 1, args.epochs))

        # train for one epoch
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        val_loss, prec1 = validate(val_loader, model, criterion)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.checkpoint)

    print('Best accuracy:')
    print(best_prec1)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, criterion, optimizer, epoch):
    train_loss = 0
    correct = 0
    total = 0
    train_loader_len = len(train_loader)
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        adjust_learning_rate(optimizer, epoch, i, train_loader_len)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        train_loss += loss.data.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += predicted.eq(target.data).cpu().sum()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(i, len(train_loader), 'Loss:%.3f | Acc1:%.3f%% (%d/%d)' % (train_loss/(i+1), float(100.*correct)/total, correct, total))
    # print('Train: Loss: {}; Acc: top1: {}, top5:{}'.format(losses.avg, top1.avg, top5.avg))
    return (train_loss/(i+1), float(100.*correct)/total)


def validate(val_loader, model, criterion):
    test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()
            output = model(input)
            loss = criterion(output, target)

            test_loss += loss.data.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

            progress_bar(i, len(val_loader), 'Loss:%.3f | Acc1:%.3f%% (%d/%d)' % (test_loss/(i+1), float(100.*correct)/total, correct, total))
    # print('Val: Loss: {}; Acc: top1: {}, top5:{}'.format(losses.avg, top1.avg, top5.avg))
    return (test_loss/(i+1), float(100.*correct)/total)

def adjust_learning_rate(optimizer, epoch, iteration, num_iter):
    lr = optimizer.param_groups[0]['lr']

    warmup_epoch = 5 if args.warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = args.epochs * num_iter

    if args.lr_decay == 'step':
        lr = args.lr * (args.gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif args.lr_decay == 'cos':
        lr = args.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif args.lr_decay == 'linear':
        lr = args.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif args.lr_decay == 'schedule':
        count = sum([1 for s in args.schedule if s <= epoch])
        lr = args.lr * pow(args.gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_decay))

    if epoch < warmup_epoch:
        lr = args.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


if __name__ == "__main__":
    main()