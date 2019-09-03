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

# import models.imagenet as customized_models

# from utils import AverageMeter, accuracy, mkdir_p

from math import cos, pi
import shutil

import argparse
import os
import time

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d', '--data', metavar='DIR', default='../tiny-imagenet-200',
                    help='path to dataset')
parser.add_argument('-r', '--resume', metavar='DIR', default='checkpoints/model_best.pth.tar',
                    help='path to model')

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
    print("=> creating model")
    model = MobileNetV2()
    model = torch.nn.DataParallel(model).cuda()
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.resume, checkpoint['epoch']))
    args.checkpoint = os.path.dirname(args.resume)

    cudnn.benchmark = True

    # Data loading code
    testdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(testdir, transforms.Compose([
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            normalize,
        ])), batch_size=256, shuffle=False)


    test(test_loader, model, criterion)

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

def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)

            with torch.no_grad():
                # compute output
                output = model(input)
                loss = criterion(output, target)

            # measure accuracy and record loss
            # print('input:{}, output:{}, target:{}'.format(input, output, target))
            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    print('Test: Loss: {}; Acc: top1: {}, top5:{}'.format(losses.avg, top1.avg, top5.avg))
    return (losses.avg, top1.avg)


if __name__ == "__main__":
    main()




