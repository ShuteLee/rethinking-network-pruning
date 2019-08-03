import argparse
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from models import *

parser = argparse.ArgumentParser(description='evaluate the saved model')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to the model')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
args = parser.parse_args()
args.cuda = torch.cuda.is_available()

if args.model:
    if os.path.isfile(args.model):
        print("=> loading checkpoint '{}'".format(args.model))
        checkpoint = torch.load(args.model) #state_dict
        cfg = checkpoint['cfg']
        weights = checkpoint['state_dict']

keys = list(weights.keys())
val = []
for key in keys:
    if len(weights[key].shape) == 4:
        temp = weights[key].cpu().abs().numpy()
        val.extend(np.sum(temp, axis=(1,2,3)) / (temp.shape[1] * temp.shape[2] * temp.shape[3]))


# his, edge = np.histogram(weights[keys[1]].cpu(), range=(0, 10))
# print(his, edge)
plt.hist(val, bins=100, range=(0, 0.2), alpha=0.7)
plt.show()