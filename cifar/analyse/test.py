from __future__ import print_function
import argparse
import numpy as np
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models

x = np.ones((5, 3), np.int)
temp = np.array([1, 0, 0, 1, 0]).reshape(-1, 1)
print(x[:])
x[:] = temp[:]
print(x)