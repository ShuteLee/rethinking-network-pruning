import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable

# import models


model = models.alexnet()

for m in model.modules():
    print(m)

#print(model.parameters().value())