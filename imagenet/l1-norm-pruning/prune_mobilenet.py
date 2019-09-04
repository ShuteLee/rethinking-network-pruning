import argparse
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import datasets, transforms

from mobilenet import mobileNetV2
from utils import progress_bar


# Prune settings
parser = argparse.ArgumentParser(description='Pruning filters for efficient ConvNets')
parser.add_argument('--data', type=str, default='../tiny-imagenet-200',
                    help='Path to imagenet validation data')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--lr', '--learning-rate', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=4e-5, type=float,
                    metavar='W', help='weight decay (default: 4e-5)',
                    dest='weight_decay')
parser.add_argument('--model-pth', default='model_mobilenetv2/checkpoints/model_best.pth.tar', type=str, metavar='PATH',
                    help='path to load prune model (default: none)')
parser.add_argument('--save', default='.', type=str, metavar='PATH',
                    help='path to save prune model (default: none)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 20)')

args = parser.parse_args()

if not os.path.exists(args.save):
    os.makedirs(args.save)

model = mobileNetV2()
model = torch.nn.DataParallel(model).cuda()

print("=> loading checkpoint '{}'".format(args.model_pth))
checkpoint = torch.load(args.model_pth)
args.start_epoch = checkpoint['epoch']
best_prec1 = checkpoint['best_prec1']
model.load_state_dict(checkpoint['state_dict'])

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
        normalize
    ]))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, sampler=None)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        normalize
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

# criterion = nn.CrossEntropyLoss().cuda()
# optimizer = torch.optim.SGD(model.parameters(), args.lr,
#                             momentum=args.momentum,
#                             weight_decay=args.weight_decay)

print('Pre-processing Successful!')

# Fine-tuning
def fine_tuning(model):
    train_loss = 0
    correct = 0
    total = 0
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        
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
    return (train_loss/(i+1), float(100.*correct)/total)

# Validation
def validate(model):
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
            # print(predicted)
            total += target.size(0)
            correct += predicted.eq(target.data).cpu().sum()

            progress_bar(i, len(val_loader), 'Loss:%.3f | Acc1:%.3f%% (%d/%d)' % (test_loss/(i+1), float(100.*correct)/total, correct, total))
    return (test_loss/(i+1), float(100.*correct)/total)

# Pruning list
filter_num_scale = []
state = []
block_num_scale = [2, 3, 4, 8, 12, 20, 40]
block_state = [16, 24, 32, 64, 96, 160, 320]
block_inc_state = 32
block_ouc_state = 1280
state = []
for m in model.modules():
    if isinstance(m, nn.Conv2d):
        filter_num_scale.append(m.weight.data.shape[0] / 8)
        state.append(m.weight.data.shape[0])

def pruning_list(self, num_list):
    acc_p = []
    for num in num_list:
        self.pruning_step(num)
        self.fine_tuning(1)
        acc_p.append(self.test())
        self.state = temp.copy()
    self.fine_tuning(20)
    acc_f = self.test()
    return acc_p.copy(), acc_f

# pruning step
def pruning_step(model, num):
    cfg = np.array(state) - np.array(filter_num_scale) * num
    cfg = cfg.astype(np.int)
    cfg = cfg.tolist()
    cfg_mask = []
    layer_id = 0
    for m in model.modules():
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

    block_cfg = np.array(block_state) - np.array(block_num_scale) * num
    newmodel = mobileNetV2(cfg=block_cfg, inc=block_inc_state)
    newmodel = torch.nn.DataParallel(newmodel).cuda()

    start_mask = torch.ones(3)  # [1 1 1]
    layer_id_in_cfg = 0
    end_mask = cfg_mask[layer_id_in_cfg]  # 获取layer_id_in_cfg对应mask
    for [m0, m1] in zip(model.modules(), newmodel.modules()):
        # if layer_id_in_cfg < 2:
        #     if isinstance(m0, nn.BatchNorm2d):
        #         layer_id_in_cfg += 1
        #         start_mask = end_mask
        #         end_mask = cfg_mask[layer_id_in_cfg]
        #     continue
        if isinstance(m0, nn.BatchNorm2d):
            # print('-----------------')
            # print(m1)
            # print('-----------------')
            if layer_id_in_cfg < len(cfg_mask) - 1 and layer_id_in_cfg > 1:
                idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # 获取mask的index，并去掉维度为1的维度
                if idx1.size == 1:
                    idx1 = np.resize(idx1,(1,))      # 变成 idx * 1 维度
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()  # BN层赋值
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
            else:
                m1.weight.data = m0.weight.data.clone()  # BN层赋值
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()
            layer_id_in_cfg += 1
            start_mask = end_mask
            if layer_id_in_cfg < len(cfg_mask) - 1:  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]                

        elif isinstance(m0, nn.Conv2d):
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy()))) # input channel
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))  # filter amount
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data.cpu().numpy()
            # print('m0:{}'.format(m0.weight.data.shape))
            # print('w1:{}'.format(w1.shape))
            # print('idx1:{}'.format(idx1.shape))
            # print('type_w1:{}'.format(type(w1)))
            # print('m0_groups:{}'.format(m0.groups))
            # return
            if m0.groups == 1 and layer_id_in_cfg < len(cfg_mask) and layer_id_in_cfg > 2:
                w1 = w1[:, idx0.tolist(), :, :]
            if layer_id_in_cfg < len(cfg_mask) - 1 and layer_id_in_cfg > 1:  # do not change in Final FC
                w1 = w1[idx1.tolist(), :, :, :]
            m1.weight.data = torch.from_numpy(w1).cuda()

            # layer_id_in_cfg += 1
            # start_mask = end_mask
            # if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
            #     end_mask = cfg_mask[layer_id_in_cfg]
        elif isinstance(m0, nn.Linear):
            # if layer_id_in_cfg == len(cfg_mask):
            #     temp = np.asarray(cfg_mask[-1].cpu().numpy()).reshape(-1, 1)
            #     idx0 = np.zeros((len(temp), 16), dtype=np.int)
            #     idx0[:] = temp[:]
            #     idx0 = idx0.reshape(1, -1)
            #     idx0 = np.squeeze(np.argwhere(np.squeeze(idx0)))
            #     if idx0.size == 1:
            #         idx0 = np.resize(idx0, (1,))
            #     m1.weight.data = m0.weight.data[:, idx0].clone()
            #     m1.bias.data = m0.bias.data.clone()
            #     layer_id_in_cfg += 1
            #     continue
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
        elif isinstance(m0, nn.BatchNorm1d):
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

    return newmodel

new_model = pruning_step(model, 2)
# i = 0
# for [p0, p1] in zip(model.parameters(), new_model.parameters()):
#     i += 1
#     # if i == 158:
#     print(p0.shape, p1.shape)
# print(i)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(new_model.parameters(), args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
for i in range(5):
    fine_tuning(new_model)
validate(new_model)



