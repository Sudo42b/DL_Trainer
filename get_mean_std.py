'''Get dataset mean and std with PyTorch.'''
from __future__ import print_function

import logging
from datetime import datetime
from copy import deepcopy
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


import os
import argparse
import numpy as np
import models
import utils
import time
from dataset import VIBETest
from dataset import VIBETrain
from dataset import VIBEF_Test
from dataset import VIBEF_Train
from torchvision import transforms, datasets

# from models import *
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset')
# parser.add_argument('--batch_size', default='200', type=int, help='dataset')

# args = parser.parse_args()

def mean__std(data_loader):
    cnt = 0
    mean = torch.empty(3)
    std = torch.empty(3)

    for data, _, _ in data_loader:

        b, c, h, w = data.shape
        
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * mean + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * std + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return mean, torch.sqrt(std - mean ** 2)

if __name__ == '__main__':
        
        
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    # train_dataset = VIBETrain(os.path.join('./data/vibe'), 
    #                           data_type='mfcc',
    #                           transform=transform_train)
    train_dataset = VIBETest(os.path.join('./data/vibe'), data_type='mfcc', transform=transform_train)
    
    training_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, num_workers=1, batch_size=1)
    print('%d training samples.' % len(train_dataset))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    h, w = 0, 0
    ###mean
    # mean, std = mean__std(training_loader)
    # print('mean: {}, std: {}'.format(mean, std))
    
    ###
    
    for batch_idx, (inputs) in enumerate(training_loader):
        
        inputs = inputs[0].to(device)
        if batch_idx == 0:
            h, w = inputs.size(2), inputs.size(3)
            print(inputs.min(), inputs.max())
            chsum = inputs.sum(dim=(0, 2, 3), keepdim=True)
            #chsum = inputs.sum(dim=0, keepdim=True)
        else:
            chsum += inputs.sum(dim=(0, 2, 3), keepdim=True)
            #chsum += inputs.sum(dim=0, keepdim=True)
    mean = chsum/len(train_dataset)/h/w
    print('mean: %s' % mean.view(-1))
    
    chsum = None
    for batch_idx, (inputs) in enumerate(training_loader):
        inputs = inputs[0].to(device)
        if batch_idx == 0:
            chsum = (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
            #chsum = inputs.sum(dim=0, keepdim=True)
        else:
            chsum += (inputs - mean).pow(2).sum(dim=(0, 2, 3), keepdim=True)
            #chsum += inputs.sum(dim=0, keepdim=True)
    std = torch.sqrt(chsum/(len(train_dataset) * h * w - 1))
    print('std: %s' % std.view(-1))

    print('Done!')