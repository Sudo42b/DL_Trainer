"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):
    def __init__(self, features, num_class=100):
        super().__init__()
        del features.avgpool
        del features.classifier
        self.features = features
        self.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #Average Pooling VGG
        self.features.classifier = nn.Sequential(
            nn.Linear(512, num_class)
        )
            
    def forward(self, x):
        
        output = self.features(x)
        
        # output = nn.AdaptiveAvgPool2d((1, 1))(output)
        # output = output.view(output.size(0), -1)
        # output = torch.flatten(output, 1)
        #https://www.programmersought.com/article/1959104726/
        #https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
        
        #global_average_pooling
        # output = nn.AdaptiveAvgPool2d((7, 7))(output)
        # print(output.shape)
        # output = self.avgpool(output)
        
        
        # output = output.view(output.size()[0], -1)
        
        # # output = output.view(-1, 512)
        # # print(output.size()[0])
        
        # output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False, channel_size = 3):
    layers = []

    input_channel = channel_size
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    
    return nn.Sequential(*layers)

def vgg11_bn(args):
    return VGG(make_layers(cfg['A'], batch_norm=True, channel_size = args.cs),num_class=args.nc)

def vgg13_bn(args):
    return VGG(make_layers(cfg['B'], batch_norm=True, channel_size = args.cs),num_class=args.nc)

def vgg16_bn(args):
    return VGG(make_layers(cfg['D'], batch_norm=True, channel_size = args.cs),num_class=args.nc)

def vgg19_bn(args):
    return VGG(make_layers(cfg['E'], batch_norm=True, channel_size = args.cs),num_class=args.nc)


