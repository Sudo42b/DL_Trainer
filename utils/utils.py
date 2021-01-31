# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 21:01:16 2021

@author: Lee SeonWoo
"""
import itertools
import torch
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from .dice_loss import dice_coeff
# Accuracy
def accuracy_fn(y_pred, y_true):
    n_correct = torch.eq(y_pred, y_true).sum().item()
    
    accuracy = (n_correct / (y_pred.view(-1, 1).shape[0]))
    # accuracy = dice_coeff(y_true, y_pred)
    f1, precision, recall = f1_score(y_true=y_true, y_pred=y_pred)
    
    return accuracy, precision, recall, f1

def f1_score(y_true:torch.Tensor, y_pred:torch.Tensor, is_training=False) -> torch.Tensor:
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    
    '''
    
    
    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return f1.item(), precision.item(), recall.item()

    
def detach_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_hidden(v) for v in h)
    
def get_chunks(inp, t, k, chunk_size):    
    """ Split into chunks """
    
    idx = torch.arange(k * chunk_size , (k+1) * chunk_size)
    
    inp = inp[:,:,idx]
    
    t = t[:,idx]
    return inp.float(), t


def train_val_test_split(X, y, val_size, test_size, shuffle):
    """Split data into train/val/test datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test

def model_performance(ground, pred, th):
    """ Custom way to quantify the performance of the peak detector"""
    
    TC, TP, FP, TC_p = 0, 0, 0, 0
    idx = np.arange(len(pred)-1)
    
    diff = np.diff(ground)
    a = diff > 0
    b = diff < 0
       
    if ground[0] == 1: a[0] = True
    if ground[-1] == 1: b[-1] = True
    if np.sum(a) > 0: 
        if np.sum(a) == np.sum(b):
            for l,n in zip(idx[a],idx[b]):
                if np.sum(pred[l:n]) / (n - l) > th:
                    TP += 1
                TC += 1
        else:
            print('Error during model performance check')
    
    diff = np.diff(pred)
    c = diff > 0
    d = diff < 0
    
    if pred[0] == 1: c[0] = True
    if pred[-1] == 1: d[-1] = True
    if np.sum(c) > 0: 
        if np.sum(c) == np.sum(d):
            for l,n in zip(idx[c],idx[d]):
                if np.sum(ground[l:n]) / (n - l) < th:
                    FP += 1
                TC_p += 1
        else:
            print('Error during model performance check')        
            
    return TP, TC, FP, TC_p


def plot_loss_graph(train_loss, train_acc, train_pre, train_rec, train_f1,
                    val_loss, val_acc, val_pre, val_rec, val_f1, save_path=None):
    # Plot performance
    plt.figure(figsize=(15,5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.legend(loc='upper right')

    # Plot Metric
    plt.subplot(1, 2, 2)
    plt.title("Metric")
    plt.plot(train_acc, label="train_acc")
    plt.plot(train_pre, label="train_pre")
    plt.plot(train_rec, label="train_rec")
    plt.plot(train_f1, label="train_f1")
    plt.plot(val_acc, label="val_acc")
    plt.plot(val_pre, label="val_pre")
    plt.plot(val_rec, label="val_rec")
    plt.plot(val_f1, label="val_f1")
    plt.legend(loc='lower right')

    # Show plots
    plt.show()
    if save_path:
        plt.savefig(save_path)
    
def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):
    """Plot a confusion matrix using ground truth and predictions."""
    
    plt.rcParams["figure.figsize"] = (7,7)
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #  Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Axis
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    ax.xaxis.set_label_position('bottom') 
    ax.xaxis.tick_bottom()

    # Values
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:d} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    print (classification_report(y_true, y_pred))
    # Display
    plt.show()
    
def get_performance(y_true, y_pred, classes):
    """Per-class performance metrics. """
    performance = {'overall': {}, 'class': {}}
    metrics = precision_recall_fscore_support(y_true, y_pred)

    # Overall performance
    performance['overall']['precision'] = np.mean(metrics[0])
    performance['overall']['recall'] = np.mean(metrics[1])
    performance['overall']['f1'] = np.mean(metrics[2])
    performance['overall']['num_samples'] = np.float64(np.sum(metrics[3]))

    # Per-class performance
    for i in range(len(classes)):
        performance['class'][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i])
        }

    return performance

def get_probability_distribution(y_prob, classes):
    results = {}
    for i, class_ in enumerate(classes):
        results[class_] = np.float64(y_prob[i])
    sorted_results = {k: v for k, v in sorted(
        results.items(), key=lambda item: item[1], reverse=True)}
    return sorted_results

def get_network(args, use_gpu=True):
    """ return given network
    """
    if args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(args)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(args)
    elif args.net == 'vgg13':
        from models.vgg import vgg13_bn
        net = vgg13_bn(args)
    elif args.net == 'vgg11':
        from models.vgg import vgg11_bn
        net = vgg11_bn(args)
    elif args.net == 'vgg19':
        # from models.vgg import vgg19_bn
        # net = vgg19_bn(args)
        from torchvision.models import vgg19_bn
        import torch.nn as nn
        net = vgg19_bn(pretrained=True)
        net.classifier[6]=nn.Linear(4096, args.nc)
    elif args.net == 'densenet121':
        from models.densenet import densenet121
        net = densenet121(args)
    elif args.net == 'densenet161':
        from models.densenet import densenet161
        net = densenet161(args)
    elif args.net == 'densenet169':
        from models.densenet import densenet169
        net = densenet169(args)
    elif args.net == 'densenet201':
        from models.densenet import densenet201
        net = densenet201(args)
    elif args.net == 'googlenet':
        from models.googlenet import googlenet
        net = googlenet(args)
    elif args.net == 'inceptionv3':
        from models.inceptionv3 import inceptionv3
        net = inceptionv3(args)
    elif args.net == 'inceptionv4':
        from models.inceptionv4 import inceptionv4
        net = inceptionv4(args)
    elif args.net == 'inceptionresnetv2':
        from models.inceptionv4 import inception_resnet_v2
        net = inception_resnet_v2(args)
    elif args.net == 'xception':
        from models.xception import xception
        net = xception(args)
    elif args.net == 'resnet18':
        # from models.resnet import resnet18
        # net = resnet18(args)
        from torchvision.models import resnet18
        import torch.nn as nn
        net = resnet18(pretrained=True)
        net.fc = nn.Linear(512, args.nc)
    elif args.net == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(args)
    elif args.net == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(args)
    elif args.net == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(args)
    elif args.net == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(args)
    elif args.net == 'preactresnet18':
        from models.preactresnet import preactresnet18
        net = preactresnet18(args)
    elif args.net == 'preactresnet34':
        from models.preactresnet import preactresnet34
        net = preactresnet34(args)
    elif args.net == 'preactresnet50':
        from models.preactresnet import preactresnet50
        net = preactresnet50(args)
    elif args.net == 'preactresnet101':
        from models.preactresnet import preactresnet101
        net = preactresnet101(args)
    elif args.net == 'preactresnet152':
        from models.preactresnet import preactresnet152
        net = preactresnet152(args)
    elif args.net == 'resnext50':
        from models.resnext import resnext50
        net = resnext50(args)
    elif args.net == 'resnext101':
        from models.resnext import resnext101
        net = resnext101(args)
    elif args.net == 'resnext152':
        from models.resnext import resnext152
        net = resnext152(args)
    elif args.net == 'shufflenet':
        from models.shufflenet import shufflenet
        net = shufflenet(args)
    elif args.net == 'shufflenetv2':
        from models.shufflenetv2 import shufflenetv2
        net = shufflenetv2(args)
    elif args.net == 'squeezenet':
        from models.squeezenet import squeezenet
        net = squeezenet(args)
        
    elif args.net == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(args)
    elif args.net == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        net = mobilenetv2(args)
    elif args.net == 'mobilenetv3':
        from models.mobilenetv3 import mobileNetv3
        net = mobileNetv3(args)
    elif args.net == 'mobilenetv3_l':
        from models.mobilenetv3 import mobileNetv3
        net = mobileNetv3(args, mode='large')
    elif args.net == 'mobilenetv3_s':
        from models.mobilenetv3 import mobileNetv3
        net = mobileNetv3(args, mode='small')
    elif args.net == 'nasnet':
        from models.nasnet import nasnetalarge
        net = nasnetalarge(args)
    elif args.net == 'attention56':
        from models.attention import attention56
        net = attention56(args)
    elif args.net == 'attention92':
        from models.attention import attention92
        net = attention92(args)
    elif args.net == 'seresnet18':
        from models.senet import seresnet18
        net = seresnet18(args)
    elif args.net == 'seresnet34':
        from models.senet import seresnet34
        net = seresnet34(args)
    elif args.net == 'seresnet50':
        from models.senet import seresnet50
        net = seresnet50(args)
    elif args.net == 'seresnet101':
        from models.senet import seresnet101
        net = seresnet101(args)
    elif args.net == 'seresnet152':
        from models.senet import seresnet152
        net = seresnet152(args)
    elif args.net.lower() == 'sqnxt_23_1x':
        from models.SqueezeNext import SqNxt_23_1x
        net = SqNxt_23_1x(args)
    elif args.net.lower() == 'sqnxt_23_1xv5':
        from models.SqueezeNext import SqNxt_23_1x_v5
        net = SqNxt_23_1x_v5(args)
    elif args.net.lower() == 'sqnxt_23_2x':
        from models.SqueezeNext import SqNxt_23_2x
        net = SqNxt_23_2x(args)
    elif args.net.lower() == 'sqnxt_23_2xv5':
        from models.SqueezeNext import SqNxt_23_2x_v5
        net = SqNxt_23_2x_v5(args)
    elif args.net.lower() == 'mnasnet':
        # from models.MnasNet import mnasnet
        # net = mnasnet(args)
        from models.nasnet_mobile import nasnet_Mobile
        net = nasnet_Mobile(args)
    elif args.net == 'efficientnet_b0':
        from models.efficientnet import efficientnet_b0
        net = efficientnet_b0(args)
    elif args.net == 'efficientnet_b1':
        from models.efficientnet import efficientnet_b1
        net = efficientnet_b1(args)
    elif args.net == 'efficientnet_b2':
        from models.efficientnet import efficientnet_b2
        net = efficientnet_b2(args)
    elif args.net == 'efficientnet_b3':
        from models.efficientnet import efficientnet_b3
        net = efficientnet_b3(args)
    elif args.net == 'efficientnet_b4':
        from models.efficientnet import efficientnet_b4
        net = efficientnet_b4(args)
    elif args.net == 'efficientnet_b5':
        from models.efficientnet import efficientnet_b5
        net = efficientnet_b5(args)
    elif args.net == 'efficientnet_b6':
        from models.efficientnet import efficientnet_b6
        net = efficientnet_b6(args)
    elif args.net == 'efficientnet_b7':
        from models.efficientnet import efficientnet_b7
        net = efficientnet_b7(args)
    elif args.net == 'mlp':
        from models.mlp import MLPClassifier
        net = MLPClassifier(args)
    elif args.net =='alexnet':
        from torchvision.models import alexnet
        import torch.nn as nn
        net = alexnet(pretrained=True)
        net.classifier[6]=nn.Linear(4096, args.nc)
    elif args.net =='lambda18':
        from models._lambda import LambdaResnet18
        net = LambdaResnet18(num_classes=args.nc,
                              channels=args.cs)
    elif args.net =='lambda34':
        from models._lambda import LambdaResnet34
        net = LambdaResnet34(num_classes=args.nc,
                              channels=args.cs)
    elif args.net =='lambda50':
        from models._lambda import LambdaResnet50
        net = LambdaResnet50(num_classes=args.nc,
                              channels=args.cs)
    elif args.net =='lambda101':
        from models._lambda import LambdaResnet101
        net = LambdaResnet101(num_classes=args.nc)
    elif args.net =='lambda152':
        from models._lambda import LambdaResnet152
        net = LambdaResnet152(num_classes=args.nc,
                              channels=args.cs)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    
    if use_gpu:
        net = net.cuda()
        
    return net