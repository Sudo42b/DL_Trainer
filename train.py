# -*- coding: utf-8 -*-
"""
Created on Sun Jan 16 22:00:00 2021

@author: Lee SeonWoo
"""
#Pytorch Package
import torch
from torch import nn, optim
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
#Learning Rate Scheduler
from torch.optim import lr_scheduler
#Tensorboard
from torch.utils.tensorboard import SummaryWriter

#Utility Package
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import precision_score, recall_score
#Custom Package
from utils.config import Hyperparameter
from utils.air_dataloader import AirDataSet as DataLoader

from utils.utils import plot_loss_graph, \
                        get_performance, \
                        plot_confusion_matrix,\
                        get_probability_distribution,\
                        f1_score,\
                        accuracy_fn,\
                        get_network
#Custom Trainer
from utils.trainer import Trainer
from utils.lr_scheduler import CosineAnnealingWarmupRestarts
from torchvision import models

time = datetime.now()
time = time.strftime("%y%m%d%H%M")

log_path = os.path.join(Hyperparameter.log_dir, time)
batch_size = Hyperparameter.BATCH_SIZE
input_channels = Hyperparameter.INPUT_CHANNEL
num_classes = Hyperparameter.NUM_CLASSES
lr = Hyperparameter.LEARNING_RATE
num_epochs = Hyperparameter.NUM_EPOCHS


dataset = DataLoader('./dataset/', 
                  batch_size= batch_size,
                  n_classes= num_classes,
                  transforms=Hyperparameter.TRAIN_TRANSFORMS)

torch.manual_seed(Hyperparameter.RANDOM_SEED)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('-net', type=str, required=True, help='net type')
parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
parser.add_argument('-nc', type=int, default=5, help='number of class')
parser.add_argument('-sz', type=int, default=3, help='input channel size')
parser.add_argument('-cs', type=int, default=3, help='input channel size')
args = parser.parse_args()
#net = get_network(args, use_gpu=args.gpu)
net = models.resnet50(pretrained=True)
num_ftrs = net.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
net.fc = nn.Linear(num_ftrs, 2)
net.to(Hyperparameter.device)
#### DATA PARALLEL START ####
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs")
    net = nn.DataParallel(net)

# Create writer to store values 

if not os.path.exists(log_path):
    # remove if it already exists
    os.mkdir(log_path)
    #os.remove(log_path)
writer = SummaryWriter(log_dir=log_path)


#class_weights = {(f'{k}',f'{v:.4f}')for k, v in dataset.get_weight().items()}
class_weights = dataset.get_weight()
# config = {'kernel_size' : kernel_size, 
#             'depth_step' : depth_step, 
#             'batch_size' : batch_size, 
#             'Optimizer' : 'Adam', 
#             'lr' : lr}

# json_path = os.path.join(log_path,'config.json')
# with open(json_path, 'w') as outfile:
#     json.dump(config, outfile)


weights = torch.Tensor([class_weights[key] \
                        for key in sorted(class_weights.keys())]).to(Hyperparameter.device)
# loss_fn = nn.NLLLoss(weights)
if Hyperparameter.NUM_CLASSES > 1:
    loss_fn = nn.CrossEntropyLoss(weights)
else:
    loss_fn = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(net.parameters(),lr = lr)
scheduler = scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps= Hyperparameter.NUM_EPOCHS//2,
                                          cycle_mult=0.5,
                                          max_lr= Hyperparameter.LEARNING_RATE,
                                          min_lr=0.0,
                                          warmup_steps=1,
                                          gamma=0.5)

train_set= DataLoader('./dataset', batch_size= Hyperparameter.BATCH_SIZE,
                        use='train', 
                        n_classes=Hyperparameter.NUM_CLASSES,
                        transforms=Hyperparameter.TRAIN_TRANSFORMS)
val_set= DataLoader('./dataset', batch_size= Hyperparameter.BATCH_SIZE,
                    use='valid',
                    n_classes=Hyperparameter.NUM_CLASSES,
                    transforms=Hyperparameter.VALID_TRANSFORMS)
test_set= DataLoader('./dataset', batch_size= Hyperparameter.BATCH_SIZE,
                     use='valid',
                    n_classes=Hyperparameter.NUM_CLASSES,
                    transforms=Hyperparameter.VALID_TRANSFORMS) 
# Train
trainer = Trainer(train_set= train_set, val_set= val_set, test_set= test_set, 
                    model= net, 
                    optimizer= optimizer, 
                    scheduler= scheduler, 
                    num_classes = num_classes,
                    loss_fn= loss_fn, 
                    accuracy_fn= accuracy_fn, 
                    patience= Hyperparameter.PATIENCE, 
                    writer=writer, 
                    save_path=os.path.join(log_path,'best_model.pt'),
                    device= Hyperparameter.device)

train_loss, train_acc, train_pre, train_rec, train_f1,\
    val_loss, val_acc, val_pre, val_rec, val_f1, \
    best_val_loss = trainer.train_loop(num_epochs=Hyperparameter.NUM_EPOCHS)

plot_loss_graph(train_loss=train_loss, 
                train_acc=train_acc, 
                train_pre=train_pre,
                train_rec=train_rec,
                train_f1=train_f1,
                val_loss=val_loss, 
                val_acc=val_acc,
                val_pre= val_pre,
                val_rec=val_rec,
                val_f1= val_f1,
                save_path=os.path.join(log_path, 'result.png'))

max_acc = 0


# Evaluation
test_loss, test_acc, test_pre, test_rec, test_f1, y_preds, y_targets = trainer.test_loop()
print (f"test_loss: {test_loss:.3f}, \
        test_acc: {test_acc:.3f}, test_pre: {test_pre:.3f},\
        test_rec: {test_rec:.3f}, test_f1: {test_f1:.3f}")

writer.close()
torch.save(net.state_dict(), os.path.join(log_path,'last_model.pt'))
