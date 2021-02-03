# -*- coding: utf-8 -*-

"""
Created on Mon Nov  9 15:49:13 2020

@author: Lee SeonWoo
"""

import os
import math
import glob
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
import glob

from torchvision import transforms


class AirDataSet(Dataset):
    """Sleep dataset."""
    def __init__(self, 
                root, 
                batch_size, 
                use='train',
                transforms = None,
                n_classes = 2):
        self.batch_size = batch_size
        self.transform = transforms
        self.n_classes = n_classes
        normal = [[i, 0] for i in glob.iglob(os.path.join(root, 'normal', '**'), recursive=True) if i.endswith('.png') ]
        abnormal = [[i, 1] for i in glob.iglob(os.path.join(root, 'abnormal', '**'), recursive=True) if i.endswith('.png') ]
        tot_list = normal+abnormal
        self.y = [i[1] for i in tot_list]
        if use == 'train':
            self._list, _ = train_test_split(tot_list, test_size=0.2, random_state = 0, shuffle=True)
        elif use == 'test' or use=='valid':
            _, self._list = train_test_split(tot_list, test_size=0.2, random_state = 0, shuffle=True)

    def __str__(self):
        return f"<Dataset(N={len(self)}, batch_size={self.batch_size}, num_batches={self.get_num_batches()})>"

    def __len__(self):
        return len(self._list)

    def __getitem__(self, index):
        #X = self.X[index]
        #y = self.y[index]
        
        X = Image.open(self._list[index][0]).convert('RGB')
        
        y = self._list[index][1]
        if self.transform:
            X = self.transform(X)
            
        return X, y

    def get_num_batches(self):
        return math.ceil(len(self)/self.batch_size)

    def collate_fn(self, batch):
        """Processing on a batch."""
        # Get inputs
        
        X = torch.FloatTensor(np.array([np.asarray(entry[0]) for entry in batch]))
        
        if self.n_classes == 1:
            y = torch.FloatTensor(np.array([entry[1] for entry in batch]))
            # y = y.squeeze(1)
        else:
            y = torch.LongTensor(np.array([entry[1] for entry in batch]))
            
            # y = y.squeeze(1)
        
        return X, y

    def get_weight(self, verbose=True):
        # Class weights
        try:
            key, counts = np.unique(self.y, return_counts=True)
            class_weights = {key: 1-(1.0*count)/counts.sum() 
                                for key, count in zip(key, counts)}
        except:
            return None
        
        if verbose:
            print (f"class counts: {counts},\nclass weights: {class_weights}")
        return class_weights
    
    def generate_batches(self, 
                        shuffle=False, 
                        drop_last=False):
        dataloader = DataLoader(dataset=self, 
                                batch_size=self.batch_size, 
                                collate_fn=self.collate_fn, 
                                shuffle=shuffle, 
                                drop_last=drop_last)
        for (X, y) in dataloader:
            yield X, y
    
    
if __name__ == "__main__":
    a = AirDataSet(root='../dataset', batch_size=1, 
                transforms=transforms.Compose([transforms.ToTensor()]))
    
    print(len(a))
    