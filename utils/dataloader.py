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

class SleepData(Dataset):
    """AirQuality dataset."""
    def __init__(self, 
                root, 
                batch_size, 
                n_classes = 6,
                split_length=720):
        self.n_classes = n_classes
        self.batch_size = batch_size
        try:
            self.json_files = [i for i in glob.iglob(os.path.join(root, '*')) if i.endswith(".png")]
        except:
            print('The path to the synthetic data does not exist')
            raise EOFError
        
        self.root_dir = root
        
        
        if len(self.json_files) < 1:
            print('No json files were found in the path')
            return None
        self.split_length = split_length
        self.X = []
        self.y = []
        
        for file in self.json_files:
            with open(file, 'r') as f:
                json_f = json.load(f)
                json_f = json_f['data']
                
                
                
                #Preprocessing
                data = [json_f['SO2'], json_f['SO2_MAX'], json_f['SO2_MIN']]
                list_len = len(data)
                data = np.array(data).reshape(-1, 1)
                data = MinMaxScaler().fit_transform(data)
                data = data.reshape(list_len, -1).tolist()
                split_list = self.split_chunk(stacked_air=data)
                
                self.X.extend(split_list)
                self.y.extend(self.split_chunk(stacked_air=[json_f['SO2_CODE']]))
                
        self.X = np.array(self.X)
        self.y = np.array(self.y)
        self.y = np.where(self.y==3, 1, 0)
        
    def split_chunk(self, stacked_air:list):
        stacked_air = np.vstack(stacked_air)
        #Split
        split_list = np.split(stacked_air, 
                                range(0, stacked_air.shape[1], 
                                    self.split_length), 
                                axis=1)
        
        #Remove residual length
        try:
            split_list = [i for i in split_list if i.shape[1] == self.split_length]
        except:
            print(split_list)
            exit()
        return split_list
    
    def __str__(self):
        return f"<Dataset(N={len(self)}, batch_size={self.batch_size}, num_batches={self.get_num_batches()})>"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

    def get_num_batches(self):
        return math.ceil(len(self)/self.batch_size)

    def collate_fn(self, batch):
        """Processing on a batch."""
        # Get inputs
        X = torch.FloatTensor([entry[0].astype(np.float32) for entry in batch])
        if self.n_classes == 1:
            y = torch.FloatTensor([entry[1].astype(np.int32) for entry in batch])
            # y = y.squeeze(1)
        else:
            y = torch.LongTensor([entry[1].astype(np.int32) for entry in batch])
            y = y.squeeze(1)
        
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
    a = SleepData(root='../dataset', batch_size=1)
    
    y_train = a.y

    #print(class_weights)