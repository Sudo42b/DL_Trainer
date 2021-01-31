import torch
from enum import Enum
import os
class Hyperparameter(object):
    
    # Hyperparameters
    RANDOM_SEED = 1

    # Architecture
    NUM_FEATURES = 480*270
    NUM_CLASSES = 6
    BATCH_SIZE = 64*torch.cuda.device_count()
    INPUT_CHANNEL = 3
    GRAYSCALE = False

    #### DATA PARALLEL END ####
    PATIENCE = 100 # early stopping
    LEARNING_RATE = 1e-3 #0.00007
    NUM_EPOCHS = 2000
    
    #CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Tensorboard Log Directory
    log_dir = './logs/'


if __name__ == "__main__":
    print(Hyperparameter.RANDOM_SEED)
