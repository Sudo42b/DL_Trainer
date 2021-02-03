import torch
from enum import Enum
import os
from torchvision import transforms
class Hyperparameter(object):
    
    # Hyperparameters
    RANDOM_SEED = 1

    # Architecture
    NUM_CLASSES = 2
    BATCH_SIZE = 2*torch.cuda.device_count()
    INPUT_CHANNEL = 3
    GRAYSCALE = False

    #### DATA PARALLEL END ####
    PATIENCE = 100 # early stopping
    LEARNING_RATE = 1e-3 #0.00007
    NUM_EPOCHS = 200
    
    #CPU or GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Tensorboard Log Directory
    log_dir = './logs/'
    TRAIN_TRANSFORMS = transforms.Compose([transforms.ToTensor(),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.Normalize((0.9693, 0.8982, 0.9621),
                                                                (0.1125, 0.2896, 0.1358))])
    VALID_TRANSFORMS = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.9696, 0.8998, 0.9626),
                                                                (0.1129, 0.2878, 0.1357))])
    
    


if __name__ == "__main__":
    print(Hyperparameter.RANDOM_SEED)
