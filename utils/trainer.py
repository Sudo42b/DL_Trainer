
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
class Trainer(object):

    def __init__(self, **kwargs):
        self.__dict__ = kwargs
        #self.patience = 0 
        
    def train_loop(self, num_epochs):
        """Training and validation steps."""
        # Metrics
        self.train_loss = []
        self.train_acc = []
        self.train_pre = []
        self.train_rec = []
        self.train_f1 = []
        
        self.val_loss = []
        self.val_acc = []
        self.val_pre = []
        self.val_rec = []
        self.val_f1 = []
        
        best_val_loss = np.inf
        patience = 0
        
        
        # Epochs
        tbar = tqdm(range(num_epochs), position=0)
        
        for epoch in tbar:
            # Steps
            self.train_step(epoch)
            self.val_step(epoch)
            msg = (f"t_a|p|r|f: {self.train_acc[-1]:.3f}| "
                            f" {self.train_pre[-1]:.3f}| "
                            f" {self.train_rec[-1]:.3f}| "
                            f" {self.train_f1[-1]:.3f}, "
                    f"v_a|p|r|f: {self.val_acc[-1]:.3f}| "
                            f" {self.val_pre[-1]:.3f}| "
                            f" {self.val_rec[-1]:.3f}| "
                            f" {self.val_f1[-1]:.3f}")

            tbar.set_description(f"{msg}, Epo: {epoch} | t_loss: {self.train_loss[-1]:.3f}, v_loss: {self.val_loss[-1]:.3f}")
            
            # Early stopping
            if self.val_loss[-1] < best_val_loss:
                best_val_loss = self.val_loss[-1]
                patience = 0 # reset patience
                torch.save(self.model.state_dict(), self.save_path)
                
            else:
                patience += 1
            
            if self.patience < patience: # 0
                print ("Stopping early!")
                break
        return self.train_loss, self.train_acc, self.train_pre, self.train_rec, self.train_f1, \
                self.val_loss, self.val_acc, self.val_pre, self.val_rec, self.val_f1, best_val_loss

    def train_step(self, epoch):
        """Training one epoch."""
        # Set model to train mode
        self.model.train()

        # Reset batch metrics
        running_train_loss = 0.0
        running_train_acc = 0.0
        running_train_pre = 0.0
        running_train_rec = 0.0
        running_train_f1 = 0.0

        # Iterate over train batches
        for i, (X, y) in enumerate(self.train_set.generate_batches()):

            # Set device
            X = X.to(self.device)
            y = y.to(self.device)

            # Forward pass
            y_pred = self.model(X)
            
            loss = self.loss_fn(y_pred, y)

            # Backward pass + optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            #Prediction
            if self.num_classes > 1:
                predictions = y_pred.max(dim=1)[1] # class
            else:
                y_pred = torch.sigmoid(y_pred)
                predictions = (y_pred>0.5).float().detach().cpu()
                
            # Metrics
            #accuracy, precision, recall, f1 = self.accuracy_fn(y_pred=predictions, y_true=y)
            accuracy = accuracy_score(y_pred = predictions.detach().cpu().numpy(), 
                                        y_true=y.detach().cpu())
            precision = precision_score(y_pred = predictions.detach().cpu().numpy(), 
                                        y_true=y.detach().cpu())
            recall = recall_score(y_pred = predictions.detach().cpu().numpy(), 
                                    y_true=y.detach().cpu())
            f1 = f1_score(y_pred = predictions.detach().cpu().numpy(), 
                            y_true=y.detach().cpu())
            
            
            # Update batch metrics
            running_train_loss = (loss + i*running_train_loss) / (i + 1)
            running_train_acc = (accuracy + i*running_train_acc) / (i + 1)
            running_train_pre = (precision + i*running_train_pre) / (i + 1)
            running_train_rec = (recall + i*running_train_rec) / (i + 1)
            running_train_f1 = (f1 + i*running_train_f1) / (i + 1)
        
        # Update epoch metrics
        self.train_loss.append(running_train_loss)
        self.train_acc.append(running_train_acc)
        self.train_pre.append(running_train_pre)
        self.train_rec.append(running_train_rec)
        self.train_f1.append(running_train_f1)

        # Write to TensorBoard
        self.writer.add_scalar(tag='training loss', scalar_value=running_train_loss, global_step=epoch)
        self.writer.add_scalar(tag='training accuracy', scalar_value=running_train_acc, global_step=epoch)
        self.writer.add_scalar(tag='training precision', scalar_value=running_train_pre, global_step=epoch)
        self.writer.add_scalar(tag='training recall', scalar_value=running_train_rec, global_step=epoch)
        self.writer.add_scalar(tag='training f1', scalar_value=running_train_f1, global_step=epoch)

    def val_step(self, epoch):
        """Validate one epoch."""
        # Set model to eval mode
        self.model.eval()

        # Reset batch metrics
        running_val_loss = 0.0
        running_val_acc = 0.0
        running_val_pre = 0.0
        running_val_rec = 0.0
        running_val_f1 = 0.0

        # Iterate over val batches
        for i, (X, y) in enumerate(self.val_set.generate_batches()):

            # Set device
            X = X.to(self.device)
            y = y.to(self.device)

            # Forward pass
            with torch.no_grad():
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)
                
            #Prediction
            if self.num_classes > 1:
                predictions = y_pred.max(dim=1)[1] # class
            else:
                y_pred = torch.sigmoid(y_pred)
                predictions = (y_pred>0.5).float()
            # Metrics
            #accuracy, precision, recall, f1 = self.accuracy_fn(y_pred=predictions, y_true=y)
            accuracy = accuracy_score(y_pred = predictions.detach().cpu().numpy(), 
                                        y_true=y.detach().cpu().numpy())
            precision = precision_score(y_pred = predictions.detach().cpu().numpy(),
                                        y_true=y.detach().cpu().numpy())
            recall = recall_score(y_pred = predictions.detach().cpu().numpy(),
                                    y_true=y.detach().cpu().numpy())
            f1 = f1_score(y_pred = predictions.detach().cpu().numpy(),
                            y_true=y.detach().cpu().numpy())
            
            # Update batch metrics
            running_val_loss = (loss + i*running_val_loss) / (i + 1)
            running_val_acc = (accuracy + i*running_val_acc) / (i + 1)
            running_val_pre = (precision + i*running_val_pre) / (i + 1)
            running_val_rec = (recall + i*running_val_rec) / (i + 1)
            running_val_f1 = (f1 + i*running_val_f1) / (i + 1)

        # Update epoch metrics
        self.val_loss.append(running_val_loss)
        self.val_acc.append(running_val_acc)
        self.val_pre.append(running_val_pre)
        self.val_rec.append(running_val_rec)
        self.val_f1.append(running_val_f1)

        # Write to TensorBoard
        self.writer.add_scalar(tag='validation loss', scalar_value=running_val_loss, global_step=epoch)
        self.writer.add_scalar(tag='validation accuracy', scalar_value=running_val_acc, global_step=epoch)
        self.writer.add_scalar(tag='validation precision', scalar_value=running_val_pre, global_step=epoch)
        self.writer.add_scalar(tag='validation recall', scalar_value=running_val_rec, global_step=epoch)
        self.writer.add_scalar(tag='validation f1', scalar_value=running_val_f1, global_step=epoch)

        # Adjust learning rate
        self.scheduler.step()

    def test_loop(self):
        """Evalution of the test set."""
        # Metrics
        running_test_loss = 0.0
        running_test_acc = 0.0
        running_test_pre = 0.0
        running_test_rec = 0.0
        running_test_f1 = 0.0
        
        y_preds = []
        y_targets = []

        # Iterate over val batches
        for i, (X, y) in enumerate(self.test_set.generate_batches()):

            # Set device
            X = X.to(self.device)
            y = y.to(self.device)

            # Forward pass
            with torch.no_grad():
                y_pred = self.model(X)
                loss = self.loss_fn(y_pred, y)

            # Metrics
            predictions = y_pred.max(dim=1)[1] # class
            #accuracy, precision, recall, f1 = self.accuracy_fn(y_pred=predictions, y_true=y)
            accuracy = accuracy_score(y_pred = predictions.detach().cpu().numpy(), 
                                        y_true=y.detach().cpu().numpy())
            precision = precision_score(y_pred = predictions.detach().cpu().numpy(), 
                                        y_true=y.detach().cpu().numpy())
            recall = recall_score(y_pred = predictions.detach().cpu().numpy(), 
                                        y_true=y.detach().cpu().numpy())
            f1 = f1_score(y_pred = predictions.detach().cpu().numpy(), 
                                        y_true=y.detach().cpu().numpy())

            # Update batch metrics
            running_test_loss = (loss + i*running_test_loss) / (i + 1)
            running_test_acc = (accuracy + i*running_test_acc) / (i + 1)
            running_test_pre = (precision + i*running_test_pre) / (i + 1)
            running_test_rec = (recall + i*running_test_rec) / (i + 1)
            running_test_f1 = (f1 + i*running_test_f1) / (i + 1)

            # Store values
            y_preds.extend(predictions.cpu().numpy())
            y_targets.extend(y.cpu().numpy())

        return running_test_loss, \
                running_test_acc, \
                running_test_pre, \
                running_test_rec, \
                running_test_f1, \
                y_preds, y_targets