# Neural Network Model for Avila Data in PyTorch
import sys
sys.path.insert(0, 'C:/tmp/avila_classification/')

import copy
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from scripts.model import AvilaDataset
from scripts.model import AvilaNNModelV1
from scripts.utils import train
from scripts.utils import test
from scripts.utils import get_dataloaders

# Parameters
batch_size = 256
chkpoint_file = "../models/avila_chkpoint.pth"
use_scheduler = False
epochs = 100
learning_rate = 0.01
gamma = 0.1
log_file = "../logs/run1.log"

# Set logging
logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)


logging.info(f"Model Training using Python {sys.version} and PyTorch {torch.__version__}:")
logging.info(f"Batch Size: {batch_size}")
logging.info(f"Scheduler Use: {use_scheduler}")
logging.info(f"Total Epochs: {epochs}")
logging.info(f"Learning Rate: {learning_rate}")
logging.info(f"Gamma: {gamma}")
logging.info("")

# Get data
train_dl, valid_dl, le = get_dataloaders(logging=logging, batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate the model
logging.info("Instantiating model...")
model = AvilaNNModelV1(comps=10, classes=12)

# Send to device
model = model.to(device)

# Set loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Decay LR by a factor of 0.1 every 7 epochs
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training Loop
best_test_loss = 100000
best_train_loss = 100000
best_train_acc = 0
best_test_acc = 0
best_epoch = 0

logging.info("Training Starts...")
for t in range(epochs):
    train_acc, train_loss = train(train_dl, model, loss_fn, optimizer, device)
    test_acc, test_loss = test(valid_dl, model, loss_fn, optimizer, device)
    if test_acc > best_test_acc:
        # save model checkpoint
        best_epoch = t + 1
        torch.save({'epochs': best_epoch,
                    'loss': test_loss,
                    'accuracy': test_acc,
                    'model': copy.deepcopy(model.state_dict()),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()},
                    chkpoint_file)

        best_test_loss = test_loss
        best_train_loss = train_loss
        best_test_acc = test_acc
        best_train_acc = train_acc
    
    if use_scheduler:
        scheduler.step()
    
    logging.info(f"Epoch {t+1} Train Accuracy: {train_acc:>0.1f}%, Train Loss: {train_loss:>8f}, Test Accuracy: {test_acc:>0.1f}%, Test loss: {test_loss:>8f}")

logging.info("End of Training!")
logging.info("")
logging.info(f"Best Epoch: {best_epoch}")
logging.info(f"Best Test Loss: {best_test_loss}")
logging.info(f"Best Train Loss: {best_train_loss}")
logging.info(f"Best Test Accuracy: {best_test_acc}")
logging.info(f"Best Train Accuracy: {best_train_acc}")
logging.info(f"Saved Checkpoint: {chkpoint_file}")
