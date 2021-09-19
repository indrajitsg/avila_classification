# Neural Network Model for Avila Data in PyTorch
import sys
sys.path.insert(0, 'C:/tmp/avila_classification/')

import copy
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import torch
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from scripts.model import AvilaDataset
from scripts.model import AvilaNNModelV1


# Get data loaders
def get_dataloaders(logging, batch_size):
    """Function to return data loaders"""
    # Load Data
    logging.info("Reading Train and Test data...")
    train_df = pd.read_csv("C:/tmp/avila_classification/data/avila-tr.txt", header=None)
    test_df = pd.read_csv("C:/tmp/avila_classification/data/avila-ts.txt", header=None)

    # Fix column names
    col_names = ['col_' + str(j + 1) for j in range(train_df.shape[1] - 1)]
    indep_cols = col_names.copy()
    col_names.append('y')

    logging.debug("Assigning columns")
    train_df.columns = col_names
    test_df.columns = col_names

    # Encode dependent variable column
    le = LabelEncoder()
    le.fit(train_df['y'])
    logging.debug(f"Classes: {le.classes_}")
    logging.debug(f"Transformed Classes: {le.transform(le.classes_)}")

    train_df['y_enc'] = le.transform(train_df['y'])
    test_df['y_enc'] = le.transform(test_df['y'])

    # train_df.head()
    logging.debug(f"Shape of train data: {train_df.shape}")
    logging.debug(f"Shape of test data: {test_df.shape}")

    # Create train and validation dataloaders
    train_ds = AvilaDataset(data_frame=train_df, indep_cols=indep_cols, dep_col='y_enc')
    valid_ds = AvilaDataset(data_frame=test_df, indep_cols=indep_cols, dep_col='y_enc')

    # Should be some exponent of 2 (128, 256)
    # batch_size = 256
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)

    return train_dl, valid_dl, le


# Training Function
def train(dataloader, model, loss_fn, optimizer, device):
    # model.train() # Having this line prevents my model accuracy going beyond 70%
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_loss = 0
    correct = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y = y.type(torch.LongTensor)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Compute prediction error
        with torch.set_grad_enabled(True):
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
        
        # Statistics
        train_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    
    train_loss /= num_batches
    correct /= size
    return 100*correct, train_loss


# Inference
def test(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y = y.type(torch.LongTensor)

        # Zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            pred = model(X)
            loss = loss_fn(pred, y)

        test_loss += loss.item() * X.size(0)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()  
            
    test_loss /= num_batches
    correct /= size
    return 100*correct, test_loss


# Check for autostop
def autostop():
    """Check if autostop.txt has value 1 to stop the training midway"""
    