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
chkpoint_file = "avila_chkpoint2.pth"
use_scheduler = False
epochs = 100
learning_rate = 0.01
gamma = 0.1

# Set logging
logging.basicConfig(level=logging.INFO)

# Get data
train_dl, valid_dl, le = get_dataloaders(logging=logging, batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load checkpoint
logging.info(f"Loading checkpoint {chkpoint_file}...")
best_model = AvilaNNModelV1(10, 12)
checkpoint = torch.load(chkpoint_file)
best_model.load_state_dict(checkpoint['model'])
best_model.eval()
logging.debug(f"Loaded model from epoch {checkpoint['epochs']}")

# Get predicted values from the validation data loader using the best model
preds = []
best_model.eval()
with torch.no_grad():
    for X, y in valid_dl:
        X, y = X.to(device), y.to(device)
        out = best_model(X)        
        prob = F.softmax(out, dim=1)
        preds.append(prob)

# Concatenate the batch outputs of probabilities
for j in range(len(preds)):
    if j == 0:
        pred_array = preds[j]
    else:
        pred_array = torch.cat((pred_array, preds[j]), 0)
