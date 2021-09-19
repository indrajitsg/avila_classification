import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Create PyTorch Dataset
class AvilaDataset(Dataset):
    def __init__(self, data_frame, indep_cols, dep_col):
        data_frame = data_frame.copy()
        self.x = data_frame.loc[:, indep_cols].copy().values.astype(np.float32)        
        self.y = data_frame.loc[:, dep_col].copy().values
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class AvilaNNModelV1(nn.Module):
    """
    Simple Neural Network model with a single hidden layer with batch normalization
    """
    def __init__(self, comps, classes):
        super().__init__()
        self.comps = comps
        self.classes = classes
        self.bn1 = nn.BatchNorm1d(self.comps)
        self.lin1 = nn.Linear(self.comps, 50)
        self.lin2 = nn.Linear(50, 70)
        self.lin3 = nn.Linear(70, self.classes)
    
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        logits = self.lin3(x)
        return logits


class AvilaNNModelV2(nn.Module):
    
    def __init__(self, comps, classes):
        super().__init__()
        self.comps = comps
        self.classes = classes
        self.bn1 = nn.BatchNorm1d(self.comps)
        self.lin1 = nn.Linear(self.comps, 50)
        self.drops = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(50)
        self.lin2 = nn.Linear(50, 70)
        self.bn3 = nn.BatchNorm1d(70)
        self.lin3 = nn.Linear(70, self.classes)
    
    def forward(self, x):
        x = self.bn1(x)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        logits = self.lin3(x)
        return logits
