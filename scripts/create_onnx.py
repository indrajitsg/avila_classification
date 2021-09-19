# Convert PyTorch Model to ONNX format
import sys
sys.path.insert(0, 'C:/tmp/avila_classification/')
import logging

import torch
import torch.onnx

from scripts.utils import get_dataloaders
from scripts.model import AvilaNNModelV1

# Parameters
batch_size = 256
chkpoint_file = "models/avila_chkpoint.pth"

# Get data
train_dl, valid_dl, le = get_dataloaders(logging=logging, batch_size=batch_size)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load model
best_model = AvilaNNModelV1(10, 12)
checkpoint = torch.load(chkpoint_file)
best_model.load_state_dict(checkpoint['model'])
best_model.eval()

# create input to the model
i = 1
for X, y in valid_dl:
    X, y = X.to(device), y.to(device)
    if i == 1:
        break

obs = X[0]
obs = obs.view(1, 10)

best_model.eval()
with torch.no_grad():
    obs_out = best_model(obs)

# Export the model
torch.onnx.export(best_model,               # model being run
                  obs,                         # model input (or a tuple for multiple inputs)
                  "avila_nn1.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  verbose=True)
