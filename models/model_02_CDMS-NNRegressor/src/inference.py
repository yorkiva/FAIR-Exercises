import numpy as np
import torch
from torch import nn
from matplotlib import pyplot as plt
from DataUtils import *
from model import *

## Getting the data

DL = CDMS_DataLoader(loc='../data/CDMS_Dataset.csv', sep=',')
XY_train, XY_valid, XY_test = DL.split_data('random')
XY_train, XY_valid, XY_test, x_mean, x_std = DL.normalize_data(XY_train, XY_valid, XY_test)

## Building the Model

NhiddenLayers = 1
NNodes = 32
Layers = [DL.n_features] + NhiddenLayers*[NNodes] + [1]
lossfn = nn.MSELoss()
device = torch.device('cpu')
activation = nn.LeakyReLU(negative_slope=0.01)
n_epochs = 100
batch_size = 128

NNreg = CDMS_Regressor(Layers = Layers, 
                      XY_train = XY_train, 
                      XY_test = XY_test, 
                      XY_valid = XY_valid,
                      device = device, 
                      activation=activation)

checkpoint = torch.load("../trained-model.pt")
NNreg.MLP.load_state_dict(checkpoint['model_state_dict'])
NNreg.MLP.eval()

plt.figure()
y_orig = NNreg.XY_test[:,-1].cpu().detach().numpy()
y_pred = NNreg.predict(NNreg.XY_test[:,:-1]).reshape(-1).cpu().detach().numpy()
plt.scatter(y_orig, y_pred, label='Test data')
y_orig_2 = NNreg.XY_train[:,-1].cpu().detach().numpy()
y_pred_2 = NNreg.predict(NNreg.XY_train[:,:-1]).reshape(-1).cpu().detach().numpy()
plt.scatter(y_orig_2, y_pred_2, label='Training data')
y_orig_2 = NNreg.XY_valid[:,-1].cpu().detach().numpy()
y_pred_2 = NNreg.predict(NNreg.XY_valid[:,:-1]).reshape(-1).cpu().detach().numpy()
plt.scatter(y_orig_2, y_pred_2, label='Validation data')
plt.plot(y_orig_2, y_orig_2, color='k')
plt.legend()

