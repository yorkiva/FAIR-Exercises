import numpy as np
import json
import torch
from torch import nn
from matplotlib import pyplot as plt
from DataUtils import *
from model import *

## Getting the data

DL = CDMS_DataLoader(loc='../data/CDMS_Dataset.csv', sep=',')
XY_train, XY_valid, XY_test = DL.split_data('label', [-12.502, -29.5, -41.9])
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

Train_Losses, Test_Losses, Validation_Losses = NNreg.Train(n_epochs = n_epochs,
                                                           batch_size = batch_size,
                                                           lossfn = lossfn)

## Making Plots
plt.figure()
plt.scatter(np.arange(1,n_epochs+1), Train_Losses, marker='x', label="Training Loss")
plt.scatter(np.arange(1,n_epochs+1), Validation_Losses, label="Validation Loss")
plt.scatter(np.arange(1,n_epochs+1), Test_Losses, label="Test Loss")
plt.xlabel("Epochs")
plt.ylabel("RMSE Loss")
plt.legend(fontsize=12)

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

## Save model metadata

model_metadata = {}

model_metadata['genInfo'] = '''
This set of models represents the trained parameters for a Linear Regression model for determining the impact location on a superCDMS prototype based on timing measurements. 
'''
model_metadata['data_ID']  = 'https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/data/processed_csv/processed_combined.csv?raw=true'
model_metadata['model_ID'] = ''
model_metadata['model_keys'] = {
    'epoch': 'The iteration ID when this model was saved',
    'loss': 'The value of training loss when this model was saved',
    'model_state_dict': 'parameters of the model',
    'optimizer_state_dict': 'parameters of the optimizer'
}
model_metadata['hyperparameters'] ={
    'n_epochs': 100,
    'batch_size': 128,
    'optimizer': 'Adam',
    'learning_rate': 1e-3,
    'n_hidden_layers': 1,
    'n_nodes_in hidden_layers': 32,
    'activation':'LeakyReLU',
    'activation_hyperparameter': 0.01
}

json.dump(model_metadata, open("../model-metadata.json","w"), indent=3)



