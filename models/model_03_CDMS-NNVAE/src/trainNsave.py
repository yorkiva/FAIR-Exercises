import numpy as np
import json
import torch
from torch import nn
from matplotlib import pyplot as plt
from DataUtils import *
from model import *

## Getting the data

DL = CDMS_DataLoader(loc='../data/CDMS_Dataset.csv', sep=',')
XY_train, XY_test = DL.split_data('random')
y_norm = -41.9

## Building the Model

NLatent = 3
NNodes = 32
NhiddenLayers = 2
enc_Layers =  [DL.n_features+1] + NhiddenLayers*[NNodes] + [2*NLatent]
dec_Layers =  [NLatent]  + NhiddenLayers*[NNodes] + [DL.n_features+1]
lossfn = nn.MSELoss(reduction='mean')
device = torch.device('cpu')
activation = nn.Sigmoid()
n_epochs = 1000
batch_size = 128


NNvae =  CDMS_NNVAE(XY_train = XY_train, 
                    XY_test = XY_test, 
                    enc_layers = enc_Layers, 
                    dec_layers = dec_Layers,
                    device = device, 
                    activation = activation)

Train_MSE_Losses, Train_KL_Losses, Test_MSE_Losses, Test_KL_Losses = NNvae.Train(n_epochs = n_epochs,
                                                                                 batch_size = batch_size,
                                                                                 lossfn = lossfn)

## Getting inference for train and test data

mu, logvar, z, dec_output, mse_loss, KL_div = NNvae.Evaluator(NNvae.XY_train, lossfn)
mu2, logvar2, z2, dec_output2, mse_loss2, KL_div2 = NNvae.Evaluator(NNvae.XY_test, lossfn)

f1_org = NNvae.XY_train[:,-1].numpy()
f1_pred = dec_output[:,-1].detach().numpy()
f1_org2 = NNvae.XY_test[:,-1].numpy()
f1_pred2 = dec_output2[:,-1].detach().numpy()

f1_org = (f1_org + 1)*y_norm/2
f1_pred = (f1_pred + 1)*y_norm/2
f1_org2 = (f1_org2 + 1)*y_norm/2
f1_pred2 = (f1_pred2 + 1)*y_norm/2

## Making Plots
plt.figure()
plt.scatter(f1_org, f1_pred, label='Training Set')
plt.scatter(f1_org2, f1_pred2, label = 'Test Set')
plt.plot(f1_org, f1_org, color='k')
plt.xlabel("Original Impact Location")
plt.ylabel("Impact Location from Decoder")
plt.legend(fontsize=12)

y = NNvae.XY_train[:,-1].numpy()
z0 = z[:,0].detach().flatten().numpy()
z1 = z[:,1].detach().flatten().numpy()
z2 = z[:,2].detach().flatten().numpy()

plt.figure()
plt.scatter(y,z0,label='z0',marker='.', s=2)
plt.xlabel('Impact Location')
plt.ylabel('Latent Dimension 0')

plt.figure()
plt.scatter(y,z1,label='z1',marker='.', s=2)
plt.xlabel('Impact Location')
plt.ylabel('Latent Dimension 1')

plt.figure()
plt.scatter(y,z2,label='z2',marker='.', s=2)
plt.xlabel('Impact Location')
plt.ylabel('Latent Dimension 2')

## Save model metadata

model_metadata = {}

model_metadata['genInfo'] = '''
This models represents the trained parameters for a Variational Auto-Encoder for the superCDMS prototype dataset. 
'''
model_metadata['data_ID']  = 'https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/data/processed_csv/processed_combined.csv?raw=true'
model_metadata['model_ID'] = ''
model_metadata['model_keys'] = {
    'epoch': 'The iteration ID when this model was saved',
    'loss': 'The value of training loss when this model was saved',
    'encoder_state_dict': 'parameters of the Encoder',
    'decoder_state_dict': 'parameters of the Decoder',
    'optimizer_state_dict': 'parameters of the optimizer'
}
model_metadata['hyperparameters'] ={
    'n_epochs': n_epochs,
    'batch_size': batch_size,
    'optimizer': 'Adam',
    'learning_rate': 1e-3,
    'n_hidden_layers_encoder': NhiddenLayers,
    'n_hidden_layers_decoder': NhiddenLayers,
    'n_nodes_in hidden_layers_encoder': NNodes,
    'n_nodes_in hidden_layers_decoder': NNodes,
    'activation':'Sigmoid'
}

json.dump(model_metadata, open("../model-metadata.json","w"), indent=3)



