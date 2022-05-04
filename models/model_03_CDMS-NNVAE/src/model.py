import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np

class Network(nn.Module):
    def __init__(self, Layers, device, activation, 
                 end_in_sigmoid = False,
                 add_dropout = False,
                 add_batchnorm = False):
        super(Network, self).__init__()
        self.Layers = Layers
        self.device = device
        self.activation = activation
        self.end_in_sigmoid = end_in_sigmoid
        self.add_dropout = add_dropout,
        self.add_batchnorm = add_batchnorm,
        self.nn = self.build_model().to(device)
        
    def build_model(self):
        Seq = nn.Sequential()
        for ii in range(len(self.Layers)-1):
            this_module = nn.Linear(self.Layers[ii], self.Layers[ii+1])
            Seq.add_module("Linear" + str(ii), this_module)
            if not (ii == len(self.Layers)-2):
                if self.add_batchnorm[0]:
                    Seq.add_module("BatchNorm" + str(ii), nn.BatchNorm1d(self.Layers[ii+1]))
                Seq.add_module("Activation" + str(ii), self.activation)
                if self.add_dropout[0]:
                    Seq.add_module("Dropout" +str(ii), nn.Dropout(p=0.5, inplace=False))
            if self.end_in_sigmoid:
                Seq.add_module("Sigmoid" + str(ii), nn.Sigmoid())
        return Seq
    
    def forward(self, X):
        X = X.to(self.device)
        return self.nn(X)
    
class CDMS_NNVAE:
    def __init__(self, XY_train, XY_test, enc_layers, dec_layers, device, activation=nn.Sigmoid()):
        self.XY_train = torch.tensor(XY_train, dtype = torch.float32, device = device)
        self.XY_test  = torch.tensor(XY_test,  dtype = torch.float32, device = device)
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.device = device
        self.Encoder = Network(self.enc_layers, device, activation,
                              end_in_sigmoid = False, 
                              add_dropout = False, 
                              add_batchnorm = False)
        self.Decoder = Network(self.dec_layers, device, activation)
        
    def Evaluator(self, xy, lossfn):
        enc_output = self.Encoder.forward(xy).to(self.device)
        mu, logvar = enc_output[:, :int(self.enc_layers[-1]/2)], enc_output[:, int(self.enc_layers[-1]/2):]
        #print(mu.shape, logvar.shape)
        z = mu + torch.exp(logvar/2)*torch.randn_like(mu) # reparameterization
        z = z.reshape((-1, int(self.enc_layers[-1]/2)))
        #print(z.shape)
        dec_output = self.Decoder(z).to(self.device)
                
        mse_loss = torch.sum((dec_output.reshape(-1) - xy.reshape(-1))**2)/xy.shape[0]
        KL_div = -0.5 * torch.sum(1 + logvar - mu**2 - torch.exp(logvar))/xy.shape[0]
        
        return mu, logvar, z, dec_output, mse_loss, KL_div
    
    
    def Train(self, n_epochs, batch_size, lossfn=nn.MSELoss(reduction='mean')):
        Train_MSE_Losses = []
        Train_KL_Losses = []
        Test_MSE_Losses = []
        Test_KL_Losses = []
        params = list(self.Encoder.nn.parameters()) + list(self.Decoder.nn.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        max_loss = 9999.0
        for ii in range(n_epochs):
            self.Encoder.train()
            self.Decoder.train()
            DL = DataLoader(self.XY_train, batch_size=batch_size, shuffle=True, drop_last=True)
            total_loss = 0.
            for xy in DL:
                mu, logvar, z, dec_output, mse_loss, KL_div = self.Evaluator(xy, lossfn)
                optimizer.zero_grad()
                loss = mse_loss + KL_div
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if total_loss/DL.__len__() < max_loss:
                max_loss = total_loss/DL.__len__()
                self.Save("../trained-model.pt", self.Encoder, self.Decoder, optimizer, ii, max_loss)
                
            self.Encoder.eval()    
            self.Decoder.eval()
            with torch.no_grad():
                mu, logvar, z, dec_output, mse_loss, KL_div = self.Evaluator(self.XY_train, lossfn)
                Train_MSE_Losses.append(mse_loss.item())
                Train_KL_Losses.append(KL_div.item())
                
                mu, logvar, z, dec_output, mse_loss, KL_div = self.Evaluator(self.XY_test, lossfn)
                Test_MSE_Losses.append(mse_loss.item())
                Test_KL_Losses.append(KL_div.item())
                
                if ii % int(n_epochs/10) == 0 or ii == n_epochs - 1:
                    print('''Epoch {0}/{1}, 
                        Training MSE Loss = {2:0.4f} 
                        Training KL Div   = {3:0.4f}
                        Test MSE Loss     = {4:0.4f}
                        Test KL div       = {5:0.4f}
                        '''.format(ii, n_epochs, 
                                   Train_MSE_Losses[-1], Train_KL_Losses[-1],
                                   Test_MSE_Losses[-1],  Test_KL_Losses[-1]
                                  )
                         )
            
        
        return Train_MSE_Losses, Train_KL_Losses, Test_MSE_Losses, Test_KL_Losses
    
    
    def Save(self, fname, encoder, decoder, optimizer, epoch, loss):
        print("Saving Model on epoch {}".format(epoch))
        torch.save({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, fname)

        
