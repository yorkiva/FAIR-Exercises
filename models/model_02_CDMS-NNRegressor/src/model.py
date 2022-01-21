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
    
class CDMS_Regressor:
    def __init__(self, Layers, XY_train, XY_valid, XY_test, device, activation=nn.Sigmoid()):
        self.XY_train = torch.tensor(XY_train, dtype = torch.float32, device = device)
        self.XY_test  = torch.tensor(XY_test,  dtype = torch.float32, device = device)
        self.XY_valid = torch.tensor(XY_valid, dtype = torch.float32, device = device)
        self.device = device
        self.MLP = Network(Layers, device, activation, 
                           end_in_sigmoid = False,
                           add_dropout = True,
                           add_batchnorm = True)
        
    def predict(self,X):
        return self.MLP.forward(X).to(self.device)
    
    def Train(self, n_epochs, batch_size, lossfn):
        Train_Losses = []
        Validation_Losses = []
        Test_Losses = []
        params = list(self.MLP.nn.parameters())
        optimizer = optim.Adam(params, lr=1e-3)
        max_loss = 9999.0
        for ii in range(n_epochs):
            self.MLP.train()
            DL = DataLoader(self.XY_train, batch_size=batch_size, shuffle=True, drop_last=True)
            total_loss = 0.
            for xy in DL:
                optimizer.zero_grad()
                y_pred = self.predict(xy[:,:-1]).reshape(-1)
                loss = torch.sqrt(lossfn(y_pred, xy[:,-1].reshape(-1)))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if total_loss/DL.__len__() < max_loss:
                max_loss = total_loss/DL.__len__()
                self.Save("../trained-model.pt", self.MLP, optimizer, ii, max_loss)
                
            self.MLP.eval()    
            with torch.no_grad():
                train_loss = np.sqrt(float(lossfn(self.predict(self.XY_train[:,:-1]).reshape(-1), 
                                                  self.XY_train[:,-1])))
                test_loss =  np.sqrt(float(lossfn(self.predict(self.XY_test[:,:-1]).reshape(-1), 
                                                  self.XY_test[:,-1])))
                val_loss =   np.sqrt(float(lossfn(self.predict(self.XY_valid[:,:-1]).reshape(-1), 
                                                  self.XY_valid[:,-1])))
                Train_Losses.append(train_loss)
                Test_Losses.append(test_loss)
                Validation_Losses.append(val_loss)
                
                if ii % int(n_epochs/10) == 0 or ii == n_epochs - 1:
                    print("Epoch {0}/{1}, Training Loss = {2:0.4f}, Validation Loss = {3:0.4f}, Test Loss = {4:0.4f}".format(ii, n_epochs, train_loss, val_loss, test_loss))
            
        
        return Train_Losses, Test_Losses, Validation_Losses
    
    def Save(self, fname, model, optimizer, epoch, loss):
        print("Saving Model on epoch {}".format(epoch))
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, fname)

        