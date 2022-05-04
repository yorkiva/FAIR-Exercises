import numpy as np
import pandas as pd
from sklearn.model_selection import *


class CDMS_DataLoader:
    def __init__(self, loc, sep=','):
        self.data = pd.read_csv(loc, sep)
        self.features = list(self.data.columns)[1:]
        self.n_features = len(self.features)-1
        self.xy_data = self.uniformize(self.data.values[:,1:])

    def uniformize(self, xy_data):
        unique_ys = np.unique(xy_data[:,-1])
        counts = []
        for y in unique_ys:
            counts.append(int(np.sum(xy_data[:,-1] == y)))
        min_count = min(counts)
        xy_data_uniformized = np.zeros((min_count*len(unique_ys), self.n_features+1))
        for ii,y in enumerate(unique_ys):
            indices = xy_data[:,-1] == y
            selected = xy_data[indices][:min_count, :]
            xy_data_uniformized[ii*min_count:(ii+1)*min_count, :] = selected
        return xy_data_uniformized
        
    def split_data(self, split_type, labels=[]):
        y_norm = -41.9 # largest y dimension to be used to normalize y data
        x_mean = np.mean(self.xy_data[:,:-1], axis = 0)
        x_std  = np.std(self.xy_data[:,:-1], axis = 0)
        if split_type == 'random':
            XY_train, XY_test  = train_test_split(self.xy_data, test_size=0.25, stratify=self.xy_data[:, -1])
        if split_type == 'label':
            XY_train = self.data.loc[~self.data[features[-1]].isin(labels)].values[:, 1:]
            XY_test  = self.data.loc[self.data[features[-1]].isin(labels)].values[:, 1:]
        XY_train[:,:-1] = (XY_train[:,:-1] - x_mean)/x_std
        XY_test[:,:-1]  = (XY_test[:,:-1] - x_mean)/x_std
        XY_train[:,-1]  = 1*(2*XY_train[:,-1]/y_norm - 1)
        XY_test[:,-1]   = 1*(2*XY_test[:,-1]/y_norm -1)
    
        return XY_train, XY_test
