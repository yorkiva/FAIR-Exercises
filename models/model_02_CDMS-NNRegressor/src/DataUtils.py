import numpy as np
import pandas as pd
from sklearn.model_selection import *


class CDMS_DataLoader:
    def __init__(self, loc, sep=','):
        self.data = pd.read_csv(loc, sep)
        self.features = list(self.data.columns)[1:]
        self.xy_data = self.data.values[:,1:]
        self.n_features = len(self.features)-1
        
    def split_data(self, split_type, labels=[]):
        if split_type == 'random':
            XY_train, XY_test = train_test_split(self.xy_data, test_size=0.20, random_state=42)
        if split_type == 'label':
            XY_train = self.data.loc[~self.data[self.features[-1]].isin(labels)].values[:, 1:]
            XY_test = self.data.loc[self.data[self.features[-1]].isin(labels)].values[:, 1:]
            
        XY_train, XY_valid = train_test_split(XY_train, test_size=0.2, stratify=XY_train[:, -1])
        return XY_train, XY_valid, XY_test
    
    def normalize_data(self, XY_train, XY_valid, XY_test, x_mean = [], x_std = []):
        if x_mean == []:
            x_mean = np.mean(XY_train[:,:-1], axis = 0)
        if x_std == []:
            x_std  = np.std(XY_train[:,:-1], axis = 0)
            
        XY_train[:,:-1] = (XY_train[:,:-1] - x_mean)/x_std
        XY_test[:,:-1]  = (XY_test[:,:-1] - x_mean)/x_std
        XY_valid[:,:-1]  = (XY_valid[:,:-1] - x_mean)/x_std
    
        return XY_train, XY_valid, XY_test, x_mean, x_std
        