import sys
import numpy as np
import matplotlib.pyplot as plt
import json
from DataUtils import CDMS_DataLoader as DL
from model import CDMS_LR


## This script serves as an example on how to use the model for inference purposes

## First getting the model info

model_info = json.load(open("../model-info.json", "r"))
x_mean = np.array(model_info['normalization']['x_mean'])
x_std  = np.array(model_info['normalization']['x_std'])
y_norm = model_info['normalization']['y_norm']

## Now getting a random subset of data

myDL = DL('../data/CDMS_Dataset.csv', sep=',')
X_train, X_test, y_train, y_test = myDL.split_data(split_type='random')
X_train, X_test, y_train, y_test, x_mean, x_std = myDL.normalize_data(X_train, X_test, y_train, y_test,
                                                                      x_mean = x_mean,
                                                                      x_std  = x_std,
                                                                      y_norm = y_norm)

## Now creating a LR model

lr = CDMS_LR(X_train, y_train, X_test, y_test, x_mean, x_std, y_norm, mode="LR")
lr.set_bias(model_info['Linear_Regression']['coeffs'][0])
lr.set_coefficients(np.array(model_info['Linear_Regression']['coeffs'][1:]))
lr.model_eval()

## Now creating a LR-Ridge model

alpha = model_info['Ridge_Regression']['alpha']
lr = CDMS_LR(X_train, y_train, X_test, y_test, x_mean, x_std, y_norm, mode="Ridge", alpha=alpha)
lr.set_bias(model_info['Ridge_Regression']['coeffs'][0])
lr.set_coefficients(np.array(model_info['Ridge_Regression']['coeffs'][1:]))
lr.model_eval()

## Now creating a LR-Lasso model

alpha = model_info['Lasso_Regression']['alpha']
lr = CDMS_LR(X_train, y_train, X_test, y_test, x_mean, x_std, y_norm, mode="Lasso", alpha=alpha)
lr.set_bias(model_info['Lasso_Regression']['coeffs'][0])
lr.set_coefficients(np.array(model_info['Lasso_Regression']['coeffs'][1:]))
lr.model_eval()