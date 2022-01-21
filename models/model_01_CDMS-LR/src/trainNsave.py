import sys
import numpy as np
import matplotlib.pyplot as plt
from DataUtils import CDMS_DataLoader as DL
from model import CDMS_LR


## Obtaining the dataset 

myDL = DL('../data/CDMS_Dataset.csv', sep=',')
X_train, X_test, y_train, y_test = myDL.split_data(split_type='label', 
                                                   labels = [-12.502, -29.5, -41.9])
X_train, X_test, y_train, y_test, x_mean, x_std = myDL.normalize_data(X_train, X_test, y_train, y_test)
y_norm = myDL.y_norm

## Performing Linear Regression

lr = CDMS_LR(X_train, y_train, X_test, y_test, x_mean, x_std, y_norm, mode="LR")
lr.do_LR()
rmse_lr_train, rmse_lr_test, coeffs_opt_LR = lr.model_eval()

## Performing scan over alpha for best Ridge and Lasso Regression

alphas = np.logspace(-6, +6, 50)
RMSE_train_Ridge = []
RMSE_test_Ridge = []
RMSE_train_Lasso = []
RMSE_test_Lasso = []
min_ridge_loss = 99.0
min_lasso_loss = 99.0
alpha_opt_ridge = 0.
alpha_opt_lasso = 0.
coeffs_opt_ridge = []
coeffs_opt_lasso = []

for ii, alpha in enumerate(alphas):
    lr = CDMS_LR(X_train, y_train, X_test, y_test, x_mean, x_std, y_norm, mode="Ridge", alpha=alpha)
    lr.do_LR()
    etr, etst, coeffs = lr.model_eval(do_plot=False)
    RMSE_train_Ridge.append(etr)
    RMSE_test_Ridge.append(etst)
    if etst < min_ridge_loss:
        min_ridge_loss = etst
        alpha_opt_ridge = alpha
        coeffs_opt_ridge = coeffs
    lr = CDMS_LR(X_train, y_train, X_test, y_test, x_mean, x_std, y_norm, mode="Lasso", alpha=alpha)
    lr.do_LR()
    etr, etst, coeffs = lr.model_eval(do_plot=False)
    RMSE_train_Lasso.append(etr)
    RMSE_test_Lasso.append(etst)
    if etst < min_lasso_loss:
        min_lasso_loss = etst
        alpha_opt_lasso = alpha
        coeffs_opt_lasso = coeffs

fig = plt.figure(figsize=(8,8))
plt.plot(alphas, RMSE_train_Ridge, marker='x', color='b', label='Training Loss (Ridge)')
plt.plot(alphas, RMSE_test_Ridge, marker='o',  color='r', label='Test Loss (Ridge)')
plt.plot(alphas, RMSE_train_Lasso, marker='+', color='b', label='Training Loss (Lasso)')
plt.plot(alphas, RMSE_test_Lasso, marker='.', color='r', label='Test Loss (Lasso)')

plt.plot(alphas, [rmse_lr_train]*len(alphas), color='k', linestyle='--', label='Training Loss (Linear Regression)')
plt.plot(alphas, [rmse_lr_test]*len(alphas), color='g', linestyle = '--', label='Test Loss (Linear Regression)')

plt.xscale('log')
plt.xlabel(r'$\alpha$')
plt.ylabel("Loss Value")
plt.legend(fontsize = 20)
plt.ylim([1.0, 3.0])
plt.show()

print("Minimum Test Loss for Ridge: ", min_ridge_loss)
print("Alpha for Minimum Test Loss for Ridge: ", alpha_opt_ridge)

print("Minimum Test Loss for Lasso: ", min_lasso_loss)
print("Alpha for Minimum Test Loss for Lasso: ", alpha_opt_lasso)

## Now save the models

import json

model_info ={}

model_info['normalization'] = {'x_mean':list(x_mean), 
                               'x_std':list(x_std), 
                               'y_norm':y_norm}
model_info['test_data'] = {'labels':[-12.502, -29.5, -41.9]}
model_info['Linear_Regression']={'coeffs':list(coeffs_opt_LR)}
model_info['Ridge_Regression'] ={'alpha':alpha_opt_ridge,
                                 'coeffs':list(coeffs_opt_ridge)}
model_info['Lasso_Regression'] ={'alpha':alpha_opt_lasso,
                                 'coeffs':list(coeffs_opt_lasso)}

json.dump(model_info, open("../model-info.json","w"), indent=3)

## Save model metadata

model_metadata = {}

model_metadata['genInfo'] = '''
This set of models represents the trained parameters for a Linear Regression model for determining the impact location on a superCDMS prototype based on timing measurements. 
'''
model_metadata['data_ID']  = 'https://github.com/FAIR-UMN/FAIR-UMN-CDMS/blob/main/data/processed_csv/processed_combined.csv?raw=true'
model_metadata['model_ID'] = ''
model_metadata['keywords'] = [key for key in model_info.keys()]
model_metadata['model_keys'] = {
    'normalization':{
        'description': 'Vectors for data standardization',
        'entries':{'x_mean': 'Mean value of input features',
                   'x_std':  'St. deviation of input features',
                   'y_norm': 'Normalization constant for target variable'
                  }
    },
    'test_data':{
        'description': 'How the test_data has been isolated',
        'entries':{'labels': 'Data label for target variable to separate test dataset'
                  }
    },
    'Linear_Regression':{
        'description': 'Model parameters for a simple linear least squared model',
        'entries':{'coeffs': 'List of linear coefficients for input features, the first entry represents the intercept'
                  }
    },
    'Ridge_Regression':{
        'description': 'Model parameters for a linear least squared model with Ridge regularization',
        'entries':{'coeffs': 'List of linear coefficients for input features, the first entry represents the intercept',
                   'alpha': 'Value of regularization strength determining model hyperparameter'
                  }
    },
    'Lasso_Regression':{
        'description': 'Model parameters for a linear least squared model with Lasso regularization',
        'entries':{'coeffs': 'List of linear coefficients for input features, the first entry represents the intercept',
                   'alpha': 'Value of regularization strength determining model hyperparameter'
                  }
    },
}

json.dump(model_metadata, open("../model-metadata.json","w"), indent=3)

