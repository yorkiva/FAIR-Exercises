import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import *
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import seaborn as sns
import pandas as pd
import json

class DataProcessor:
    def __init__(self, data, data_type = 'extended',
                 preprocess = True, skip_amp = True):
        self.data_type = data_type
        self.preprocess = preprocess
        self.skip_amp = skip_amp
        self.get_xy(data)
        
    def get_xy(self, data): 
        if self.data_type == 'extended' and self.skip_amp:
            first_feature = 6
        else:
            first_feature = 1
        self.ff = first_feature
        self.features = list(data.columns)[first_feature:-1]
        self.x = data.values[:,first_feature:-1]
        self.y = data.values[:, -1]
        self.nfeat = len(self.features)
        self.ndata = len(self.x)
        if self.data_type == 'extended':
            # dropping the trivial entry 'PAr20'
            keep = ~(np.array(self.features) == 'PAr20')
            self.x = self.x[:,keep]
            self.nfeat -= 1
            self.features.remove('PAr20')
        
    def get_unique_labels(self):
        return np.unique(self.y)

        
    def get_data(self, split_type = 'random', labels = [], test_size = 0.2):
        if split_type == 'random' or labels == []:
            X_train, X_test, y_train, y_test = train_test_split(self.x, 
                                                                self.y, 
                                                                test_size = test_size)
        elif split_type == 'labels':
            Idx = np.isin(self.y, labels, invert=True)
            X_train = self.x[Idx, :]
            X_test  = self.x[~Idx, :]
            y_train = self.y[Idx]
            y_test  = self.y[~Idx]
        else:
            raise Exception("Unknown split type")
            
        X_train, X_val, y_train, y_val = train_test_split(X_train, 
                                                          y_train, 
                                                          test_size = test_size)
    
        if self.preprocess:
            x_mean = np.mean(X_train, axis = 0)
            x_std  = np.std(X_train, axis = 0)
            y_norm = -41.9
        else:
            x_mean = 0
            x_std = 1
            y_norm = 1
        X_train = (X_train - x_mean)/x_std
        X_test  = (X_test - x_mean)/x_std
        X_val  = (X_val - x_mean)/x_std
        y_train = y_train/y_norm
        y_test  = y_test/y_norm
        y_val = y_val/y_norm

        return X_train, X_val, X_test, y_train, y_val, y_test, x_mean, x_std, y_norm

class CDMS_LR:
    def __init__(self, 
                 X_train, X_val, X_test, 
                 y_train, y_val, y_test, 
                 x_mean, x_std, y_norm,
                 mode="LR", alpha=1.0):
        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_norm = y_norm
        self.mode = mode
        self.alpha = alpha
        if mode == "LR":
            self.LR = LinearRegression()
        if mode == "Ridge":
            self.LR = Ridge(alpha = alpha, max_iter = 2000)
        if mode == "Lasso":
            self.LR = Lasso(alpha = alpha, max_iter = 2000)
    
    
    def get_bias(self):
        return self.LR.intercept_
    
    def get_coefficients(self):
        return self.LR.coef_
    
    def do_LR(self, do_plot=True, plotname = "fig.pdf"):
        y_norm = self.y_norm
        self.LR = self.LR.fit(self.X_train, self.y_train)
        bias, coeffs = self.get_bias(), self.get_coefficients()
        coeffs = np.append(bias, coeffs)
        names = ['$\\beta_{' +str(i) +'}$' for i in range(len(coeffs))]
        
        y_train_pred = self.LR.predict(self.X_train)
        MSE_train = np.sqrt(np.mean((self.y_train*y_norm - y_train_pred*y_norm)**2))
        
        y_val_pred = self.LR.predict(self.X_val)
        MSE_val = np.sqrt(np.mean((self.y_val*y_norm - y_val_pred*y_norm)**2))
        
        y_test_pred = self.LR.predict(self.X_test)
        MSE_test = np.sqrt(np.mean((self.y_test*y_norm - y_test_pred*y_norm)**2))
        
        if do_plot:
            print("Regression mode:", self.mode)
            if self.mode != 'LR':
                print("alpha = ", self.alpha)
            print("RMSE from train data = {}".format(MSE_train))
            print("RMSE from val data = {}".format(MSE_val))
            print("RMSE from test data = {}".format(MSE_test))
        
        
            fig = plt.figure(figsize=(8,8))
            plt.scatter(self.y_train*y_norm, y_train_pred*y_norm, 
                        marker='x', s=100, color='b', label='Train')
            plt.scatter(self.y_val*y_norm, y_val_pred*y_norm, 
                        marker='.', s=100, color='g', label='Val')
            plt.scatter(self.y_test*y_norm,  y_test_pred*y_norm, 
                        marker='.', color='r', label='Test')
            plt.plot(self.y_train*y_norm, self.y_train*y_norm, color='k')
            plt.xlabel("True Value", fontsize = 20)
            plt.xticks(fontsize = 20)
            plt.ylabel("Predicted Value", fontsize = 20)
            plt.yticks(fontsize = 20)
            # plt.title(self.mode + (' a = {}'.format(self.alpha) if self.mode != 'LR' else ''))
            plt.legend(fontsize = 20)
            plt.savefig(plotname)
            data_dict = {
                'y_train' : list(self.y_train*y_norm),
                'y_val': list(self.y_val*y_norm),
                'y_test': list(self.y_test*y_norm),
                'y_train_pred': list(y_train_pred*y_norm),
                'y_val_pred': list(y_val_pred*y_norm),
                'y_test_pred': list(y_test_pred*y_norm)
            }
            
            _f_ = open(plotname + '.DATA.json', 'w')
            json.dump(data_dict, _f_)
            plt.show()
        
        return MSE_train, MSE_val, MSE_test, coeffs


class CDMS_PCA(CDMS_LR):
    def __init__(self, 
                 X_train, X_val, X_test, 
                 y_train, y_val, y_test, 
                 x_mean, x_std, y_norm,
                 mode="LR", alpha=1.0,
                 n_components = 0.9999):
        
        self.pca = PCA(n_components = n_components)
        self.pca.fit(X_train)
        CDMS_LR.__init__(self,
                         self.pca.transform(X_train), self.pca.transform(X_val), self.pca.transform(X_test),
                         y_train, y_val, y_test,
                         x_mean, x_std, y_norm,
                         mode = mode, alpha = alpha)
    
    def get_components(self):
        return self.pca.components_
    
    def get_ncomponents(self):
        return self.pca.n_components_
    
    def get_variances(self):
        return self.pca.explained_variance_
    
    def get_variance_ratios(self):
        return self.pca.explained_variance_ratio_
    
    def transform(self, X):
        return self.pca.transform(X)

    
# class CDMS_RF:
#     def __init__(self, 
#                  X_train, X_val, X_test, 
#                  y_train, y_val, y_test, 
#                  x_mean, x_std, y_norm,
#                  mode="LR", alpha=1.0):
#         self.X_train = X_train
#         self.X_test = X_test
#         self.X_val = X_val
#         self.y_train = y_train
#         self.y_test = y_test
#         self.y_val = y_val
#         self.x_mean = x_mean
#         self.x_std = x_std
#         self.y_norm = y_norm
#         self.mode = mode
#         self.rf = 
    
    
#     def get_bias(self):
#         return self.LR.intercept_
    
#     def get_coefficients(self):
#         return self.LR.coef_
    
#     def do_LR(self, do_plot=True):
#         y_norm = self.y_norm
#         self.LR = self.LR.fit(self.X_train, self.y_train)
#         bias, coeffs = self.get_bias(), self.get_coefficients()
#         coeffs = np.append(bias, coeffs)
#         names = ['$\\beta_{' +str(i) +'}$' for i in range(len(coeffs))]
        
#         y_train_pred = self.LR.predict(self.X_train)
#         MSE_train = np.sqrt(np.mean((self.y_train*y_norm - y_train_pred*y_norm)**2))
        
#         y_val_pred = self.LR.predict(self.X_val)
#         MSE_val = np.sqrt(np.mean((self.y_val*y_norm - y_val_pred*y_norm)**2))
        
#         y_test_pred = self.LR.predict(self.X_test)
#         MSE_test = np.sqrt(np.mean((self.y_test*y_norm - y_test_pred*y_norm)**2))
        
#         if do_plot:
#             print("RMSE from train data = {}".format(MSE_train))
#             print("RMSE from val data = {}".format(MSE_val))
#             print("RMSE from test data = {}".format(MSE_test))
        
        
#             fig = plt.figure(figsize=(8,8))
#             plt.scatter(self.y_train*y_norm, y_train_pred*y_norm, 
#                         marker='x', s=100, color='b', label='Train')
#             plt.scatter(self.y_val*y_norm, y_val_pred*y_norm, 
#                         marker='.', s=100, color='g', label='Val')
#             plt.scatter(self.y_test*y_norm,  y_test_pred*y_norm, 
#                         marker='.', color='r', label='Test')
#             plt.plot(self.y_train*y_norm, self.y_train*y_norm, color='k')
#             plt.xlabel("True Value", fontsize = 20)
#             plt.ylabel("Predicted Value", fontsize = 20)
#             plt.title(self.mode + (' a = {}'.format(self.alpha) if self.mode != 'LR' else ''))
#             plt.legend(fontsize = 20)
#             plt.show()
        
#         return MSE_train, MSE_val, MSE_test, coeffs
    
def LR_analysis(dataset, mode = "Full", do_plot = True, plotnametag = "LR", n_components = 0.99):
    X_train, X_val, X_test,\
    y_train, y_val, y_test,\
    x_mean, x_std, y_norm = dataset.get_data(split_type='labels',
                            labels = [-12.502,-29.5,-41.9])
    if mode != "PCA":
        lr = CDMS_LR(X_train, X_val, X_test, 
                     y_train, y_val, y_test, 
                     x_mean, x_std, y_norm,
                     mode="LR")
        lr_ridge = CDMS_LR(X_train, X_val, X_test, 
                   y_train, y_val, y_test, 
                   x_mean, x_std, y_norm,
                   mode="Ridge", alpha = 1.0)
        lr_lasso = CDMS_LR(X_train, X_val, X_test, 
                   y_train, y_val, y_test, 
                   x_mean, x_std, y_norm,
                   mode="Lasso", alpha = 0.01)
    else:
        lr = CDMS_PCA(X_train, X_val, X_test, 
                     y_train, y_val, y_test, 
                     x_mean, x_std, y_norm,
                     mode="LR", n_components = n_components)
        lr_ridge = CDMS_PCA(X_train, X_val, X_test, 
                   y_train, y_val, y_test, 
                   x_mean, x_std, y_norm,
                   mode="Ridge", alpha = 1.0, n_components = n_components)
        lr_lasso = CDMS_PCA(X_train, X_val, X_test, 
                   y_train, y_val, y_test, 
                   x_mean, x_std, y_norm,
                   mode="Lasso", alpha = 0.01, n_components = n_components)
        
    trL_lr, vL_lr, tL_lr, _ = lr.do_LR(do_plot = do_plot,
                                       plotname = plotnametag + "_LR_" + mode + ".png")
    if mode == "PCA":
        print("Number of principal components: ", lr.get_ncomponents())
    if do_plot:
        _ = lr_ridge.do_LR(True, plotname = plotnametag + "_Ridge_" + mode + "_alpha_1.0.png")
        _ = lr_lasso.do_LR(True, plotname = plotnametag + "_Lasso_" + mode + "_alpha_0.1.png")
    
    
    alphas = np.logspace(-6, +6, 50)
    RMSE_train_Ridge = []
    RMSE_test_Ridge = []
    RMSE_val_Ridge = []
    RMSE_train_Lasso = []
    RMSE_test_Lasso = []
    RMSE_val_Lasso = []

    for ii, alpha in enumerate(alphas):
        if mode == "PCA":
            lr = CDMS_PCA(X_train, X_val, X_test, 
                         y_train, y_val, y_test, 
                         x_mean, x_std, y_norm,
                         mode="Ridge", alpha = alpha, 
                         n_components = n_components)
        else:
            lr = CDMS_LR(X_train, X_val, X_test, 
                         y_train, y_val, y_test, 
                         x_mean, x_std, y_norm,
                         mode="Ridge", alpha = alpha)
            
        etr, ev, etst, _ = lr.do_LR(do_plot=False)
        RMSE_train_Ridge.append(etr)
        RMSE_test_Ridge.append(etst)
        RMSE_val_Ridge.append(ev)
        if mode == "PCA":
            lr = CDMS_PCA(X_train, X_val, X_test, 
                         y_train, y_val, y_test, 
                         x_mean, x_std, y_norm,
                         mode="Lasso", alpha = alpha, 
                         n_components = n_components)
        else:
            lr = CDMS_LR(X_train, X_val, X_test, 
                         y_train, y_val, y_test, 
                         x_mean, x_std, y_norm,
                         mode="Lasso", alpha = alpha)
        etr, ev, etst, _ = lr.do_LR(do_plot=False)
        RMSE_train_Lasso.append(etr)
        RMSE_test_Lasso.append(etst)
        RMSE_val_Lasso.append(ev)    
        
    if do_plot:
        fig = plt.figure(figsize=(8,8))
        plt.plot(alphas, RMSE_train_Ridge, marker='x', color='b', label='Training (Ridge)')
        plt.plot(alphas, RMSE_test_Ridge, marker='o',  color='r', label='Test (Ridge)')
        plt.plot(alphas, RMSE_val_Ridge, marker='o',  color='g', label='Validation (Ridge)')
        plt.plot(alphas, RMSE_train_Lasso, marker='+', color='b', label='Training (Lasso)')
        plt.plot(alphas, RMSE_test_Lasso, marker='.', color='r', label='Test (Lasso)')
        plt.plot(alphas, RMSE_val_Lasso, marker='.', color='g', label='Validation Loss (Lasso)')

        plt.plot(alphas, [trL_lr]*len(alphas), color='k', linestyle='--', label='Training (OLS)')
        plt.plot(alphas, [tL_lr]*len(alphas), color='c', linestyle = '--', label='Test (OLS)')
        plt.plot(alphas, [vL_lr]*len(alphas), color='y', linestyle = '--', label='Validation (OLS)')
        
        data_dict = {
            'RMSE_train_Ridge': list(RMSE_train_Ridge),
            'RMSE_test_Ridge' : list(RMSE_test_Ridge),
            'RMSE_val_Ridge'  : list(RMSE_val_Ridge),
            'RMSE_train_Lasso': list(RMSE_train_Lasso),
            'RMSE_test_Lasso' : list(RMSE_test_Lasso),
            'RMSE_val_Lasso'  : list(RMSE_val_Lasso),
            'alphas'          : list(alphas),
            'RMSE_train_OLS'  : [trL_lr]*len(alphas),
            'RMSE_test_OLS'   : [tL_lr]*len(alphas),
            'RMSE_val_OLS'    : [vL_lr]*len(alphas)
        }
        
        _f_ = open(plotnametag + "_alphascan_" + mode + '.DATA.json', 'w')
        json.dump(data_dict, _f_)
        

        plt.xscale('log')
        plt.xlabel(r'$\alpha$', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel("Loss Value", fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize = 13, ncol=1)
        plt.ylim([1.0, 3.0])
        plt.savefig(plotnametag + "_alphascan_" + mode + ".png")
        plt.show()
    
    idx = np.argmin(RMSE_val_Ridge)
    print("Minimum Val Loss for Ridge: ", RMSE_val_Ridge[idx])
    print("Alpha for Minimum Val Loss for Ridge: ", alphas[idx])
    print("Corresponding Test Loss: ", RMSE_test_Ridge[idx])
    a_ridge, vL_ridge, tL_ridge = alphas[idx], RMSE_val_Ridge[idx], RMSE_test_Ridge[idx]
    
    idx = np.argmin(RMSE_val_Lasso)
    print("Minimum Val Loss for Lasso: ", RMSE_val_Lasso[idx])
    print("Alpha for Minimum Val Loss for Lasso: ", alphas[idx])
    print("Corresponding Test Loss: ", RMSE_test_Lasso[idx])
    a_lasso, vL_lasso, tL_lasso = alphas[idx], RMSE_val_Lasso[idx], RMSE_test_Lasso[idx]
    
    return a_ridge, vL_ridge, tL_ridge, a_lasso, vL_lasso, tL_lasso, vL_lr, tL_lr
    
    