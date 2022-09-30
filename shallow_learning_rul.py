'''
Shallow Learning for RUL:
- Maximum Likelihood
- Linear Regression
- Ridge Regression
- LASSO
- Support Vector Regression
- Random Forest Regression
- Hidden Markov Model
'''

import imp
from subprocess import IDLE_PRIORITY_CLASS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from scipy import stats
from sklearn.decomposition import PCA
from fcmeans import FCM

from rul_classes import RULFrame

class MaximumLikelihood:

    def __init__(self) -> None:
        pass
    
    class Weibull:
        '''
        Maximum likelihood with two parameter Weibull ditribution. Model only
        sees total lifetimes, no conditional data and nothing.
        
        Inputs: rulframe    |       RULFrame
        '''

        def __init__(self, rulframe) -> None:
            self.dstrbtn = 'weibull'
            self.nprmtr = 2
            self.prmtr = (0.0, 0.0, 0.0) # shape, location, scale
            self.stmte = np.array()

            # for fit
            self.Xtrain = rulframe.train[[rulframe.id_col, rulframe.time_col]].groupby(rulframe.id_col).max() - rulframe.start

        def fit(self):
            self.prmtr = stats.weibull_min.fit(self.Xtrain, floc = 0)

        def plot(self):
            x = np.linspace(0, x.max(), 1000)
            y = stats.weibull_min.pdf(x, c = self.prmtr[0], loc = self.prmtr[1], scale = self.prmtr[2])
    
            plt.figure(figsize=(15,5))

            self.Xtrain.hist(density = True)
            plt.plot(x, y)

            plt.ylabel('y')
            plt.xlabel('x')
            plt.show()

    class Gamma:
        '''
        Maximum likelihood with two parameter Gamma ditribution. Model only
        sees total lifetimes, no conditional data and nothing.
        
        Inputs: rulframe    |       RULFrame
        '''

        def __init__(self, rulframe) -> None:
            self.dstrbtn = 'weibull'
            self.nprmtr = 2
            self.prmtr = (1.0, 1.0) # shape, scale
            self.stmte = np.array()

            # for fit
            self.Xtrain = rulframe.train[[rulframe.id_col, rulframe.time_col]].groupby(rulframe.id_col).max() - rulframe.start

        def fit(self):
            pass

        def plot(self):
            x = np.linspace(0, x.max(), 1000)
            y = stats.placeholder.pdf(x, c = self.prmtr[0], loc = self.prmtr[1], scale = self.prmtr[2])
    
            plt.figure(figsize=(15,5))

            self.Xtrain.hist(density = True)
            plt.plot(x, y)

            plt.ylabel('y')
            plt.xlabel('x')
            plt.show()

    class Exponential:
        '''
        Maximum likelihood with two parameter Gamma ditribution. Model only
        sees total lifetimes, no conditional data and nothing.
        
        Inputs: rulframe    |       RULFrame
        '''

        def __init__(self, rulframe) -> None:
            self.dstrbtn = 'weibull'
            self.nprmtr = 2
            self.prmtr = (0.0, 0.0, 0.0) # shape, location, scale
            self.stmte = np.array()

            # for fit
            self.Xtrain = rulframe.train[[rulframe.id_col, rulframe.time_col]].groupby(rulframe.id_col).max() - rulframe.start

        def fit(self):
            self.prmtr = stats.weibull_min.fit(self.Xtrain, floc = 0)

        def plot(self):
            x = np.linspace(0, x.max(), 1000)
            y = stats.weibull_min.pdf(x, c = self.prmtr[0], loc = self.prmtr[1], scale = self.prmtr[2])
    
            plt.figure(figsize=(15,5))

            self.Xtrain.hist(density = True)
            plt.plot(x, y)

            plt.ylabel('y')
            plt.xlabel('x')
            plt.show()


class Classical_Estimation:

    def __init__(self) -> None:
        pass

    def fit_weibull(self, rulframe, label, plot_fit = False) -> None:
        '''
        'Stupid' maximum likelihood with two parameter Weibull ditribution. Why stupid? Model only
        sees total lifetimes, no conditional data and nothing.
        
        Inputs: data        |       RULFrame
                rul_col     |       string, rul_column for fit

        Outputs: tuple (shape, location, scale), where location is set by design 0
        '''
        x = rulframe.train[[rulframe.id_col, label]].groupby(rulframe.id_col).max()
        shape, location, scale = stats.weibull_min.fit(x, floc = 0)

        if plot_fit:
            x2 = np.linspace(0, x.max(), 1000)
            y = stats.weibull_min.pdf(x2, c = shape, loc = location, scale = scale)
    
            plt.figure(figsize=(15,5))
            x.hist(density = True)
            plt.plot(x, y)
            plt.ylabel('y')
            plt.xlabel('x')
            plt.show()
    


    def linear(self, rulframe, label, independent = ['sensors', 'lagged_sensors', 'poly', 'diff']) -> None:
        '''
        Linear Regression with design matrix containing sensors, lagged sensors, polynomial and differences if there are any.
        '''
        pass

    def support_vector_regression(self, rulframe, label, independent = ['sensors', 'lagged_sensors', 'poly', 'diff']) -> None:
        pass

    def random_forest_regression(self, rulframe, label, independent = ['sensors', 'lagged_sensors', 'poly', 'diff']) -> None:
        pass