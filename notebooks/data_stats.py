import pandas as pd
import numpy as np
import scipy.stats as sc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
import math as m
import random as r
import numpy.matlib

from scipy.stats import norm
from scipy.stats import uniform
from datetime import datetime


class FactorDescriptor:
    def __init__(self, data):
        self.data = data.fillna(method='bfill').fillna(method='ffill')
        
    def plot_data(self):
        fig, axs = plt.subplots(2,1, figsize=(20, 10))
        plot_series(self.data, ax=axs[0])
        plot_series(self.data, pct_change=True, ax=axs[1])
        plt.show()
    
    def plot_dist(self):
        fig, axs = plt.subplots(1,3, figsize=(20,4))
        plot_hist(self.data, pct_change=True, ax=axs[0])
        plot_qq_plot(self.data, ax=axs[1])
        plot_qq_plot(self.data, ax=axs[2], dist='t')
        plt.show()

    def decomposition(self):
        self.result_add = plot_decomposition(self.data)
        self.result_mult = plot_decomposition(self.data, 'multiplicative')
        
    def stationarity_info(self, lag=None):
        if lag is None:
            stationarity_tests(self.data)
        else:
            stationarity_tests(self.data.diff(periods=lag).dropna())

    def residuals_info(self):
        assert self.result_add, "Run applicative decomposition first"
        assert self.result_mult, "Run multiplicative decomposition first"
        print('\tApplicative:')
        residulals_test(self.result_add.resid)
        print('\n\tMultiplicative:')
        residulals_test(self.result_mult.resid)

    def full_info(self):
        self.plot_data()
        self.plot_dist()
        self.decomposition()
        print('Tests on residuals')
        print('*' * 10)
        self.residuals_info()
        print('\n\nStationarity tests')
        print('*' * 10)
        print('\tObserved data')
        self.stationarity_info()
        plot_acf_and_pacf(self.data)
        print('\n\tDiff-1')
        self.stationarity_info()
        print('\n\tSeasonal difference (30)')
        self.stationarity_info(lag=30)
        print('\n\tSeasonal difference (90)')
        self.stationarity_info(lag=90)
        print('\n\tSeasonal difference (365)')
        self.stationarity_info(lag=365)

def plot_hist(data, title="", pct_change=False, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(6,4)).add_subplot(111)
    if pct_change:
        data = data.pct_change()
    if title == "":
        if pct_change:
            title = "Percentage changes distribution"
        else:
            title = "Data distribution"
    data.hist(bins=50, density=True, histtype="stepfilled", alpha=0.5, ax=ax)
    ax.set_title(title)

def plot_series(data, title="", pct_change=False, ax=None):
    if ax is None:
        ax = plt.figure(figsize=(20,6)).add_subplot(111)
    if pct_change:
        data = data.pct_change()
    if title == "":
        if pct_change:
            title = "Data percentage changes"
        else:
            title = "Data"
    ax.set_title(title)
    sns.lineplot(x=data.index, y=data, ax=ax)

def plot_qq_plot(data, title="", pct_change=True, ax=None, dist='norm'):
    if ax is None:
        ax = plt.figure(figsize=(6,4)).add_subplot(111)
    if pct_change:
        data = data.pct_change()
    if title == "":
        if dist == 'norm':
            title_d = 'Normal'
        elif dist == 't':
            title_d = 'Student'
        if pct_change:
            title = f"{title_d} probability plot of percentage changes"
        else:
            title = f"{title_d} probability plot"
    Q = data.dropna()
    if dist == 't':
        sparams = sc.t.fit(Q)
    else:
        sparams = ()
    sc.probplot(Q, dist=getattr(sc, dist), sparams=sparams, plot=ax)
    ax.set_title(title)

def plot_corr(data, ft_names=None, ax=None):
    if ft_names is not None:
        data = data[ft_names]
    corr_matrix = data.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    if ax is None:
        ax = plt.figure(figsize=(9, 5)).add_subplot(111)
    sns.heatmap(corr_matrix, mask=mask, cmap="YlGnBu", annot=True, ax=ax)
    
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
def reduce_with_pca(data, fraction=0.95):
    # scaler = MinMaxScaler()
    # data_rescaled = scaler.fit_transform(data)
    pca = PCA(n_components = fraction)
    reduced = pca.fit_transform(data)
    return reduced, pca

def plot_explained_variance(pca, ax=None, ylim=(0.0, 1.1)):
    if ax is None:
        ax = plt.figure(figsize=(10,5)).add_subplot(111)
    xi = np.arange(1, pca.n_components_ + 1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)    
    ax.set_ylim(*ylim)
    ax.plot(xi, y, marker='o', linestyle='--', color='b')    
    ax.set_xlabel('Number of Components')
    ax.set_xticks(np.arange(0, pca.n_components_, step=1))
    ax.set_ylabel('Cumulative variance (%)')
    ax.set_title('The number of components needed to explain variance')
    
    ax.axhline(y=0.95, color='r', linestyle='-', label='95% cut-off threshold')    
    ax.grid(axis='x')
    plt.legend()

from statsmodels.tsa.seasonal import seasonal_decompose
def plot_decomposition(data, model='additive'):
    result = seasonal_decompose(data, model=model, extrapolate_trend='freq')
    # Plot
    plt.rcParams.update({'figure.figsize': (15,10)})    
    if model == 'additive':
        title = 'Additive decomposition'
    else:
        title = 'Multiplicative decomposition'
    result.plot(observed=False).suptitle(title, fontsize=18);
    plt.show()
    return result

import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")
def stationarity_adf_test(ser, lev=0.05):
    adf_test = sm.tsa.adfuller(ser)
    if (adf_test[1] < lev): 
        res = 'the series is stationary.'
    else:
        res = 'the series is not stationary.'
    print(f'ADF test: {res}')

def stationarity_kpss_test(ser, lev=0.05):
    kpss_test = sm.tsa.kpss(ser)
    if (kpss_test[1] < lev) :
        res = 'the series is not stationary.'
    else:
        res = 'the series is stationary.'
    print(f'KPSS test: {res}')

def stationarity_tests(data):
    stationarity_adf_test(data)
    stationarity_kpss_test(data)


## Residuals
def residulals_test(resid):
    bias_student_test(resid)
    stationarity_adf_test(resid)

from scipy import stats
def bias_student_test(ser, lev=0.05):
    t_test = stats.ttest_1samp(ser, 0)
    if (t_test[1] < lev): 
        res = 'the estimator is unbiased.'
    else:
        res = 'the estimator is biased.'
    print(f'Student test: {res}')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
def plot_acf_and_pacf(ser, lag_= None, fig_size = (20, 8)):
    f = plt.figure(figsize=fig_size)
    ax1 = f.add_subplot(2, 1, 1)
    ax1.set_title('Autocorrelation')
    plot_acf(ser, lags = lag_, ax = ax1)
    ax2 = f.add_subplot(2, 1, 2)
    ax2.set_title('Partial Autocorrelation')
    plot_pacf(ser, lags = lag_, ax = ax2)
    plt.show()