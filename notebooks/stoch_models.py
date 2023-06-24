import pandas as pd
import numpy as np

import datetime
from datetime import datetime

from tqdm.auto import tqdm, trange
from copy import copy, deepcopy

import sys
import warnings
warnings.simplefilter("ignore")
from tqdm.auto import tqdm


import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
sns.set_style('whitegrid')

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.api import OLS

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def Wiener_process(w0, dt, N_traj):
    d = 1
    N_t = len(dt) + 1
    W_s = np.zeros([N_traj, d, N_t])
    W_s[..., 0] = w0
    for kk in np.arange(1, N_t):
        noises = np.random.randn(N_traj, d)
        W_s[...,kk] = W_s[..., kk-1] + np.sqrt(dt[kk-1]) * noises
    return W_s[:,0,:]

def generate_dW_corr(data, dt, w0=0, N_traj=1):
    cholesky_matrix = np.linalg.cholesky(np.corrcoef(data, rowvar=False))
    dW = np.array([np.diff(Wiener_process(w0=w0, dt=dt,N_traj=N_traj)) for _ in range(data.shape[1])])

    dW_corr = dW.copy()

    for i in range(0, N_traj):
        dW_corr[:, i, :] = np.dot(cholesky_matrix, dW[:, i, :])
    
    return dW_corr  # (n_ft x N_traj x ts_len)

def parametrs_estimation_1(r, dt):
    # замена
    y = np.diff(r)
    X = np.concatenate([dt.reshape(-1, 1)], axis=1)
    # lin reg
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(X, y)
    y_pred = linreg.predict(X)
    # параметры
    a = linreg.coef_[0]
    b = np.std(y - y_pred)
    return {'a': a, 'b': b}

def process_1(param, r_0, dt, dWt):
    T = len(dt) + 1
    a, b = param['a'], param['b']

    r_sim = [r_0]
    for t in range(1, T):
        r_t1 = r_sim[t-1]
        r_t = r_t1 + a*dt[t-1] + b*dWt[t-1]
        r_sim.append(r_t)

    return np.asarray(r_sim)

def parametrs_estimation_2(r, dt):
    # замена
    r_t1 = r[:-1]
    y = np.diff(r) / r_t1
    X = np.concatenate([dt.reshape(-1, 1)], axis=1)
    # lin reg
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(X, y)
    y_pred = linreg.predict(X)
    # параметры
    a = linreg.coef_[0]
    b = np.std(y - y_pred)
    return {'a': a, 'b': b}

def process_2(param, r_0, dt, dWt):
    T = len(dt) + 1
    a, b = param['a'], param['b']

    r_sim = [r_0]
    for t in range(1, T):
        r_t1 = r_sim[t-1]
        r_t = r_t1 + a*r_t1*dt[t-1] + b*r_t1*dWt[t-1]
        r_sim.append(r_t)

    return np.asarray(r_sim)

def parametrs_estimation_CIR(r, dt):
    # замена
    sqrt_r_t1 = np.sqrt(r[:-1])
    y = np.diff(r) / sqrt_r_t1
    x_1 = dt / sqrt_r_t1
    x_2 = dt * sqrt_r_t1
    X = np.concatenate([x_1.reshape(-1, 1), x_2.reshape(-1, 1)], axis=1)
    # lin reg
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(X, y)
    y_pred = linreg.predict(X)
    # параметры: #p1 = ab, p2 = -a
    ab = linreg.coef_[0]
    a = -linreg.coef_[1]
    b = ab / a
    c = np.std(y - y_pred)
    if 2*a*b < c**2:
        #print('Ошибка в параметрах')
        print('Внимание: 2ab < c^2')
    return {'a': a, 'b': b, 'c': c}

def process_CIR(param, r_0, dt, dWt):
    T = len(dt) + 1
    a, b, c = param['a'], param['b'], param['c']

    r_sim = [r_0]
    for t in range(1, T):
        r_t1 = r_sim[t-1]
        r_t = r_t1 + a*(b - r_t1)*dt[t-1] + c*np.sqrt(np.abs(r_t1))*dWt[t-1]
        r_sim.append(r_t)

    return np.asarray(r_sim)

def parametrs_estimation_FX(r, dt):
    # замена
    y = np.diff(r)
    x_1 = dt
    x_2 = dt * r[:-1]
    X = np.concatenate([x_1.reshape(-1, 1), x_2.reshape(-1, 1)], axis=1)
    # lin reg
    linreg = LinearRegression(fit_intercept=False)
    linreg.fit(X, y)
    y_pred = linreg.predict(X)
    # параметры
    a = -linreg.coef_[1]
    b = linreg.coef_[0] / a
    c = np.std(y - y_pred)
    return {'k': a, 'theta': b, 'sigma': c}

def process_FX(param, r_0, dt, dWt):
    T = len(dt) + 1
    k, th, sigma = param['k'], param['theta'], param['sigma']

    y_sim = [r_0]
    for t in range(1, T):
        y_t1 = y_sim[t-1]
        y_t = y_t1 + k*(th - y_t1)*dt[t-1] + sigma*dWt[t-1]
        y_sim.append(y_t)

    return np.exp(y_sim)

def calculate_metric(y_true, y_sim_N, mode='rmse'):
    res = []
    if mode == 'rmse':
        for y_sim in y_sim_N:
            res.append(mean_squared_error(y_true, y_sim, squared=False))
        return np.mean(res)
    elif mode == 'mae':
        for y_sim in y_sim_N:
            res.append(mean_absolute_error(y_true, y_sim))
        return np.mean(res)
    elif mode == 'mape':
        for y_sim in y_sim_N:
            res.append(mean_absolute_percentage_error(y_true, y_sim))
        return np.mean(res)
    else:
        return None

class Stoch_Models:
    def __init__(self, factor_name,
                 value_train, value_test,
                 t_train, t_test,
                 N_traj=10, dW_N=None, models=None):
        self.name = factor_name
        self.val_train = value_train
        self.val_test = value_test
        self.val_t0 = value_train[0]
        self.val_last = value_train[-1]
        self.t_train = t_train
        self.t_test = t_test
        self.t_array = t_train + t_test
        self.N_traj = N_traj

        # модели
        if models is None:
            self.models = ['m1', 'm2', 'm3', 'm4']
        else:
            self.models = models

        self.models_info = {
            'm1': '$a dt + b dW_t$',
            'm2': '$a X_t dt + b X_t dW_t$',
            'm3': 'CIR',
            'm4': 'FX'
        }

        # приращения винеровского процесса
        self.dW_N = self.Wiener_simulation() if dW_N is None else dW_N

        # параметры
        self.parametrs_func = {
            'm1': parametrs_estimation_1,
            'm2': parametrs_estimation_2,
            'm3': parametrs_estimation_CIR,
            'm4': parametrs_estimation_FX
        }
        self.paramters = {}

        # симуляции
        self.simulations_func = {
            'm1': process_1,
            'm2': process_2,
            'm3': process_CIR,
            'm4': process_FX
        }
        self.simulations = {}

        # метрика
        self.metric = 'rmse'
        self.res_metric = {}

        #для визуализации
        self.colors = {
            'm1': 'goldenrod',
            'm2': 'seagreen',
            'm3': 'steelblue',
            'm4': 'indianred',
            'factor': 'black'
        }

    def Wiener_simulation(self):
        w0=0
        dt_ = np.diff(self.t_array)
        wiener_traj = Wiener_process(w0, dt_, self.N_traj)
        dW_N = np.diff(wiener_traj, axis=1)
        return dW_N

    def find_parametrs(self):
        # для каждой стох. модели по train подбираются коээфициенты
        for m in self.models:
            self.paramters[m] = self.parametrs_func[m](self.val_train, np.diff(self.t_train))

    def make_simulations(self):
        self.find_parametrs()
        ct = len(self.t_train)-1
        for m in self.models:
            r_traj = []
            for dWt in self.dW_N[:, :ct]:
                r_tr = self.simulations_func[m](param=self.paramters[m], r_0=self.val_t0, dt=np.diff(self.t_train), dWt=dWt)
                r_traj.append(np.array(r_tr))
            self.simulations[m] = np.array(r_traj)

    def plot_simulations(self, fig_size=(18, 8)):
        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        for m in self.models:
            axs.plot(self.t_train, self.simulations[m].T, color=self.colors[m], linewidth=0.5)
            axs.plot(self.t_train, np.mean(self.simulations[m], axis=0), color=self.colors[m], linewidth=2, label=self.models_info[m])
        axs.plot(self.t_train, self.val_train, color=self.colors['factor'], linewidth=1.5, label='factor')
        axs.legend()
        axs.set_title(self.name)
        plt.show()

    def print_metric_result(self):
        print(self.metric, '\n')
        for m in self.models:
            print(f'{m}: {round(self.res_metric[m], 4)}')

    def find_best_model(self):
        min_m = min(self.res_metric.values())
        best_model = None
        for m in self.res_metric.keys():
            if self.res_metric[m] == min_m:
                best_model = m
        return best_model

    def plot_simulations_best(self, fig_size=(18, 8)):
        m = self.find_best_model()
        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        axs.plot(self.t_train, self.simulations[m].T, color=self.colors[m], linewidth=0.5)
        axs.plot(self.t_train, np.mean(self.simulations[m], axis=0), color=self.colors[m], linewidth=2, label=self.models_info[m])
        axs.plot(self.t_train, self.val_train, color=self.colors['factor'], linewidth=1.5, label='factor')
        axs.legend()
        axs.set_title(f'{self.name}, лучшая модель ({self.metric} = {round(self.res_metric[m], 2)})')
        plt.show()

    def choose_model(self, metric=None, print_metric=1, plot_all=1, plot_best=1, fig_size=(18, 8)):
        if metric is not None:
            self.metric = metric

        self.make_simulations()
        for m in self.models:
            self.res_metric[m] = calculate_metric(self.val_train, self.simulations[m], mode=self.metric)

        if print_metric == 1:
            self.print_metric_result()
        if plot_all == 1:
            self.plot_simulations(fig_size)
        if plot_best == 1:
            self.plot_simulations_best(fig_size)

    # моделирование значений на будущее
    def plot_simulations_future(self, r_traj, m, t, fig_size=(20, 5)):
        fig, axs = plt.subplots(1, 1, figsize=fig_size)
        axs.plot(self.t_test[:t], self.val_test[:t], color=self.colors['factor'], linewidth=1.5, label='factor')
        axs.plot(self.t_test[:t], r_traj.T, color=self.colors[m], linewidth=0.5)
        axs.legend()
        metr = calculate_metric(self.val_test[:t], r_traj, mode=self.metric)
        axs.set_title(f'{self.name}: моделирование будущих значений ({self.metric} = {round(metr, 2)})')
        plt.show()

    def future_simulation(self, n_steps=None, best_model=None, plot=0):
        if best_model is None:
            best_model = self.find_best_model()
        if n_steps is None:
            n_steps = len(self.t_test)
        ct = len(self.t_train)-1
        dt_ = np.diff(self.t_array[ct:ct+n_steps+1])
        dWts = self.dW_N[:, ct:ct+n_steps]
        r_traj = []
        for dWt in dWts:
            r_tr = self.simulations_func[best_model](param=self.paramters[best_model], r_0=self.val_last, dt=dt_, dWt=dWt)
            r_traj.append(np.array(r_tr[1:]))
        if plot == 1:
            self.plot_simulations_future(np.array(r_traj), best_model, n_steps)
        return np.array(r_traj)