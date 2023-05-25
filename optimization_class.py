#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from data_wrangle import data_wrangle as dw
from datetime import datetime, date, timedelta
import time

# Basics
import numpy as np
import pandas as pd
import yfinance as yf 
from scipy.stats import gmean

#Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Optimization part
from cvxopt import matrix
from cvxopt.solvers import qp
from sklearn.covariance import LedoitWolf
from pypfopt import risk_models, expected_returns, plotting, EfficientFrontier

# sclearn RandomForest - Returns, ML Metrics and tools
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.pipeline import make_pipeline

# GARCH - Volatility
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ## Optimization

# In[ ]:


def kelly_optimize(M_df:pd.DataFrame, C_df:pd.DataFrame, risk_free_rate)->pd.DataFrame:
    "objective function to maximize is: g(F) = r + F^T(M-R) - F^TCF/2"
    r = risk_free_rate
    M = M_df.to_numpy()
    C = C_df.to_numpy()

    n = M.shape[0]
    A = matrix(1.0, (1, n))
    b = matrix(1.0)
    G = matrix(0.0, (n, n))
    G[::n+1] = -1.0
    h = matrix(0.0, (n, 1))
    try:
        max_pos_size = float(0.99)
    except KeyError:
        max_pos_size = None
    try:
        min_pos_size = float(0)
    except KeyError:
        min_pos_size = None
    if min_pos_size is not None:
        h = matrix(min_pos_size, (n, 1))

    if max_pos_size is not None:
       h_max = matrix(max_pos_size, (n,1))
       G_max = matrix(0.0, (n, n))
       G_max[::n+1] = 1.0
       G = matrix(np.vstack((G, G_max)))
       h = matrix(np.vstack((h, h_max)))

    S = matrix((1.0 / ((1 + r) ** 2)) * C)
    q = matrix((1.0 / (1 + r)) * (M - r))
    sol = qp(S, -q, G, h, A, b)
    kelly = np.array([sol['x'][i] for i in range(n)])
    kelly = pd.DataFrame(kelly, index=C_df.columns, columns=['Weights'])
    return kelly


# In[ ]:


# Define function to calculate returns, volatility
def portfolio_annualized_performance(weights, mean_returns, cov_matrix, input_annual=False):
    # Given the avg returns, weights of equities calc. the portfolio return
    if input_annual:
        returns = np.sum(mean_returns*weights)
    else:
        returns = np.sum(mean_returns*weights) *252
    # Standard deviation of portfolio (using dot product against covariance, weights)
    # 252 trading days
    if input_annual:
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    else:
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns


# In[ ]:


def get_estimates_for_optimization(window, test_size, returns, predicted_returns):

    mean_pred_ret = pd.DataFrame(columns=predicted_returns.columns)
    mean_hist_ret = pd.DataFrame(columns=predicted_returns.columns)
    pred_cov = []
    hist_cov = []
    

    for i in range(-test_size, 0):  
        # Create rolling window to train the model
        # 100 - window size, -1 correction for lag of tech features
        
        y_train = returns.iloc[-window+1+i:i]
        
        next_day_pred = pd.DataFrame(predicted_returns.iloc[i]).T * 100 # delete 100 and in returns prediction
        
        train_window = pd.concat([y_train, next_day_pred]).iloc[1:]
        
        
        # ESTIMATION WITH PREDICTIONS
        daily_returns = expected_returns.mean_historical_return(train_window / 100, 
                                                                returns_data=True, 
                                                                compounding=True, 
                                                                frequency=52)
        cov_matrix_daily = risk_models.sample_cov(train_window / 100, 
                                                  returns_data=True, 
                                                  frequency=52)
        
        
        # ESTIMATION BASED ON HISTORIC DATA
        daily_returns_hist = expected_returns.mean_historical_return(y_train / 100, 
                                                                returns_data=True, 
                                                                compounding=True, 
                                                                frequency=52)
        cov_matrix_daily_hist = risk_models.sample_cov(y_train / 100, 
                                                  returns_data=True, 
                                                  frequency=52)
                
        
        # CONCATENATION
        mean_pred_ret = pd.concat([mean_pred_ret, pd.DataFrame(daily_returns).T], axis=0)
        mean_hist_ret = pd.concat([mean_hist_ret, pd.DataFrame(daily_returns_hist).T], axis=0)
        pred_cov.append(cov_matrix_daily)
        hist_cov.append(cov_matrix_daily_hist)

    
    
    mean_pred_ret.index = predicted_returns.iloc[-test_size:, :].index
    mean_hist_ret.index = predicted_returns.iloc[-test_size:, :].index
    
    return mean_pred_ret, mean_hist_ret, pred_cov, hist_cov

        


# In[ ]:


def calculate_performance(predicted_rets, hist_rets, pred_cov, hist_cov, risk_free_rate):
    assert len(predicted_rets) == len(hist_rets) == len(pred_cov) == len(hist_cov)
    
    sharp_weights = []
    kelly_weights = []
    for i in range(0, len(predicted_rets.index)):

        ef = EfficientFrontier(predicted_rets.iloc[i], pred_cov[i])
        ef.max_sharpe(risk_free_rate=risk_free_rate)
        weights = ef.clean_weights()
        weights = pd.DataFrame.from_dict(dict(zip(list(weights.keys()), list(weights.values()))), 
                               orient='index', 
                               columns=['Weights'])
        kelly = kelly_optimize(predicted_rets.iloc[i], pred_cov[i], risk_free_rate)
        
        
        sharp_weights.append(weights)
        kelly_weights.append(kelly)
    
    return sharp_weights, kelly_weights
        
        


# In[ ]:


def portfolio_performance(weights, ohlc, predicted_rets):
    
    portfolio_returns = []
    portfolio_standard_devs = []
    
    returns_to_prod = ohlc['Adj Close'].pct_change().dropna().loc[predicted_rets.index] + 1
    returns = np.array(ohlc['Adj Close'].pct_change().dropna().loc[predicted_rets.index])
    
    pred_start_pos = ohlc['Adj Close'].index.get_loc(predicted_rets.index[0])
    

    
    
    for i in range(0, len(weights)):
        
        cov_matrix = risk_models.sample_cov(ohlc['Adj Close'].iloc[:pred_start_pos+1+i], 
                                      returns_data=False, 
                                      frequency=1)
        
        portfolio_return = np.sum(returns[i] * np.array(weights[i].squeeze()))
        
        portfolio_returns.append(portfolio_return)
        
        std = np.sqrt(np.dot(np.array(weights[i].squeeze()).T, np.dot(cov_matrix, np.array(weights[i].squeeze()))))
        portfolio_standard_devs.append(std)
        
    
    portfolio_returns = pd.Series(portfolio_returns, index=predicted_rets.index, name='return')
    portfolio_standard_devs = pd.Series(portfolio_standard_devs, index=predicted_rets.index, name='std')
    portfolio_sd = portfolio_returns.std()
    realized_return = (portfolio_returns + 1).cumprod()[-1]
    
    return portfolio_returns, portfolio_sd, realized_return, portfolio_standard_devs
        

