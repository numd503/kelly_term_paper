#!/usr/bin/env python
# coding: utf-8

# In[1]:


from data_wrangle import data_wrangle as dw
from datetime import datetime, date, timedelta
import time
import itertools
import pickle

# Basics
import numpy as np
import pandas as pd
import yfinance as yf 


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


# In[2]:


class model_validation:
    
    def __init__(self, tickers, start, end, features_tickers):
        self.tickers = tickers
        self.start = start
        self.end = end

        self.features = yf.download(features_tickers, start=self.start, end=self.end, 
                                    interval='1wk').fillna(method='ffill')
        self.ohlc = yf.download(self.tickers, start=self.start, end=self.end, interval='1wk')
    
   
    def tech_features(self):
        inst = dw()
        output = pd.DataFrame()
        for i in self.ohlc.columns.levels[1]:

            tech = inst.get_tech_features(close=self.ohlc['Adj Close'][i],
                                              volume=self.ohlc['Volume'][i],
                                              low=self.ohlc['Low'][i],
                                              high=self.ohlc['High'][i],
                                              concat=True).dropna()
            tech.columns = pd.MultiIndex.from_product([[i], tech.columns])

            output = pd.concat([output, tech], axis=1)
            
        self.ohlc = self.ohlc.loc[output.index]
        self.features = self.features['Adj Close'].loc[output.index]
        self.tech = output
        print('Done!')
        

    def binary_classificator(self, x):
        if x > 0:
            return 1
        else:
            return -1
        
    
    def binary_map(self, prices):
        output = pd.DataFrame()
        for i in prices.columns:
            # binary classification of returns
            
            up_down = pd.DataFrame(prices[i].pct_change().dropna())
            up_down[i] = up_down[i].map(model_validation.binary_classificator)


            # concatenate all companies in dataframe
            output = pd.concat([output, up_down], axis=1)

        return output
    
    
    
    def train_test_split(self, window, val_size, test=False, test_size=False):
        self.window = window
        self.val_size = val_size

        
        self.ohlc_val = self.ohlc[:self.window + self.val_size]
        self.tech_val = self.tech[:self.window + self.val_size]
        self.feat_val = self.features[:self.window + self.val_size]
        self.binary_val = model_validation.binary_map(self.ohlc['Adj Close'])[:self.window + self.val_size - 1]
        self.returns_val = 100 * self.ohlc['Adj Close'].pct_change().dropna()[:self.window + self.val_size - 1]

        print('Initial training window is included')
        print(f'ohlc val {len(self.ohlc_val)}, tech val {len(self.tech_val)}')
        print(f'binary_val {len(self.binary_val)}, returns_val {len(self.returns_val)}')
        assert len(self.ohlc_val) == len(self.tech_val)
        assert len(self.binary_val) == len(self.returns_val)
        assert len(self.ohlc_val) == len(self.binary_val) + 1
        assert self.ohlc_val.index[1] == self.binary_val.index[0]
        
        assert self.feat_val.index[0] == self.tech_val.index[0]

        if test:
            
            self.ohlc_test = self.ohlc[self.window + self.val_size:]
            self.tech_test = self.tech[self.window + self.val_size:]
            self.feat_test = self.features[self.window + self.val_size:]
            self.binary_test = model_validation.binary_map(self.ohlc['Adj Close'])[self.window + self.val_size :]
            self.returns_test = 100 * self.ohlc['Adj Close'].pct_change().dropna()[self.window + self.val_size :]
            self.test_size = len(self.ohlc_test) - self.window
            
            print('Initial training window is included')
            print(f'ohlc test {len(self.ohlc_test)}, tech test {len(self.tech_test)}')
            print(f'binary_test {len(self.binary_test)}, returns_test {len(self.returns_test)}')
            
            assert len(self.ohlc_test) == len(self.tech_test)
            assert len(self.binary_test) == len(self.returns_test)
            assert len(self.ohlc_test) == len(self.binary_test) + 1
            assert self.ohlc_test.index[1] == self.binary_test.index[0]
            assert self.feat_test.index[0] == self.tech_test.index[0]
        
        
    def classify_next_bar(self, X, y, RF_params=(10, None, None), features=None):
        """
        X, y - features and target-vector based on a shape of rolling window
        X - tech features and external data for the date t-1
        y - binary target - [1 - up, -1 - down] for time t
        X_out - 1 day forward features to make prediction
        Make assert statement to ensure that
        
        """
        assert X.shape[0] - 1 == y.shape[0]
        assert len(X) == len(features)
        X_data = pd.concat([X, features], axis=1)
        X_train_data = X_data.shift().dropna()
        
        assert X_train_data.shape[0] == y.shape[0]
        
        X_forecast = pd.DataFrame(X_data.iloc[-1, :]).T
        
        # Fit the model - returns classifier
        model = RandomForestClassifier(n_estimators=RF_params[0], 
                                       max_depth=RF_params[1], 
                                       max_leaf_nodes=RF_params[2], 
                                       random_state=42,
                                       n_jobs=-1)
        model.fit(X_train_data, y)
        bar = model.predict(X_forecast)
        
        return bar
    
    def GARCH_model(self, returns, GARCH_params=(1, 1), summary=False):
        assert type(returns) == pd.Series
        model = arch_model(
            returns,
            p=GARCH_params[0],
            q=GARCH_params[1],
            rescale=False
        )

        model_res = model.fit(disp=0)
        next_pred = model_res.forecast(horizon=1, reindex=False).variance.iloc[0,0] ** 0.5
        
        if summary:
            aic = model_res.aic
            return next_pred, aic

        return next_pred
    
    
    
    def rolling_window(self, tech, binary, window, test_size, returns, 
                       val_model='Both', RF_params=(10, None, None), 
                       GARCH_params=(1, 1), summary_garch=False, features=None):
        """
        val_model: what model to validate. Relevant parameters: 'RF', 'GARCH', 'Both'
        
        """
        if val_model == 'Both':


            prediction_ret = pd.DataFrame(columns=binary.columns)
            prediction_std = pd.DataFrame(columns=returns.columns)

            total_time = time.time()

            for i in range(-test_size, 0):
                start_time = time.time()
                # Create rolling window to train the model
                # 100 - window size, -1 correction for lag of tech features
                X_train = tech.iloc[-window-1+i:i]
                y_train = binary.iloc[-window+i:i]

                y_train_ret = returns.iloc[-window+i:i]

                # RETURNS PART
                #list for returns predictions
                pred_list = np.array([])

                #VOLATILITY PART
                pred_list_std = np.array([])

                for company in binary.columns:
                    # RETURNS PART
                    # Train a model
                    #print(f' Start {company}')

                    next_pred = model_validation.classify_next_bar(X_train[company], y_train[company], RF_params)
                    # Collect results and append them
                    pred_list = np.append(pred_list, next_pred)


                    #VOLATILITY PART
                    # Train a model
                    next_pred_std = model_validation.GARCH_model(y_train_ret[company], GARCH_params, summary=summary_garch)
                    # Collect results and append them
                    pred_list_std = np.append(pred_list_std, next_pred_std)

                    #print(f' End {company}')

                pred_list = pd.DataFrame(pred_list, index=binary.columns).T
                prediction_ret = pd.concat([prediction_ret, pred_list], axis=0)

                pred_list_std = pd.DataFrame(pred_list_std, index=returns.columns).T
                prediction_std = pd.concat([prediction_std, pred_list_std], axis=0)
                elapsed_time = round(time.time() - start_time, 2)
                print(f"Test circle {i} trained in {elapsed_time} seconds.")



            prediction_ret.index = binary.iloc[-test_size:, :].index
            prediction_std.index = returns.iloc[-test_size:, :].index

            elapsed_total_time = round(time.time() - total_time, 2)
            print(f"Model trained in {elapsed_total_time} seconds.")

            return prediction_ret, prediction_std
        
        if val_model == 'RF':

            prediction_ret = pd.DataFrame(columns=binary.columns)

            total_time = time.time()

            for i in range(-test_size, 0):
                start_time = time.time()
                # Create rolling window to train the model
                # 100 - window size, -1 correction for lag of tech features
                X_train = tech.iloc[-window-1+i:i]
                if features is None:
                    pass
                else:
                    X_features = features.iloc[-window-1+i:i]
                    assert len(X_train) == len(X_features)
                
                y_train = binary.iloc[-window+i:i]

                # RETURNS PART
                #list for returns predictions
                pred_list = np.array([])

                for company in binary.columns:
                    # RETURNS PART
                    # Train a model
                    if features is None:
                        next_pred = model_validation.classify_next_bar(X_train[company], y_train[company], 
                                                                   RF_params)
                    else:
                        next_pred = model_validation.classify_next_bar(X_train[company], y_train[company], 
                                                                   RF_params, features=X_features)
                    
                    # Collect results and append them
                    pred_list = np.append(pred_list, next_pred)

                pred_list = pd.DataFrame(pred_list, index=binary.columns).T
                prediction_ret = pd.concat([prediction_ret, pred_list], axis=0)

                elapsed_time = round(time.time() - start_time, 2)
                print(f"Test circle {i} trained in {elapsed_time} seconds.")

            prediction_ret.index = binary.iloc[-test_size:, :].index

            elapsed_total_time = round(time.time() - total_time, 2)
            print(f"Model trained in {elapsed_total_time} seconds.")

            return prediction_ret
        
        
        if val_model == 'GARCH':
            if summary_garch:
                aic = {}
            
            prediction_std = pd.DataFrame(columns=returns.columns)

            total_time = time.time()

            for i in range(-test_size, 0):
                start_time = time.time()
                # Create rolling window to train the model

                y_train_ret = returns.iloc[-window+i:i]

                #VOLATILITY PART
                pred_list_std = np.array([])

                for company in binary.columns:

                    #VOLATILITY PART
                    # Train a model
                    if summary_garch:
                        next_pred_std, aic[(i, company)] = model_validation.GARCH_model(y_train_ret[company], 
                                                                     GARCH_params, 
                                                                     summary=True)
                    else:
                        next_pred_std = model_validation.GARCH_model(y_train_ret[company], 
                                                                     GARCH_params, 
                                                                     summary=False)
                    
                    
                    
                    # Collect results and append them
                    pred_list_std = np.append(pred_list_std, next_pred_std)
                    
                pred_list_std = pd.DataFrame(pred_list_std, index=returns.columns).T
                prediction_std = pd.concat([prediction_std, pred_list_std], axis=0)
                elapsed_time = round(time.time() - start_time, 2)
                print(f"Test circle {i} trained in {elapsed_time} seconds.")

            prediction_std.index = returns.iloc[-test_size:, :].index

            elapsed_total_time = round(time.time() - total_time, 2)
            print(f"Model trained in {elapsed_total_time} seconds.")
            
            if summary_garch:
                return prediction_std, aic
            
            return prediction_std            
              
        
        
    def get_return_predictions(self, tech, binary, window, test_size, returns):
        direction, standard_dev = model_validation.rolling_window(tech, binary, window, test_size, returns)
        returns_prediction = direction * standard_dev

        return returns_prediction / 100, direction
        
        
    
    

    
    


# In[3]:


# tickers = ['XOM', 'CVX', 'SHEL']
# features = ["EURUSD=X", '^IRX', '^FVX', 'CL=F', 'GC=F', 'NG=F', 'RB=F']
# start = '2005-01-01'
# end = '2022-12-31'


# In[4]:


model_validation = model_validation(tickers, start, end, features)


# In[5]:


model_validation.tech_features()


# In[6]:


# window = 52 * 4
# test_size = len(model_validation.tech) - window


# In[7]:


# model_validation.train_test_split(window=window, val_size=test_size, test=False)


# In[ ]:





# In[ ]:




