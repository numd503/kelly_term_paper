#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf

class data_wrangle:
    def __init__(self):
        pass
    
    def get_bars(self, ticker, period='1mo', interval='1d', start=None, end=None):
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval, start=start, end=end)
        return data
    
    def save_bars(self, ticker, period='1mo', interval='1d', start=None, end=None):
        """
        download data from yfinance and save it in the same direction 
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval, start=start, end=end)
        data.to_csv(f'{ticker}_stock_prices.csv')
        pass
    
    def read_bars(self, path):
        """
        Read data from yfinance Ticker.history previously saved as csv
        """
        data = pd.read_csv(path)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        data.set_index('Date', inplace=True)
        return data
    def get_and_save_bars(self, ticker, period='1mo', interval='1d', start=None, end=None):
        """
        Download data from yfinance, save it in the same direction and return dataframe
        """
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval, start=start, end=end)
        data.to_csv(f'{ticker}_stock_prices.csv')
        return data
    
    def train_test_split(self, data, test_size=0.2, val_set=None):
        
        split_index = int(len(data) * test_size)
        train_set = data.loc[data.index[:-split_index]]
        test_set = data.loc[data.index[len(data)-split_index:]]
        
        
        return train_set, test_set
        
    
    def get_tech_features(self, close, volume=None, low=None, high=None, alpha=0.2, K=14, concat=False):
        
        # Simple exponential smoothing
        t_0 = np.array(close)[1:]
        t_1 = np.array(close.shift().dropna())
        res_array = np.insert((t_0 * alpha + t_1 * (1 - alpha)), 0, close[0])

        SES = pd.Series(res_array, index=close.index, name='SES')
        
        
        # MACD and MACD-Signal part
        MA_Fast = close.ewm(span=12,min_periods=12).mean()
        MA_Slow = close.ewm(span=26,min_periods=26).mean()
        MACD = pd.Series(MA_Fast - MA_Slow, name='MACD')
        Signal = MACD.ewm(span=9,min_periods=9).mean().rename('MACD-Signal')
        
        # On Balance Volume
        if volume is None:
            pass
        else:
                OBV = pd.Series(index=range(len(volume.index)), dtype='float', name='OBV')
                OBV.iloc[0] = 0
                for i in range(1, len(SES)):
                    if SES[i] > SES[i-1]:
                        OBV.loc[i] = (OBV.loc[i-1] + volume[i])
                    elif SES[i] < SES[i-1]:
                        OBV.loc[i] = (OBV.loc[i-1] - volume[i])
                    else:
                        OBV.loc[i] = (OBV.loc[i-1] + 0)
                OBV.index = volume.index
        
        # stochastic oscillator
        # remember to not use AdjClose
        low_min = low.rolling(K).min()
        high_max = high.rolling(K).max()
        SO = pd.Series(100*((close - low_min)/(high_max - low_min)), index=close.index, name=f'SO_K{K}')

        
        if concat == True:
            return pd.concat([SES, MACD, Signal, OBV, SO], axis=1)
        if concat == False:
            return SES, MACD, Signal, OBV, SO
        
    
    def add_risk_free(self, cov_matrix, returns=None, risk_free=0.02):
        # add row and column of zeroes representing risk-free asset
        array_v = pd.Series([0] * (len(cov_matrix)), index=cov_matrix.index, name='FREE')
        mid = pd.concat([cov_matrix, array_v], axis=1)
        array_h = pd.DataFrame([0] * (len(mid.columns)), index=mid.columns, columns=['FREE']).T
        output = pd.concat([mid, array_h], axis=0).astype(float)

        # add risk-free asset in returns vector
        if returns is None:
            return output
        else:
            returns['FREE'] = risk_free
            return output, returns

        
        
        
    

