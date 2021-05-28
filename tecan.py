# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:01:34 2021

@author: Шевченко Роман
"""

#Library with technical indicators

import numpy as np
import pandas as pd

#Simple moving average (SMA)
def SMA(data, period=21, column='Close'):
    return data[column].rolling(window=period).mean()

#Exponencial moving average (EMA)
def EMA(data, period=21, column='Close'):
    return data[column].ewm(span=period, adjust = False).mean()

#Moving Average Convergence/Divergence (MACD):
def MACD(data, period_long=26, period_short=12, period_signal=9, column='Close'):
    
    ShortEMA = EMA(data, period_short, column=column)
    LongEMA = EMA(data, period_long, column=column)
    data['MACD'] = ShortEMA - LongEMA
    data['Signal_Line'] = EMA(data, period_signal, column='MACD')

    return data

#Relative Strength Index (RSI)
def RSI(data, period=14, column='Close'):
    delta = data[column].diff(1)
    delta = delta[1:]

    up = delta.copy()
    down = delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0

    data['Up'] = up
    data['Down'] = down

    AVG_Gain = SMA(data, period, column='Up')
    AVG_Loss = abs(SMA(data, period, column='Down'))

    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0/(1.0 + RS))
    
    data['RSI'] = RSI

    return data

#On Balance Volume (OBV)
def OBV(data, column_pr='Close', column_vol='Volume'):
    data['OBV'] = np.where(data[column_pr] > data[column_pr].shift(1), data[column_vol], np.where(data[column_pr] < data[column_pr].shift(1), -data[column_vol], 0)).cumsum()

    return data

def Bollinger(data, period=14, deviation=2, column='Close'):
    data['BB_up'] = data[column].rolling(period).mean() + deviation * data[column].rolling(period).std()
    data['BB_mid'] = data[column].rolling(period).mean()
    data['BB_dn'] = data[column].rolling(period).mean() - deviation * data[column].rolling(period).std()

    return(data)
    

def Price_ROC(data, period=14, column='Close'):
    
    if period >= 1:

        res = np.zeros(period)

        for i in range(period, len(data)):
            roc = np.array([(data[column].iloc[i] - data[column].iloc[i-period]) / data[column].iloc[i-period] * 100])
            res = np.append(res, roc)

        #prices['momentum'] = pd.Series(res_mom)
        #result = pd.Series(res)
        data['ROC'] = pd.Series(res)
    else:
        print('Period cannot be less than 1.')

    return(data)

def Williams_R(data, period=14, column='Close'):
   
    data['L_period'] = data['Low'].rolling(window=period).min()

    #Create the "H_period" column in the DataFrame
    data['H_period'] = data['High'].rolling(window=period).max()

    #Create the "%R" column in the DataFrame
    data['Williams'] = (data[column] - data['H_period']) / (data['H_period'] - data['L_period']) * 100

    data.drop(['L_period'], axis=1, inplace=True)
    data.drop(['H_period'], axis=1, inplace=True)

    return(data)

def Stochastic(data, period=14, period_ma=3, column='Close'):
   
    data['L_period'] = data['Low'].rolling(window=period).min()

    #Create the "H_period" column in the DataFrame
    data['H_period'] = data['High'].rolling(window=period).max()

    #Create the "%K" column in the DataFrame
    data['%K'] = 100*((data[column] - data['L_period']) / (data['H_period'] - data['L_period']))

    #Create the "%D" column in the DataFrame
    data['%D'] = data['%K'].rolling(window=period_ma).mean()

    data.drop(['L_period'], axis=1, inplace=True)
    data.drop(['H_period'], axis=1, inplace=True)

    return(data)

def ATR(data, period):
    
    for i in range(1, len(data)):
        d1 = data.loc[i, 'High'] - data.loc[i, 'Low']
        d2 = abs(data.loc[i, 'High'] - data.loc[i-1, 'Close'])
        d3 = abs(data.loc[i, 'Low'] - data.loc[i-1, 'Close'])
        data.loc[i, 'TR'] = max(d1, d2, d3).round(2)
    
    data['ATR'] =  data['TR'].rolling(period).mean()
    
    return data