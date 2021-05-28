# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:14:24 2021

@author: Шевченко Роман
"""

import pandas as pd
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


tickers = [
'HON', 'MMM', 'GE', 'ITW', 'ABB','EMR', 'ETN', 'ROP', 'TT', 'CMI', 'PH', 'ROK', 'AME',
'OTIS', 'GNRC', 'IR', 'XYL', 'DOV', 'IEX', 'HWM', 'GGG', 'LII', 'NDSN', 'BLDP', 'AOS',
'PNR', 'DCI', 'MIDD', 'ITT', 'GTLS', 'RBC', 'FLS', 'RXN', 'CR', 'CW', 'CFX', 'GTES', 'WTS', 'KRNT',
'JBT', 'PSN', 'AIMC', 'FELE', 'HI', 'ATKR', 'TPIC', 'B', 'SPXC', 'FLOW', 'WBT', 'MWA', 'CSWI', 'HLIO',
'KAI', 'NPO', 'HSC', 'OFLX', 'TRS', 'TNC', 'RAVN', 'EPAC', 'SXI', 'XONE', 'GRC', 'AMSC', 'CYD','CIR',
'LDL','THR','LXFR'
]
     
#tickers = ['XLI', 'GLD', 'GDX', '^GSPC']


start_date = datetime(2020, 1, 1)
end_date = datetime(2020, 12, 31)

def HV(data, column='close1'):
    logreturns = np.log(data[column] / data[column].shift(1))
    return np.sqrt(252 * logreturns.var())

for ticker in tickers:

    try:

        df = pd.DataFrame()

        df['open1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval='d')['Open']
        df['high1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval='d')['High']
        df['low1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval='d')['Low']
        df['close1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval='d')['Close']
        df['volume1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval='d')['Volume']

        #df['Ticker'] = ticker

        #df.reset_index(inplace=True)

        #df = df.rename({'index':'Date'}, axis='columns')

        p1 = df['close1'][0]
        p2 = df['close1'][-1]

        diff = (p2/p1-1)*100
        
        #1st variant
        df['Return'] = df['close1'].pct_change()
        daily_std = np.std(df['Return'])
        std_ann = daily_std * 252 ** 0.5
        
        #2nd variant
        df['Log_return'] = (np.log(df['close1'] / df['close1'].shift(-1)))
        daily_std = np.std(df['Log_return'])
        std_new = daily_std * 252 ** 0.5
        
        
        
        #hv = HV(df, 30)
        
        #Plot the chart of HV for 30 periods       
        #n_periods = 30
        #vol = df['close1'].pct_change().rolling(n_periods).std() * np.sqrt(n_periods)
        #vol.plot(figsize=(10, 10), title=ticker)
        
        #hv = HV(df)
                
                
        #df['Vola'] = df['Diff'].rolling(n_periods).std()*np.sqrt(n_periods)

        #print(ticker, ' ###### ', round(diff, 2)) #return for period (%)
        #print(ticker, ' ###### ', round(p2, 2)) # close price for period
        print(ticker, ' ###### ', round(std_ann*100, 4)) #annualized HV
        #print(hv)
  

    except:
        print('Cannot calculate for ticker: ', ticker)