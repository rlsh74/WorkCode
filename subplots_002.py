# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 08:52:42 2021

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

#tickers = ['HON']
benchmark = 'DX=F'
#benchmark = '^TNX'
industry_mark = 'XLI'
ATR_PERIOD = 12
interval = 'd'
start_date = datetime(2020, 1, 1)
end_date = datetime(2021, 3, 4)

comp_list = pd.read_excel('Real_Estate_list.xlsx')
tickers = comp_list['Ticker'].tolist()

for ticker in tickers:

    try:

        df = pd.DataFrame()

        df['open1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Open']
        df['high1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['High']
        df['low1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Low']
        df['close1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Close']
        df['volume1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Volume']

        df['open2'] = web.get_data_yahoo(benchmark, start=start_date, end=end_date, interval=interval)['Open']
        df['high2'] = web.get_data_yahoo(benchmark, start=start_date, end=end_date, interval=interval)['High']
        df['low2'] = web.get_data_yahoo(benchmark, start=start_date, end=end_date, interval=interval)['Low']
        df['close2'] = web.get_data_yahoo(benchmark, start=start_date, end=end_date, interval=interval)['Close']
        df['volume2'] = web.get_data_yahoo(benchmark, start=start_date, end=end_date, interval=interval)['Volume']

       
        df.reset_index(inplace=True)

        df = df.rename({'index':'Date'}, axis='columns')
        
        data = df[['close1', 'close2']]
        
        #Первый вариант
        #data.plot(subplots=True, legend=False, lw=2, figsize=(15,8))
        
        #Второй вариант
        #fig, axes = plt.subplots(2, 1, figsize=(15,8))
        #df['close1'].plot(ax=axes[0])
        #f['close2'].plot(ax=axes[1], color='red')
        
        #data['diff1'] = df['close1'].pct_change()
        #data['diff2'] = df['close2'].pct_change()
        
        #ax = data.plot.scatter(x='diff1', y='diff2', figsize=(15, 8), marker='$\u25EF$')
        #ax.set_xlabel('Gold', fontsize=16)
        #ax.set_ylabel('Dollar Index', fontsize=16)
        #ax.axhline(0, color='grey', lw=1)
        #ax.axvline(0, color='grey', lw=1)
        
        ax = data[['close1', 'close2']].plot(figsize=(14, 4),
                                        rot=0,
                                        secondary_y = 'close2', style=['-', '--'],
                                        title=ticker + ' with ' + benchmark)
        print(ticker)
        
    except:
        print('Cannot calculate for ticker: ', ticker)
        
    
        

