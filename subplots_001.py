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
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]
tickers = ['GC=F']
benchmark = 'DX=F'
industry_mark = '^TNX'
ATR_PERIOD = 12
interval = 'd'
start_date = datetime(2020, 1, 1)
end_date = datetime(2021, 2, 5)

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

        df['open3'] = web.get_data_yahoo(industry_mark, start=start_date, end=end_date, interval=interval)['Open']
        df['high3'] = web.get_data_yahoo(industry_mark, start=start_date, end=end_date, interval=interval)['High']
        df['low3'] = web.get_data_yahoo(industry_mark, start=start_date, end=end_date, interval=interval)['Low']
        df['close3'] = web.get_data_yahoo(industry_mark, start=start_date, end=end_date, interval=interval)['Close']
        df['volume3'] = web.get_data_yahoo(industry_mark, start=start_date, end=end_date, interval=interval)['Volume']
        
        #df.reset_index(inplace=True)

        #df = df.rename({'index':'Date'}, axis='columns')
        
        data = df[['close1', 'close2', 'close3']]
        
        #Первый вариант
        #data.plot(subplots=True, legend=False, lw=2, figsize=(15,8))
        
        #Второй вариант
        #fig, axes = plt.subplots(2, 1, figsize=(15,8))
        #df['close1'].plot(ax=axes[0])
        #f['close2'].plot(ax=axes[1], color='red')
        
        data['diff1'] = data['close1'].pct_change()
        data['diff2'] = data['close2'].pct_change()
        
        ax = data.plot.scatter(x='diff1', y='diff2', figsize=(15, 8), marker='$\u25EF$')
        ax.set_xlabel('Gold', fontsize=16)
        ax.set_ylabel('Dollar Index', fontsize=16)
        ax.axhline(0, color='grey', lw=1)
        ax.axvline(0, color='grey', lw=1)

    except:
        print('Cannot calculate for ticker: ', ticker)
        
    
        

