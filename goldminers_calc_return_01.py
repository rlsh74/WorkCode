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
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]
#tickers = ['GC=F', 'GLD', 'GDX', '^GSPC']

#tickers = ['NEM']

tickers = [
'NEM',  
'GOLD',
'PLZL.ME',  
'WPM',  
'AEM',  
'NCM.AX',  
'SBSW',  
'GFI',  
'LUN.TO',  
'NST.AX',  
'EVN.AX',  
'BTG',  
'AUY',  
'SAR.AX',  
'SSRM',  
'EDV.TO',  
'CG.TO',  
'NG',  
'AGI',  
'HMY',  
'EQX.TO',  
'CDE',  
'EGO',  
'TGZ.TO',  
'IAG',  
'POG.L',  
'SA',  
'SBM.AX',  
'OGC.TO',  
'PRU.AX',  
'NGD',  
'DPM.TO',  
'TXG.TO',  
'ANTM.JK',  
'RRL.AX',  
'WDO.TO',  
'DRD',  
'GOR.AX',  
]

#tickers = ['NEM']

start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 31)

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
        
        df['Return'] = df['close1'].pct_change()
        daily_std = np.std(df['Return'])
        std_ann = daily_std * 252 ** 0.5
        
        #hv = HV(df, 30)
        
        #Plot the chart of HV for 30 periods       
        #n_periods = 30
        #vol = df['close1'].pct_change().rolling(n_periods).std() * np.sqrt(n_periods)
        #vol.plot(figsize=(10, 10), title='HV')
        
                
        #df['Vola'] = df['Diff'].rolling(n_periods).std()*np.sqrt(n_periods)

        #print(ticker, ' ###### ', round(diff, 2)) #return for period (%)
        #print(ticker, ' ###### ', round(p2, 2)) # close price for period
        print(ticker, ' ###### ', round(std_ann, 4)) #annualized HV
        #print(hv)
  

    except:
        print('Cannot calculate for ticker: ', ticker)