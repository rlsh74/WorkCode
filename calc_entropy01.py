# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:42:19 2021

@author: Шевченко Роман
"""

import pandas as pd
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import math

tickers = [
'HON', 'MMM', 'GE', 'ITW', 'ABB','EMR', 'ETN', 'ROP', 'TT', 'CMI', 'PH', 'ROK', 'AME',
'OTIS', 'GNRC', 'IR', 'XYL', 'DOV', 'IEX', 'HWM', 'GGG', 'LII', 'NDSN', 'BLDP', 'AOS',
'PNR', 'DCI', 'MIDD', 'ITT', 'GTLS', 'RBC', 'FLS', 'RXN', 'CR', 'CW', 'CFX', 'GTES', 'WTS', 'KRNT',
'JBT', 'PSN', 'AIMC', 'FELE', 'HI', 'ATKR', 'TPIC', 'B', 'SPXC', 'FLOW', 'WBT', 'MWA', 'CSWI', 'HLIO',
'KAI', 'NPO', 'HSC', 'OFLX', 'TRS', 'TNC', 'RAVN', 'EPAC', 'SXI', 'XONE', 'GRC', 'AMSC', 'CYD','CIR',
'LDL','THR','LXFR'
]


benchmark = '^TNX'

ATR_PERIOD = 12
interval = 'd'
start_date = datetime(2020, 1, 1)
end_date = datetime(2021, 1, 27)

#comp_list = pd.read_excel('Real_Estate_list.xlsx')
#tickers = comp_list['Ticker'].tolist()

tickers = ['^GSPC']
tickers = ['HON']
tickers = ['GC=F', 'SI=F', '^TNX', 'CL=F', 'HG=F', 'NG=F', '6E=F', '6B=F', '6J=F', '6S=F', '6A=F', 'BTC-USD', 'ETH-USD', 'XRP-USD']

tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]
tickers = ['HON']

#comp_list = pd.read_excel('Real_Estate_list.xlsx')
#tickers = comp_list['Ticker'].tolist()


def ApproximateEntropy(U, m, r):
    # Function assigns a non-negative number to the time series 
    # with larger values corresponding to greater randomness or irregularity and
    # smaller values corresponding to to more instances of repetitive patterns of variation
    # m - positive interger represening the lenght of successive observations
    # r - positive real representing a tolerance level
    
    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m  + 1)]
        C = [len([1 for xj in x if _maxdist(xi, xj) <= r]) / (N - m + 1.0) for xi in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    
    N = len(U)
    return abs(_phi(m + 1) - _phi(m))


def get_vol(data, span=100, delta=1):
  # 1. compute returns of the form p[t]/p[t-1] - 1
  # 1.1 find the timestamps of p[t-1] values
  df0 = data.index.searchsorted(data.index - delta)
  df0 = df0[df0 > 0]
  # 1.2 align timestamps of p[t-1] to timestamps of p[t]
  df0 = pd.Series(data.index[df0-1],    
           index=data.index[data.shape[0]-df0.shape[0] : ])
  # 1.3 get values by timestamps, then compute returns
  df0 = data.loc[df0.index] / data.loc[df0.values].values - 1
  # 2. estimate rolling standard deviation
  df0 = df0.ewm(span=span).std()
  return df0

for ticker in tickers:

    try:

        df = pd.DataFrame()

        df['close1'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Close']
        df['close2'] = web.get_data_yahoo(benchmark, start=start_date, end=end_date, interval=interval)['Close']
        
        #dbf = get_vol(df, 14, 1)
        
        df.reset_index(inplace=True)
        
        K = 1
        #K = 15
        df['Target'] = np.where(df['close1'].shift(-K) > df['close1'], 1, 0)
        
        U = df['close1'].values
        #U = df['Target'].values
        
        
        
        print(ticker, ' ====== ', ApproximateEntropy(U, 5, 3))




    except:
        print('Cannot calculate for ticker: ', ticker)
        
    
        

