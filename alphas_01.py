# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 16:57:22 2021

@author: Шевченко Роман
"""
import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def ATR(data, period):
    
    for i in range(1, len(data)):
        d1 = data.loc[i, 'High'] - data.loc[i, 'Low']
        d2 = abs(data.loc[i, 'High'] - data.loc[i-1, 'Close'])
        d3 = abs(data.loc[i, 'Low'] - data.loc[i-1, 'Close'])
        data.loc[i, 'TR'] = max(d1, d2, d3).round(2)
    
    data['ATR'] =  data['TR'].rolling(period).mean()
    
    return data

# Calculate z-score for ATR indicator
# p_atr - period for ATR (integer) 
# p_z - period for z-score (integer)
def Z_atr(data, p_atr, p_z, chart=False):

    ATR(data, p_atr)
    data['Z'] = (data['ATR'] - data['ATR'].rolling(p_z).mean()) / data['ATR'].rolling(p_z).std()
    
    data = data.dropna()
    data = data.reset_index()
    
    if chart:
        start = datetime.strftime(data.loc[0, 'Date'], '%Y-%m-%d')
        end = datetime.strftime(data.loc[len(data)-1, 'Date'], '%Y-%m-%d')

        plt.figure(figsize=(15,8))

        plt.subplot(2,1,1)
        plt.plot(data.index, data['Adj Close'])
        plt.title(f'{ticker} from {start} to {end}')
        plt.grid(True)

        plt.subplot(2,1,2)
        plt.plot(data.index, data['Z'], c='r')
        #plt.plot(data.index, data['ZSMA'], '--', c='r')
        plt.hlines(2, min(data.index), max(data.index), linestyles='dashdot')
        plt.hlines(-2, min(data.index),  max(data.index), linestyles='dashdot')
        plt.title(f'Z-score({PERIOD_Z}) from ATR({PERIOD_ATR})')
        plt.grid(True)
    return data


PERIOD_ATR = 14
PERIOD_Z = 30

start_date = '2019-08-21'
end_date = '2021-03-31'
interval = 'd'

tickers = ['SVM']
tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]

tickers = [
'HON', 'MMM', 'GE', 'ITW', 'ABB','EMR', 'ETN', 'ROP', 'TT', 'CMI', 'PH', 'ROK', 'AME',
'OTIS', 'GNRC', 'IR', 'XYL', 'DOV', 'IEX', 'HWM', 'GGG', 'LII', 'NDSN', 'BLDP', 'AOS',
'PNR', 'DCI', 'MIDD', 'ITT', 'GTLS', 'RBC', 'FLS', 'RXN', 'CR', 'CW', 'CFX', 'GTES', 'WTS', 'KRNT',
'JBT', 'PSN', 'AIMC', 'FELE', 'HI', 'ATKR', 'TPIC', 'B', 'SPXC', 'FLOW', 'WBT', 'MWA', 'CSWI', 'HLIO',
'KAI', 'NPO', 'HSC', 'OFLX', 'TRS', 'TNC', 'RAVN', 'EPAC', 'SXI', 'XONE', 'GRC', 'AMSC', 'CYD','CIR',
'LDL','THR','LXFR'
]

tickers = ['SVM']

        
for ticker in tickers:
    try:
        data = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)
        data = data.reset_index()
        Z_atr(data, PERIOD_ATR, PERIOD_Z, chart=True)
        print(f'=========== {ticker} ===========')
    except:
        print(f'Cannot calculate for: {ticker}')
    
    
