# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:18:27 2021

@author: Шевченко Роман
"""

# Methods and functions for stock selection: Price Quintiles and Z-score ATR

import pandas as pd
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import quantstats as qs

def ATR(data, period):
    
    for i in range(1, len(data)):
        d1 = data.loc[i, 'High'] - data.loc[i, 'Low']
        d2 = abs(data.loc[i, 'High'] - data.loc[i-1, 'Close'])
        d3 = abs(data.loc[i, 'Low'] - data.loc[i-1, 'Close'])
        data.loc[i, 'TR'] = max(d1, d2, d3).round(2)
    
    data['ATR'] =  data['TR'].rolling(period).mean()
    
    return data

end = datetime.today()
start = end - timedelta(days=426)

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

tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]

tickers = ['PAAS']

PERIOD_ATR = 14
PERIOD_Z = 30

PLOT = True

scan_results = pd.DataFrame()

for ticker in tickers:
    try:
        data = web.get_data_yahoo(ticker, start, end)
        data = data.reset_index()
        
        print(f'====== {ticker} ======')
        
        data['Quintile'] = pd.qcut(data['Close'], 5, labels=False)
        ATR(data, PERIOD_ATR)
        data['Z'] = (data['ATR'] - data['ATR'].rolling(PERIOD_Z).mean()) / data['ATR'].rolling(PERIOD_Z).std()
        data['Logreturn'] = np.log(data['Adj Close']/data['Adj Close'].shift(1))
        data = data.dropna()
        #data = data.reset_index()
        
        last_quintile = data.tail(1)['Quintile'].values[0]
        last_z = round(data.tail(1)['Z'].values[0], 2)

        new_row = {'ticker': ticker, 'Quintile': last_quintile, 'Z': last_z}
        scan_results = scan_results.append(new_row, ignore_index=True)
        
        if PLOT:
            
            data = data.set_index('Date')
       
            plt.figure(figsize=(15,10))

            plt.subplot(3,1,1)
            plt.plot(data['Adj Close'])
            plt.title(ticker)
            plt.grid(True)
        
            plt.subplot(3,1,2)
            plt.plot(data['Quintile'], c='g')
            plt.grid(True)
        
            plt.subplot(3,1,3)
            plt.plot(data['Z'], c='r')
            plt.hlines(2, min(data.index), max(data.index), linestyles='dashdot')
            plt.hlines(-2, min(data.index),  max(data.index), linestyles='dashdot')
            plt.title(f'Z-score({PERIOD_Z}) from ATR({PERIOD_ATR})')
            plt.grid(True)

            plt.show()
            
            #qs.plots.drawdown(data['Logreturn'])
        
        
    except:
        print(f'Cannot calculate for {ticker}!')

scan_results.to_excel('_scan_results.xlsx')