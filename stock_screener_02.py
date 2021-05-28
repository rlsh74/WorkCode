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

#Create functions to make the SMA and EMA
def SMA(data, period = 30, column='Close'):
    return data[column].rolling(window=period).mean()

#Create the function to calculate the RSI
def RSI(data, period = 14, column='Close'):
    delta = data[column].diff(1)
    delta = delta.dropna()
    up = delta.copy()
    down = delta.copy()
    up[up<0] = 0
    down[down>0] = 0
    data['Up'] = up
    data['Down'] = down
    AVG_Gain = SMA(data, period, column='Up')
    AVG_Loss = abs(SMA(data, period, column='Down'))
    RS = AVG_Gain / AVG_Loss
    RSI = 100.0 - (100.0 / (1.0 + RS))
    
    data['RSI'] = RSI
    return data

end = datetime.today()
#start = end - timedelta(days=426) #Year
start = end - timedelta(days=252) #Half-year

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


#comp_list = pd.read_excel('List_industrials.xlsx')
#tickers = comp_list['Ticker'].tolist()

tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]


#comp_list = pd.read_excel('Real_Estate_list.xlsx')
#tickers = comp_list['Ticker'].tolist()

#comp_list = pd.read_csv('russell_3000.csv', delimiter=';')
#tickers = comp_list['A'].tolist()


# Portfolio sa of 08.04.2021
#tickers = ['AEM', 'AUY', 'AYI', 'EGO', 'EQR', 'HOLI', 'IAG', 'JOBS', 'KGC', 'PRG']

#comp_list = pd.read_excel('Consumer_Cyclical_list.xlsx')
#tickers = comp_list['Ticker'].tolist()

#comp_list = pd.read_excel('Basic_Materials_list.xlsx')
#tickers = comp_list['Ticker'].tolist()

tickers = ['WSC', 'AM']


PERIOD_ATR = 14
PERIOD_Z = 30
PERIOD_RSI = 14

PLOT = False
PLOT = True


scan_results = pd.DataFrame()

for i, ticker in enumerate(tickers):
    try:
        data = web.get_data_yahoo(ticker, start, end)
        data = data.reset_index()
        
        print(f'{i} ====== {ticker} ======')
        
        data['Quintile'] = pd.qcut(data['Close'], 5, labels=False)
        RSI(data, PERIOD_RSI, 'Close')
        ATR(data, PERIOD_ATR)
        data['Z'] = (data['ATR'] - data['ATR'].rolling(PERIOD_Z).mean()) / data['ATR'].rolling(PERIOD_Z).std()
        data['Logreturn'] = np.log(data['Adj Close']/data['Adj Close'].shift(1))
        data['RSI_q'] = pd.qcut(data['RSI'], 7, labels=False) #quantile for RSI
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
            plt.title(f'Z-score({PERIOD_Z}) from ATR({PERIOD_ATR})', fontsize=10)
            plt.grid(True)
            
            plt.figure(figsize=(15,8))

            plt.subplot(2,1,1)
            plt.plot(data['Adj Close'])
            plt.title(f'{ticker} {end}')
            plt.grid(True)
        
            plt.subplot(2,1,2)
            plt.plot(data['RSI'], c='r')
            plt.hlines(75, min(data.index), max(data.index), linestyles='dashdot')
            plt.hlines(25, min(data.index),  max(data.index), linestyles='dashdot')
            plt.title(f'RSI({PERIOD_RSI})', fontsize=10)
            plt.grid(True)          
            
            plt.figure(figsize=(15,8))

            plt.subplot(2,1,1)
            plt.plot(data['Adj Close'])
            plt.title(f'{ticker} {end}')
            plt.grid(True)
        
            plt.subplot(2,1,2)
            
            plt.scatter(data.index, data['RSI_q'], c='r')
            #plt.hlines(75, min(data.index), max(data.index), linestyles='dashdot')
            #plt.hlines(25, min(data.index),  max(data.index), linestyles='dashdot')
            plt.title(f'RSI_Q({PERIOD_RSI})', fontsize=10)
            plt.grid(True)            

            plt.show()
            
            qs.plots.drawdown(data['Logreturn'])
        
        
    except:
        print(f'Cannot calculate for {ticker}!')

scan_results.to_excel('_scan_results.xlsx', index=False)