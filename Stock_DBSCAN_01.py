# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:39:54 2021

@author: Шевченко Роман
"""
# Clustering stock data with DBSCAN

import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D

import mplfinance as mpf 

from sklearn.cluster import DBSCAN


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


tickers = ['JD']

comp_list = pd.read_excel('Candidates_0421.xlsx')
tickers = comp_list['Ticker'].tolist()


end_date = datetime.today()

start_date = end_date - timedelta(days=1150)
#start_date = datetime(2018, 1, 10) #for daily n=60
#start_date = datetime(2011, 1, 1) #for weekly n=60
#end_date = datetime(2021, 2, 14)

MY_PATH = './/Tempdata//'

N_PERIODS = 252 #number of periods for test (out of sample data) daily
#N_PERIODS = 60 #number of periods for test (out of sample data) weekly
#interval = 'w'
interval = 'd'

#Create functions to make the SMA and EMA
def SMA(data, period = 30, column='Close'):
    return data[column].rolling(window=period).mean()

def EMA(data, period = 20, column='Close'):
    return data[column].ewm(span=period, adjust=False).mean()

#Create the function to calculate the MACD
def MACD(data, period_long = 26, period_short=12, period_signal=9, column='Close'):
    shortEMA = EMA(data, period=period_short, column=column)
    longEMA = EMA(data, period=period_long, column=column)
    #Calculate and store the MACD into the dataframe
    data['MACD'] = shortEMA - longEMA
    #Calculate the signal line and store it into the data frame
    data['Signal_line'] = EMA(data, period=period_signal, column='MACD')
    
    return data

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

def Williams_R(data, period = 14, column='Close'):
    data['L_period'] = data['Low'].rolling(window=period).min()
    data['H_period'] = data['High'].rolling(window=period).max()
    data['Williams_R'] = (data[column] - data['H_period']) / (data['H_period'] - data['L_period']) * 100
    
    return(data)

def LabelsChart(data, ticker):
    #ticker = data['Ticker'][0]
    data = data[-250:]
  
    ax = data[['Close', 'Labels']].plot(figsize=(14, 4),
                                   rot=0,
                                   secondary_y = 'Labels', style=['-', '--'],
                                   title=ticker)


for ticker in tickers:
    try:
        
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)
        df = df.reset_index()
        
    
        #Add the indicators to the dataset
        df['SMA'] = SMA(df)
        df['EMA'] = EMA(df)
        MACD(df)
        RSI(df)
        Williams_R(df)
        
        NP = 13
        NP1 = 13
        df['Ticker'] = ticker
        df['Return'] = np.log(df['Close'] / df['Close'].shift()) 
        df['Vola'] = df['Return'].rolling(NP).std()
        df['Min'] = df['Close'].rolling(NP).min()
        df['Max'] = df['Close'].rolling(NP).max()
        df['Range'] = abs(df['Close'] - df['Open'])
        df['Z_score'] = (df['Range'] - df['Range'].rolling(NP1).mean()) / df['Range'].rolling(NP1).std()  
        
        #Create target column: if tomorrow's price greather than today's price - 1, otherwise 0
        #Or if the price in k days is higher than today's price
        K = 1
        df['Target'] = np.where(df['Close'].shift(-K) > df['Close'], 1, 0)
                
        #df.to_excel(MY_PATH + '_' + ticker + '_' + interval +'_dataset.xlsx', index=False)
        
        #Remove the first n days of data
        n = 29
        df = df[n:]
        
        #Split the data into a feature and indepenent dataset (X), and the the target or dependent dataset (Y)
        features = [
            'Close', 
            'MACD', 
            'RSI', 
            'SMA', 
            'EMA',
            'Williams_R',
            'Vola',
            'Min',
            'Max',
            'Z_score'
        ]
        
        xf = features[5] #Williams_R
        yf = features[9] #Z_score (C-O) rolling(13)
                       
        #x = df['SMA']
        x = df[xf]
        y = df[yf]
        
        
        #fig = plt.figure(figsize=(15,8))
        #plt.xlabel(xf)
        #plt.ylabel(yf)
        #plt.scatter(x, y)
        
        data = df[[xf, yf]]
        data = df[[xf, yf]]
        
        #ax = Axes3D(fig)
        #ax.scatter(x, y, z)
        #ax.scatter(x, y, z)
        
        eps=0.5
        min_samples = 8
        
        
        
        model = DBSCAN(eps = eps, min_samples = min_samples).fit(data)
        df['Labels'] = model.labels_
        
        #df_new = df.loc[df['Labels'] == 0]
        
        
        # visualize outputs
        #colors = model.labels_
        #plt.scatter(data[xf], data[yf], c = colors)
        #plt.title(f'{ticker}  eps={eps} min_samples={min_samples}')
        
        LabelsChart(df, ticker)
        
        data_new = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        data_new.set_index('Date', inplace=True)
        data_new = data_new[-150:]
        #mpf.plot(data_new, type='candle', volume=True, figratio=(12, 4))
        #mpf.plot(data_new, volume=True, type='candle', figratio=(14, 4), style='charles', vlines=dict(vlines=['2020-08-25'], linewidths=(0.8), colors='r'))
                
        #df.to_excel('111.xlsx', index=False)
        
        
        print(ticker, ' ====== ')
        
    except:
        print('Cannot calculate for ticker: ', ticker)


