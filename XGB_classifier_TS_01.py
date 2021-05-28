# -*- coding: utf-8 -*-
"""
Created on Fri Mar 5 22:07:54 2021

@author: Шевченко Роман
"""

# Trading system based on XGBClassifier
# Version 1.0


import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import xgboost
from xgboost import XGBClassifier

#tickers = ['NEM', 'GOLD']
tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]

#tickers = ['GC=F', '^TNX', '^GSPC']
#tickers = ['^GSPC']
#tickers =['^IXIC']
#tickers = ['HON']

tickers = [
'HON', 'MMM', 'GE', 'ITW', 'ABB','EMR', 'ETN', 'ROP', 'TT', 'CMI', 'PH', 'ROK', 'AME',
'OTIS', 'GNRC', 'IR', 'XYL', 'DOV', 'IEX', 'HWM', 'GGG', 'LII', 'NDSN', 'BLDP', 'AOS',
'PNR', 'DCI', 'MIDD', 'ITT', 'GTLS', 'RBC', 'FLS', 'RXN', 'CR', 'CW', 'CFX', 'GTES', 'WTS', 'KRNT',
'JBT', 'PSN', 'AIMC', 'FELE', 'HI', 'ATKR', 'TPIC', 'B', 'SPXC', 'FLOW', 'WBT', 'MWA', 'CSWI', 'HLIO',
'KAI', 'NPO', 'HSC', 'OFLX', 'TRS', 'TNC', 'RAVN', 'EPAC', 'SXI', 'XONE', 'GRC', 'AMSC', 'CYD','CIR',
'LDL','THR','LXFR'
]


tickers = ['GC=F', 'SI=F', '^TNX', 'CL=F', 'HG=F', 'NG=F', '6E=F', '6B=F', '6J=F', '6S=F', '6A=F', 'BTC-USD', 'ETH-USD', 'XRP-USD']

#comp_list = pd.read_excel('Real_Estate_list.xlsx')
#tickers = comp_list['Ticker'].tolist()

#tickers = ['HON']



end_date = datetime.today()
start_date = end_date - timedelta(days=1150)
#start_date = datetime(2018, 1, 10) #for daily n=60
#start_date = datetime(2011, 1, 1) #for weekly n=60
#end_date = datetime(2021, 2, 14)


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


def Signal_chart(ticker, data):
    #plot the chart witn signals
        plt.figure(figsize=(15, 8))
        plt.plot(data['Close'], label=ticker, alpha=0.45)
        plt.scatter(data.index, data['BS'], label='Buy', marker='^', color='green')
        plt.scatter(data.index, data['SS'], label='Sell', marker='v', color='red')
        plt.legend(loc='upper left')
        plt.title(ticker)

signals = pd.DataFrame() #Here we collect all signals

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
        
        
        #Create target column: if tomorrow's price greather than today's price - 1, otherwise 0
        #Or if the price in k days is higher than today's price
        K = 2
        #K = 15
        df['Target'] = np.where(df['Close'].shift(-K) > df['Close'], 1, 0)
        
        df.to_excel('_' + ticker + '_' + interval +'_dataset.xlsx', index=False)
        
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
            'Williams_R'
        ]
        
               
        train = df[:len(df) - N_PERIODS]
        test = df[-N_PERIODS:]
        X_train = train[features].values
        X_test = test[features].values
        y_train = train['Target'].values
        y_test = test['Target'].values
        
        #plt.figure(figsize=(10, 7))
        #(df['Close'].pct_change() + 1).cumprod().plot()
    
        # Set the title and axis lables and plot grid
        #plt.title(ticker + ' Returns')
        #plt.ylabel('Cumulative Returns')
        #plt.grid()
        #plt.show()
        
        model = XGBClassifier(max_depth=3, n_estimators=500, learning_rate=0.05, objective='reg:squarederror', use_label_encoder=False)

        #Cross validation

        kfold = KFold (n_splits=5, random_state=7, shuffle=True)

        results = cross_val_score(model,X_train, y_train, cv=kfold)
        mresults = results.mean()*100
        stdresults = results.std()*100

        print(f"Accuracy: {mresults} ({stdresults})")
        
        model.fit(X_train, y_train)
        
    
        print(model.feature_importances_)    
        
        
        # Create an empty dataframe to store the strategy returns of individual stocks\n",
        #portfolio = pd.DataFrame(columns=stock_list)
    
        # For each stock in the stock list, plot the strategy returns and buy and hold returns\n",
        
    # Store the predictor variables in X
        #X = df[predictor_list]
    
        #Define the train and test dataset
        #train_length = int(len(X)*0.80)
    
        # Predict the signal and store in predicted signal column
        print(1)
        s = pd.Series(model.predict(X_test))
        #df['Pred'] = s
        print(2)
    
        # Calculate the strategy returns
        res = df[-N_PERIODS:].copy()
        res = res.reset_index()
        res['Diff'] = res['Close'].pct_change()
        print(3)
        res['Pred'] = s
        print(4)
        res['Strategy_returns'] = res['Close'].pct_change(-1) * res['Pred']
    
        # Calculate the strategy returns
        #df['strategy_returns'] = df.return_next_day * df.predicted_signal
    
        # Add the strategy returns to the portfolio dataframe
        #portfolio[stock_name] = df.strategy_returns[train_length:]
    
    
        #Set the figure size
        plt.figure(figsize=(10, 7))
    
        # Calculate the cumulative strategy returns and plot
        (res['Strategy_returns']+1).cumprod().plot()
    
        # Calculate the cumulative buy and hold strategy returns
        (res['Diff']+1).cumprod().plot()    
    
        # Set the title, label and grid
        plt.title(ticker)
        plt.ylabel('Cumulative Returns')
        plt.legend(labels=['Strategy Returns', 'Buy and Hold Returns'])
        plt.grid()
        plt.show()
        #print(ticker, ' ====== ', round(score, 4))
        print(ticker)
        
    except:
        print('Cannot calculate for ticker: ', ticker)


