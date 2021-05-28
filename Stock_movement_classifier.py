# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 09:39:54 2021

@author: Шевченко Роман
"""
# Use stock indicators with machine learning to try to predict yhe direction of a stock's price

import pandas as pd
import numpy as np
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV

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


tickers = ['GC=F', 'SI=F','HG=F', 'PL=F', 'PA=F', '^TNX', 'CL=F', 'NG=F', '6E=F', '6B=F', '6J=F', '6S=F', '6A=F', 'BTC-USD', 'ETH-USD', 'XRP-USD']

#tickers = ['AEM', 'AUY', 'EGO', 'GGG', 'GTES', 'IAG', 'KGC', 'WPM']


#comp_list = pd.read_excel('Real_Estate_list.xlsx')
#tickers = comp_list['Ticker'].tolist()

#tickers = ['AEM', 'AUY', 'EGO', 'GGG', 'GTES', 'KGC', 'KRNT', 'TPIC']

#tickers = ['HON']

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

def set_seeds(seed=100):
    random.seed(seed)
    np.random.seed(seed)

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

def Signal_maker1(data, column):
    #Main logic:
    #two 1 in a row - buy
    #two 0 in a row - sell
    
    data['BS'] = 0
    data['SS'] = 0
    
    k = 0
    
    for i in range(1, len(data)):
        
        if (data[column][i] == 1):
            if (data[column][i-1] == 1):
                k += 1
                data.at[i, 'BS'] = k
            
                
    for i in range(1, len(data)):
        if (data[column][i] == 0):
            if (data[column][i-1] == 0):
                data.at[i, 'SS'] = 1

    #data['BS'] = np.where(data['BS']==1, data['Close'], np.nan)
    #data['SS'] = np.where(data['SS']==1, data['Close'], np.nan)
    
    return data

def Signal_maker(data, column):
    #Main logic:
    #two 1 in a row - buy
    #two 0 in a row - sell
    
    data['BS'] = 0
    data['SS'] = 0
    
    
    for i in range(2, len(data)):
        if (data[column][i-2] == 0) & (data[column][i-1] == 1) & (data[column][i] == 1):
            data.at[i, 'BS'] = 1
               
    for i in range(2, len(data)):
        if (data[column][i-2] == 1) & (data[column][i-1] == 0) & (data[column][i] == 0):
            data.at[i, 'SS'] = 1


    data['BS'] = np.where(data['BS']==1, data['Close'], np.nan)
    data['SS'] = np.where(data['SS']==1, data['Close'], np.nan)
    
    return data

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
        
        NP = 13
        df['Return'] = np.log(df['Close'] / df['Close'].shift()) 
        df['Vola'] = df['Return'].rolling(NP).std()
        df['Min'] = df['Close'].rolling(NP).min()
        df['Max'] = df['Close'].rolling(NP).max()
        df['Range'] = abs(df['Close'] - df['Open'])
        df['Z_score'] = (df['Range'] - df['Range'].rolling(NP).mean()) / df['Range'].rolling(NP).std()  
        
        #Create target column: if tomorrow's price greather than today's price - 1, otherwise 0
        #Or if the price in k days is higher than today's price
        K = 1
        df['Target'] = np.where(df['Close'].shift(-K) > df['Close'], 1, 0)
                
        df.to_excel(MY_PATH + '_' + ticker + '_' + interval +'_dataset.xlsx', index=False)
        
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
        
        set_seeds()
        dbf = df[-N_PERIODS:]
        X_real_test = dbf[features].values
        
        X = df[features].values
        y = df['Target'].values
        
        #Split the data into training and testing dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        #Create and train the model
        #model = DecisionTreeClassifier().fit(X_train, y_train)
                
        model = XGBClassifier(use_label_encoder=False, objective='reg:squarederror').fit(X_train, y_train)
        #print(model.score(X_train, y_train))   
        
        #Check how well the model did on the testing dataset
        score = model.score(X_test, y_test)
        
        parameters = {
            'learning_rate':[0.005, 0.01, 0.02, 0.1, 0.2],
            'n_estimators':[200, 300, 400, 500],
            #'objective':['binary:logistic', 'binary:hinge'],
            'subsample':[1.0, 1.2, 1.3],
            #'min_samples_split':[10, 50]            
            }
        
        #clf = GridSearchCV(model, parameters, scoring='accuracy', verbose=1)
        #clf.fit(X_train, y_train)
        #bp = clf.best_params_
        #bs = clf.best_score_
        
        #Show the predictions
        #pred = model.predict(X_test)
        pred = model.predict(X_real_test)
        #print(pred)
        pred_list = pred.tolist()
        #print(pred_list)
        dbf.reset_index(inplace=True, drop=True)
        df_result = dbf.copy()
        
        
        df_result['Pred'] = pred_list
        df_result['Ticker'] = ticker
        
        
                
        #Get the model metrics
        #print(classification_report(y_test, pred))
        #print(model.feature_importances_)
        
        # Here must be the backtester
        
        Signal_maker(df_result, 'Pred')
        
        res = df_result[['Date', 'Ticker', 'Close', 'BS', 'SS']]
          
        res.to_excel(MY_PATH + 'A_' + ticker+ '_' +str(K) + '_' + interval + '_signal.xlsx', index=False)
        
        #plot the chart witn signals
        #plt.figure(figsize=(15, 8))
        #plt.plot(res['Close'], label=ticker, alpha=0.45)
        #plt.scatter(res.index, res['BS'], label='Buy', marker='^', color='green')
        #plt.scatter(res.index, res['SS'], label='Sell', marker='v', color='red')
        #plt.legend(loc='upper left')
        #plt.title(ticker)
        
        Signal_chart(ticker, res)
        
        signals = pd.concat([signals, res], ignore_index=True)
            
        #print(ticker, ' ====== ')
        print(ticker, ' ====== ', round(score, 4))
        
    except:
        print('Cannot calculate for ticker: ', ticker)


signals = signals.loc[~(np.isnan(signals['BS'])) | ~(np.isnan(signals['SS']))]
signals = signals.sort_values(by=['Date'])
signals.to_excel(MY_PATH + '_All_signals.xlsx', index=False)

for ticker in tickers:
    try:
        df = pd.read_excel(MY_PATH + 'A_' + ticker+ '_' +str(K) + '_' + interval + '_signal.xlsx')
        #df = df[(df['Date']>=start_date) & (df['Date']<=end_date)]
        #df.drop(['index'], axis = 1, inplace=True)
        

        df['BS'] = df['BS'].fillna(0)
        df['SS'] = df['SS'].fillna(0)
        
            
        pos = 0
        num = 0
        returns = []
        
        for i in range(len(df)):
            if(df['BS'][i] != 0) & (pos==0):
                b_price = df['BS'][i]
                pos = 1
                #print('buy at ', b_price)
            if(df['SS'][i] != 0) & (pos==1):
                s_price = df['SS'][i]
                #print('  sell at ', s_price)
                pos = 0
                returns.append((s_price/b_price-1)*100)
            if(num ==df['Close'].count()-1 and pos == 1):
                s_price = df['Close'][-1:].values[0]
                returns.append((s_price/b_price-1)*100)
            num+=1
        
                
        total = sum(returns)
        
        #calculate the rerutn for period (buy & hold strategy)
        p1 = df['Close'][0]
        ll = len(df)
        p2 = df['Close'][ll-1]

        diff = (p2/p1-1)*100
        
        
        print(ticker, ' ====== ', round(total,2), ' ====== ', round(diff, 2))
        
    except:
        print('Cannot calculate for ticker: ', ticker)
