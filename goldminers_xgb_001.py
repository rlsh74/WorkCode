# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 08:52:42 2021

@author: Шевченко Роман
"""

import pandas as pd
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from scipy import stats

from xgboost import XGBRegressor

tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]
#tickers = ['TXG.TO']
benchmark = 'GC=F'
industry_mark = 'GDX'
ATR_PERIOD = 12
interval = 'd'
start_date = datetime(2019, 10, 20)
end_date = datetime(2020, 12, 31)

UPK = 2.2 #coefficient for upper band
DNK = 2.2 #coefficient fo lower band

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

        df['Ticker'] = ticker
 
        df.reset_index(inplace=True)

        df = df.rename({'index':'Date'}, axis='columns')

        #===== Calculate ATR for stock

        for i in range(1, len(df)):
            d1 = df.loc[i, 'high1'] - df.loc[i, 'low1']
            d2 = abs(df.loc[i, 'high1'] - df.loc[i-1, 'close1'])
            d3 = abs(df.loc[i, 'low1'] - df.loc[i-1, 'close1'])
            df.loc[i, 'TR1'] = max(d1, d2, d3).round(2)

        df['ATR1'] = df['TR1'].rolling(ATR_PERIOD).mean()
        df['ATR_P1'] = df['ATR1'] / df['close1'] 
        df['Diff1'] = df['close1'].pct_change()

        df['Diff2'] = df['close2'].pct_change()
        df['Diff3'] = df['close3'].pct_change()

        df['OBV1'] = np.where(df['close1'] > df['close1'].shift(1), df['volume1'], np.where(df['close1'] < df['close1'].shift(1), -df['volume1'], 0)).cumsum()
        df['OBV2'] = np.where(df['close2'] > df['close2'].shift(1), df['volume2'], np.where(df['close2'] < df['close2'].shift(1), -df['volume2'], 0)).cumsum()
        df['OBV3'] = np.where(df['close3'] > df['close3'].shift(1), df['volume3'], np.where(df['close3'] < df['close3'].shift(1), -df['volume3'], 0)).cumsum()

        df['Day_of_week'] = df['Date'].dt.dayofweek

        #===== Calculate dynamic Beta

        WINDOW_SIZE = 5

        for i in range(WINDOW_SIZE, len(df)):
 
            df1 = df.loc[i-WINDOW_SIZE:i, ['Diff1']]
            df2 = df.loc[i-WINDOW_SIZE:i, ['Diff2']]

            X = df2['Diff2'] #Gold
            y = df1['Diff1'] #Stock

            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

            df.at[i, 'Beta_gold'] = slope

        for i in range(WINDOW_SIZE, len(df)):
 
            df1 = df.loc[i-WINDOW_SIZE:i, ['Diff1']]
            df2 = df.loc[i-WINDOW_SIZE:i, ['Diff3']]

            X = df2['Diff3'] #Gold
            y = df1['Diff1'] #Stock

            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

            df.at[i, 'Beta_sector'] = slope
            
        #df.to_excel(ticker + '_dataset_.xlsx', index=False)

        #df = pd.read_excel(ticker+'_dataset_.xlsx')

        data = df.copy()

        features = [
          #'Date',
          #'open1',    #it's a target
          'high1',
          'low1',
          'close1',
          'volume1',
          'open2',
          'high2',
          'low2',
          'close2',
          'volume2',
          'open3',
          'high3',
          'low3',
          'close3',
          'volume3',
          #'Ticker',
          'TR1',
          'ATR1',
          'ATR_P1',
          'Diff1',
          'Diff2',
          'Diff3',
          'OBV1',
          'OBV2',
          'OBV3',
          'Day_of_week',
          'Beta_gold',
          'Beta_sector'
        ]

        TEST_WINDOW = 30
        TEST_LENGTH = 1
        #n_train = TEST_WINDOW

        target = ['open1']
        new_data = pd.DataFrame()

        for i in range(TEST_WINDOW, len(data)):
            
            X_train = data.loc[i-TEST_WINDOW:i, features][:-TEST_LENGTH]
            X_test =  data.loc[i-TEST_WINDOW:i, features][-TEST_LENGTH:]
            
            y_train = data.loc[i-TEST_WINDOW:i, target][:-TEST_LENGTH]
            y_test =  data.loc[i-TEST_WINDOW:i, target][-TEST_LENGTH:]

            model = XGBRegressor(objective='reg:squarederror')
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            new_data.loc[i, ['Date']] = data.loc[i, ['Date']].values[0]
            new_data.loc[i, ['Price']] = y_test.values[0][0]
            new_data.loc[i, ['Pred']] = round(y_pred[0], 2)

            PERIOD_MEAN = 21
            new_data['Main'] = new_data['Pred'].rolling(PERIOD_MEAN).mean()
            new_data['St'] = new_data['Pred'].rolling(12).std()

            new_data['Up'] = new_data['Main']+UPK*new_data['St']
            new_data['Dn'] = new_data['Main']-DNK*new_data['St']

            new_data['B_cond'] = np.where(new_data['Price']>new_data['Dn'], 1, 0)
            new_data['S_cond'] = np.where(new_data['Price']<new_data['Up'], 1, 0)

            plot_data = new_data.copy()[PERIOD_MEAN:]
            plot_data = plot_data.reset_index()

            #plot_data.to_excel(ticker+ '_pred.xlsx', index=False)
            
        df = plot_data

        df['Ticker'] = ticker


        sigPriceBuy = []
        sigPriceSell = []

        df['BS'] = 0
        for i in range(3, len(df)):
            if (df['B_cond'][i] == 1):
                if (df['B_cond'][i-1] == 1):
                    if (df['B_cond'][i-2] == 0):
                        if (df['B_cond'][i-3] == 0):
                            df.at[i, 'BS'] = 1
                            #df['BS'][i] = 1
        df['SS'] = 0
        for i in range(3, len(df)):
            if (df['S_cond'][i] == 1):
                if (df['S_cond'][i-1] == 1):
                    if (df['S_cond'][i-2] == 0):
                        if (df['S_cond'][i-3] == 0):
                            df.at[i, 'SS'] = 1
                            #df['SS'][i] = 1

        df['BS'] = np.where(df['BS']==1, df['Price'], np.nan)
        df['SS'] = np.where(df['SS']==1, df['Price'], np.nan)   
                           
#Visualize the data and the stategy to buy and sell the stock
#        df = df.set_index(df['Date'])

        #plt.figure(figsize=(15, 8))
        #plt.plot(df['Price'], label=ticker, alpha=0.45)
        #plt.plot(df['Up'], label='Upper', alpha=0.45)
        #plt.plot(df['Dn'], label='Lower', alpha=0.45)
        #plt.scatter(df.index, df['BS'], label='Buy', marker='^', color='green')
        #plt.scatter(df.index, df['SS'], label='Sell', marker='v', color='red')
        #plt.legend(loc='upper left')
        #plt.title(ticker + ' (' + str(UPK) +'/' +str(DNK) + ')')
            
        print(ticker)
        
        df.to_excel(ticker+'_pred.xlsx', index=False)
            
    except:
        print('Cannot calculate for ticker: ', ticker)
        

for ticker in tickers:
    try:
        df = pd.read_excel(ticker+ '_pred.xlsx')

        df['BS'] = df['BS'].fillna(0)
        df['SS'] = df['SS'].fillna(0)
        
            
        pos = 0
        num = 0
        returns = []
        
        for i in range(len(df)):
            if(df['BS'][i] != 0) & (pos==0):
                b_price = df['BS'][i]
                pos = 1
            if(df['SS'][i] != 0) & (pos==1):
                s_price = df['SS'][i]
                pos = 0
                returns.append((s_price/b_price-1)*100)
            if(num ==df['Price'].count()-1 and pos == 1):
                s_price = df['Price'][-1:].values[0]
                returns.append((s_price/b_price-1)*100)
            num+=1
        
                
        total = sum(returns)
        print(ticker, ' ====== ', round(total,2))
    except:
        print('Cannot calculate for ticker: ', ticker)
