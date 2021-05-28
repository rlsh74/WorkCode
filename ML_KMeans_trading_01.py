# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:57:40 2021

@author: Шевченко Роман
"""

#Version 1.0 (base) 
#KMeans clustering subsets of features of stock price data

import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import numpy as np
#import talib
from sklearn.cluster import KMeans

from pandas_datareader import data as web
from tecan import *

ticker = 'NEM'

end_date = datetime.now()
#start_date = datetime(end_date.year-YEARS, end_date.month, end_date.day)
start_date = datetime(2010, 1, 1)

interval ='d'

df = pd.DataFrame()

df['Open'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Open']
df['High'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['High']
df['Low'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Low']
df['Close'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Close']
df['Volume'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Volume']

#Call the functions
MACD(df)
RSI(df)
OBV(df)
Bollinger(df)
Price_ROC(df)
Williams_R(df)
Stochastic(df)
df['SMA'] = SMA(df)
df['EMA'] = EMA(df)


#Make some new features

df['Vol'] = df['Volume'] / df['Volume'].rolling(20).mean()
df['Ret'] = df['Open'].shift(-2) - df['Open'].shift(-1)
df['Target'] = 0
df = df[30:] #Removing the first 30 rows with NaN 

df_train = df[df.index < datetime(2018,1,1)]
df_test = df[df.index >= datetime(2018,1,1)]

feat_1 = 'Vol'
feat_2 = 'RSI'
features = [feat_1, feat_2]

X = df_train[features]
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters).fit(X)
y_kmeans = kmeans.predict(X)
df_train['Tar'] = y_kmeans

# --------------------------
#  Plot Training
#
plt.figure(figsize=(15,8))
centers = kmeans.cluster_centers_
plt.scatter(df_train[feat_1],df_train[feat_2],c=y_kmeans)
plt.scatter(centers[:,0],centers[:,1],c='red',s=100,marker='x')

#-------------------------------
#  K Means - Testing
#
x = df_test[features]
y_kmeans = kmeans.predict(x)
df_test['Tar'] = y_kmeans

# --------------------------------
#  Plot Testing
#
plt.figure(figsize=(15,8))
plt.scatter(df_test[feat_1],df_test[feat_2],c=y_kmeans)
plt.scatter(centers[:,0],centers[:,1],c='red',s=100,marker='x')
plt.show()

# ------------------------------------------
#  Compare Training and Testing
#
print("Total Points Earned by Cluster Prediction")

#print("Cluster 1 Train: %.2f\tCluster 1 Test: %2.f" % (df_train['Ret'].loc[df_train['Tar'] == 0].sum(),df_test['Ret'].loc[df_test['Tar'] == 0].sum()))

#print("Cluster 2 Train: %.2f\tCluster 2 Test: %.2f" % (df_train['Ret'].loc[df_train['Tar'] == 1].sum(),df_test['Ret'].loc[df_test['Tar'] == 1].sum()))

#print("Cluster 3 Train: %.2f\tCluster 3 Test: %.2f" % (df_train['Ret'].loc[df_train['Tar'] == 2].sum(),df_test['Ret'].loc[df_test['Tar'] == 2].sum()))

for i in range(1, n_clusters+1):
    print(f"Cluster {i} Train: %.2f\tCluster {i} Test: %2.f" % (df_train['Ret'].loc[df_train['Tar'] == i-1].sum(),df_test['Ret'].loc[df_test['Tar'] == i-1].sum()))

# ------------------------------
#  Equity Curves
#
plt.figure(figsize=(15,8))
plt.plot(np.cumsum(df_test['Ret'].loc[df_test['Tar'] == 0]),label='Cluster Low')
plt.plot(np.cumsum(df_test['Ret'].loc[df_test['Tar'] == 1]),label='Cluster High')
plt.plot(np.cumsum(df_test['Ret'].loc[df_test['Tar'] == 2]),label='Cluster Med')
plt.legend()
plt.show()


