# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:57:40 2021

@author: Шевченко Роман
"""

#Version 1.0 (base) 
# Research for improvin classification results using new methods of labelling
# and feature engeneering

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
interval = 'w'

df = pd.DataFrame()

df['Open'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Open']
df['High'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['High']
df['Low'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Low']
df['Close'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Close']
df['Volume'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Volume']

df = df.reset_index()

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

K = 1 #NUmber of periods for target variable

#Make some new features

df['Vol'] = df['Volume'] / df['Volume'].rolling(20).mean()
df['Ret'] = df['Open'].shift(-2) - df['Open'].shift(-1)

df['Target'] = np.where(df['Close'].shift(-K) > df['Close'], 1, 0)

df = df[30:] #Removing the first 30 rows with NaN 

df_train = df[df['Date'] < datetime(2018,1,1)]
df_test = df[df['Date'] >= datetime(2018,1,1)]

feat_1 = 'Vol'
feat_2 = 'RSI'
feats = [feat_1, feat_2]

X = df[feats]
n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters).fit(X)
y_kmeans = kmeans.predict(X)
df['Cluster'] = y_kmeans

PLOT = False
# Plot the clusters
if PLOT:
    plt.figure(figsize=(15,8))
    centers = kmeans.cluster_centers_
    plt.scatter(df[feat_1],df[feat_2],c=y_kmeans)
    plt.scatter(centers[:,0],centers[:,1],c='red',s=100,marker='x')
    
#Split the data into training and test sets
#features = df.columns.tolist()
features = [
    #'Date', 
    #'Open', 
    #'High', 
    #'Low', 
    #'Close', 
    #'Volume', 
    'MACD', 
    #'Signal_Line', 
    #'Up', 
    #'Down', 
    'RSI', 
    'OBV', 
    #'BB_up', 
    #'BB_mid', 
    #'BB_dn', 
    'ROC', 
    'Williams', 
    #'%K', 
    #'%D', 
    #'SMA', 
    #'EMA', 
    #'Vol', 
    #'Ret', 
    #'Target', 
    #'Cluster'
    ]

X_train = df_train.loc[:, features]
y_train = df_train.loc[:, "Target"]
X_test = df_test.loc[:, features]
y_test = df_test.loc[:, "Target"]

    
#import autosklearn.classification
#cls = autosklearn.classification.AutoSklearnClassifier()
#cls.fit(X_train, y_train)
#predictions = cls.predict(X_test)

#import autokeras as ak

#clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
#clf.fit(X_train, y_train, epochs=10)
#predicted_y = clf.predict(X_test)
#print(clf.evaluate(X_test. y_test))

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#clf = RandomForestClassifier().fit(X_train, y_train)
clf = XGBClassifier(use_label_encoder=False)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


score = accuracy_score(y_test, y_pred)

print(score)