# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 10:02:57 2021

@author: Шевченко Роман
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Database():
    def __init__(self, ticker, days):
        self.ticker = ticker
        data = yf.download(ticker, start="2020-01-01", end="2021-04-21")
        self.df = pd.DataFrame(data)
        self.df['Date'] = pd.to_datetime(self.df.index)
        pd.set_option('display.max_columns', None)
        self.df = self.df[-days:]

    def quote(self):
        return self.df

    def MACD(self, per1, per2, per3):
        short = self.df['Adj Close'].ewm(span=per1, adjust=False).mean()
        long = self.df['Adj Close'].ewm(span=per2, adjust=False).mean()
        MACD = short - long
        signal = MACD.ewm(span=per3, adjust=False).mean()
        return [MACD, signal]

    def MACD_bar(self, df):
        MACD_bar = []
        for i in range(0, len(df)):
            value = df['MACD'][i] - df['signal'][i]
            MACD_bar.append(value)
        return MACD_bar

    def MACD_color(self, df):
        MACD_color = []
        for i in range(0, len(df)):
            if df['MACD_bar'][i] > df['MACD_bar'][i - 1]:
                MACD_color.append(1)
            else:
                MACD_color.append(-1)
        return MACD_color

def Data(ticker, days):
    db=Database(ticker, days)
    df=db.quote()
    df['MACD']=db.MACD(12, 26, 9)[0]
    df['signal']=db.MACD(12, 26, 9)[1]
    df['MACD_bar'] = db.MACD_bar(df)
    df['MACD_color'] = db.MACD_color(df)
    df['positive'] = df['MACD_color'] > 0
    return df

def buy_sell(df, risk):
    Buy=[]
    Sell=[]
    flag=False

    for i in range(0, len(df)):
        if df['MACD'][i] > df['signal'][i]:
            Sell.append(np.nan)
            if flag ==False:
                Buy.append(df['Adj Close'][i])
                flag=True
            else:
                Buy.append(np.nan)
      
        elif df['MACD'][i] < df['signal'][i]:
            Buy.append(np.nan)
            if flag ==True:
                Sell.append(df['Adj Close'][i])
                flag=False
            else:
                Sell.append(np.nan)

        elif flag == True and df['Adj Close'][i] < Buy[-1] * (1 - risk):
            Sell.append(df["Adj Close"][i])
            Buy.append(np.nan)
            flag = False

        elif flag == True and df['Adj Close'][i] < df['Adj Close'][i - 1] * (1 - risk):
            Sell.append(df["Adj Close"][i])
            Buy.append(np.nan)
            flag = False

        else:
            Buy.append(np.nan)
            Sell.append(np.nan)
    return (Buy, Sell)

def Plot(df, name):
    plt.rcParams.update({'font.size': 10})
    fig, ax1 = plt.subplots(figsize=(14,8))
    fig.suptitle(name, fontsize=10, backgroundcolor='blue', color='white')
    ax1 = plt.subplot2grid((14, 8), (0, 0), rowspan=8, colspan=14)
    ax2 = plt.subplot2grid((14, 8), (8, 0), rowspan=6, colspan=14)
    ax1.set_ylabel('Price in €')
    ax1.plot('Adj Close',data=df, label='Close Price', linewidth=0.5, color='blue')
    ax1.scatter(df.index, df['Buy_signal'], color='green', marker='^', alpha=1)
    ax1.scatter(df.index, df['Sell_signal'], color='red', marker='v', alpha=1)
    ax1.legend()
    ax1.grid()
    ax1.set_xlabel('Date', fontsize=8)
    fig.tight_layout()

    ax2.set_ylabel('MACD', fontsize=8)
    ax2.plot('MACD', data=df, label='MACD', linewidth=0.5, color='blue')
    ax2.plot('signal', data=df, label='signal', linewidth=0.5, color='red')
    ax2.bar('Date', 'MACD_bar', data=df, label='Volume', color=df.positive.map({True: 'g', False: 'r'}), width=1,alpha=0.8)
    ax2.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax2.legend()
    ax2.grid()
    plt.show()

def Run(name, ticker):
    days = 252
    df = Data(ticker, days)
    buy_sell_markers = buy_sell(df, 0.025) # df en riskpercentage
    df['Buy_signal'] = buy_sell_markers[0]
    df['Sell_signal'] = buy_sell_markers[1]
    Plot(df, name)

tickers = ['NEM', 'GOLD']

for ticker in tickers:
    Run(ticker, ticker)

