# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:40:04 2021

@author: Trader

"""
class Ticker:
    def __init__(self, ticker, interval, start, end):
        self.ticker = ticker
        self.interval = interval
        self.start = start
        self.end = end
        self.data = web.get_data_yahoo(self.ticker, 
                                       start=self.start, 
                                       end=self.end, 
                                       interval=self.interval).reset_index()
                
#    def load_data(self):
#        df = web.get_data_yahoo(self.ticker, start=self.start, end=self.end, interval=self.interval).reset_index()
#        return df

# Function returns Heiken Ashi dataframe
def HA(df):
    df['HA_Close']=(df['Open']+ df['High']+ df['Low']+df['Close'])/4

    idx = df.index.name
    df.reset_index(inplace=True)

    for i in range(0, len(df)):
        if i == 0:
            df.at[i, 'HA_Open']  = (df.loc[i, 'Open'] + df.loc[i, 'Close']) / 2
        else:
            df.at[i, 'HA_Open'] = (df.loc[i - 1, 'HA_Open'] + df.loc[i - 1, 'HA_Close']) / 2
    
    if idx:
        df.set_index(idx, inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','High']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','Low']].min(axis=1)
    
    return df