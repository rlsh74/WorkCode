# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 16:51:56 2021

@author: Шевченко Роман
"""

from scipy import signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas_datareader import data as web
from datetime import datetime, timedelta

def filter(values, percentage):
    previous = values[0] 
    mask = [True]
    for value in values[1:]: 
        relative_difference = np.abs(value - previous)/previous
        if relative_difference > percentage:
            previous = value
            mask.append(True)
        else:
            mask.append(False)
    return mask

def Zigzag(data, column, p=0.2, chart=False):
    np.random.seed(0)

    #date_rng = pd.date_range('2019-01-01', freq='s', periods=30)

    s = data[column]
    # Find peaks(max).
    peak_indexes = signal.argrelextrema(s.values, np.greater)
    peak_indexes = peak_indexes[0]
    # Find valleys(min).
    valley_indexes = signal.argrelextrema(s.values, np.less)
    valley_indexes = valley_indexes[0]
    # Merge peaks and valleys data points using pandas.
    df_peaks = pd.DataFrame({'date': s.index[peak_indexes], 'zigzag_y': s[peak_indexes]})
    df_valleys = pd.DataFrame({'date': s.index[valley_indexes], 'zigzag_y': s[valley_indexes]})
    df_peaks_valleys = pd.concat([df_peaks, df_valleys], axis=0, ignore_index=True, sort=True)
    # Sort peak and valley datapoints by date.
    df_peaks_valleys = df_peaks_valleys.sort_values(by=['date'])
    
    #p = 0.2 # 20% 
    filter_mask = filter(df_peaks_valleys.zigzag_y, p)
    filtered = df_peaks_valleys[filter_mask]
    
    if chart:

        # Instantiate axes.
        (fig, ax) = plt.subplots(figsize=(10,10))
        # Plot zigzag trendline.
        ax.plot(df_peaks_valleys['date'].values, df_peaks_valleys['zigzag_y'].values, 
                                                        color='red', label="Extrema")
        # Plot zigzag trendline.
        ax.plot(filtered['date'].values, filtered['zigzag_y'].values, 
                                                        color='blue', label="ZigZag")

        # Plot original line.
        ax.plot(s.index, s, linestyle='dashed', color='black', label="Org. line", linewidth=1)

        # Format time.
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        plt.gcf().autofmt_xdate()   # Beautify the x-labels
        plt.autoscale(tight=True)

        plt.legend(loc='best')
        plt.grid(True, linestyle='dashed')

    dbz = filtered.copy()
    dbz = dbz.reset_index()

    data['idx'] = data.index
    dbz['idx'] = dbz.date

    data = data.merge(dbz, on='idx', how='left')
    res = data[['Date', 'Close', 'zigzag_y']]
    
    return res

interval = 'd'
start_date = datetime(2018, 1, 1)
end_date = datetime(2021, 3, 8)

#comp_list = pd.read_excel('Real_Estate_list.xlsx')
#tickers = comp_list['Ticker'].tolist()

ticker = 'TQQQ'

df = pd.DataFrame()

df['Close'] = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)['Close']
        
df.reset_index(inplace=True)

new_data = Zigzag(df, 'Close', chart=(True))