# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 12:25:45 2021

@author: Шевченко Роман
"""
import pandas as pd
from pandas_datareader import data as web
import plotly.graph_objects as go

from plotly.offline import plot

ticker = 'NEM'
start = '2020-01-01'
end = '2021-04-21'

df = web.get_data_yahoo(ticker, start, end)
df = df.reset_index()
#df['Date'] = pd.to_datetime(df['Date'])

fig = go.Figure()

#fig.update_layout(xaxis_rangeslider_visible=False)

fig.update_xaxes(
        rangeslider_visible=False,
        rangebreaks=[
            # NOTE: Below values are bound (not single values), ie. hide x to y
            dict(bounds=["sat", "mon"]),  # hide weekends, eg. hide sat to before mon
            dict(bounds=[16, 9.5], pattern="hour"),  # hide hours outside of 9.30am-4pm
            # dict(values=["2020-12-25", "2021-01-01"])  # hide holidays (Christmas and New Year's, etc)
        ]
    )
fig.update_layout(
        title='Stock Analysis',
        yaxis_title=f'{ticker} Stock'
    )

fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])

plot(fig)