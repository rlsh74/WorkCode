# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:36:02 2021

@author: Trader
"""
#The cod had been changed on 27.01.2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from pandas_datareader import data as web
from datetime import datetime, timedelta

import streamlit as st
import altair as alt

from functions import *


#start_date = datetime(2020, 6, 14)
#end_date = datetime(2021, 1, 14)

#ticker = 'NEM'
#interval = 'w'
int_list = ['d', 'w', 'm']


ticker = st.sidebar.text_input('Ticker', '^GSPC')
interval = st.sidebar.selectbox('Interval', int_list, 0)
start_date = st.sidebar.date_input('Start date', datetime(2020, 1, 1))
end_date = st.sidebar.date_input('End date', datetime.today())

data = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)
data = data.reset_index()

HA(data)

# Unit fo Heiken Ashi Altair chart
#=================================
#open_close_color = alt.condition("datum.open <= datum.close",
#                                 alt.value('red'),
#                                 alt.value('green'))

open_close_color = alt.condition("datum.HA_Open <= datum.HA_Close",
                                 alt.value('green'),
                                 alt.value('red'))

base = alt.Chart(data).encode(
    alt.X('Date:T',
          axis=alt.Axis(
              format='%m/%d',
              labelAngle=-45,
              title='Date'
          )
    ),
    color=open_close_color
)

rule = base.mark_rule().encode(
    alt.Y(
        'HA_Low:Q',
        title='Price',
        scale=alt.Scale(zero=False),
    ),
    alt.Y2('HA_High:Q')
)

bar = base.mark_bar().encode(
    alt.Y('HA_Open:Q'),
    alt.Y2('HA_Close:Q')
)


#=================================
st.altair_chart(rule + bar)


#print('Everything is OK')


