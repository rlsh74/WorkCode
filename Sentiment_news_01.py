# -*- coding: utf-8 -*-
"""
Created on Sat Apr  3 11:29:52 2021

@author: Шевченко Роман
"""
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = 'https://finviz.com/quote.ashx?t='

tickers = ['AMZN', 'GOOG', 'FB']
tickers = ['AEM', 'KGC', 'WPM']

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    
    html = BeautifulSoup(response, features='lxml')
    
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table
    
# =============================================================================
# amzn_data = news_tables['AMZN']
# amzn_rows = amzn_data.findAll('tr')
# 
# for index, row in enumerate(amzn_rows):
#     title = row.a.text
#     timestamp = row.td.text
#     print(timestamp + ' ' + title)
# =============================================================================

parsed_data = []

for ticker, news_table in news_tables.items():
    for row in news_table.findAll('tr'):
        title = row.a.text
        date_data = row.td.text.split(' ')
        
        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns = ['ticker', 'date', 'time', 'title'])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)

df['date'] = pd.to_datetime(df['date']).dt.date

mean_df = df.groupby(['ticker', 'date']).mean()
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis='columns').transpose()  #cross section table
#mean_df.plot(kind='bar')

mean_df[50:].plot(kind='bar', figsize=(15,8))


        
    
