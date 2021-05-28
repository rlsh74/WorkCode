# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:14:24 2021

@author: Шевченко Роман
"""

import pandas as pd
from pandas_datareader import data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]
#tickers = ['WDO.TO']
benchmark = 'GC=F'
industry_mark = 'GDX'
ATR_PERIOD = 12
interval = 'd'
start_date = datetime(2019, 1, 1)
end_date = datetime(2019, 12, 31)

for ticker in tickers:
    try:
        df = pd.read_excel(ticker+ '_pred.xlsx')
        df = df[(df['Date']>=start_date) & (df['Date']<=end_date)]
        df.drop(['index'], axis = 1, inplace=True)
        

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

