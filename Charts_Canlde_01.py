# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 18:57:40 2021

@author: Шевченко Роман
"""

#Version 1.0 (base) 
# Candlesticks charts to find patterns for new method of dynamic labelling

import matplotlib.pyplot as plt
from functions import *
from datetime import datetime
import pandas as pd
import mplfinance as mpf

from pandas_datareader import data as web
#from tecan import *

def Find_pattern():
    pass


#tickers = ['AEM', 'AU', 'CLW', 'ENIA', 'FLWS', 'HOLI', 'JD', 'JOBS', 'KL', 'MESA', 'NAT', 'PRG', 'SMP', 'SPH', 'ZEPP', 'ZTO']

# candiddaet for 11/05/2021
tickers = [
'CRM', 
'EBS',
'ORA',
'TYL',
]

tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]

tickers = ['DRD', 'EQC', 'BRMK']

tickers = ['AEM', 'AUY', 'AYI', 'EGO', 'ENIA', 'HOLI', 'IAG', 'JOBS', 'KGC', 'PRG']

#tickers = ['CL=F', 'GC=F', 'BTC-USD', 'DOGE-USD', 'ENIA', 'TQQQ']



end_date = datetime.now()
#start_date = datetime(end_date.year-YEARS, end_date.month, end_date.day)
start_date = datetime(2010, 1, 1)

interval ='d'
#interval = 'w'

for ticker in tickers:
    try:
        df = web.get_data_yahoo(ticker, start=start_date, end=end_date, interval=interval)
        #df = df.reset_index()

        HA(df)
        df_ha = df[['HA_High', 'HA_Low', 'HA_Open', 'HA_Close']]
        df_ha = df_ha.rename(columns={'HA_High':'High', 'HA_Low':'Low', 'HA_Open':'Open', 'HA_Close':'Close',})

        #s  = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size':10})
        # we have to add style-s in setup dictionary

        setup=dict(type='candle',figratio=(15,8), show_nontrading=False, volume=False, ylabel='', tight_layout=False)

        mpf.plot(df_ha['2021-03':'2021-06'],
                 **setup,
                 mav=(3,8),
                 title=f'{ticker}'
                 )
    except:
        print(f'Cannot calculate for {ticker}!')


