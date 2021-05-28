
import numpy as np
import pandas as pd
import statsmodels
from statsmodels.tsa.stattools import coint
import seaborn
import matplotlib.pyplot as plt
from pandas_datareader import data as web
from datetime import datetime, timedelta

from itertools import combinations
from statsmodels.tsa.stattools import coint

def zscore(series):
    return ((series - series.mean()) / np.std(series))

tickers = [
           'PLZL.ME', 'DPM.TO', 'RMS.AX', 'CG.TO', '2899.HK', 'BTG', 'EGO', 'GOR.AX', 'NCM.AX', 'WDO.TO', 'POLY.L', 
           'EQX.TO', 'CEY.L', 'PVG', 'SBM.AX', 'BVN', 'PRU.AX', 'NST.AX', 'EDV.TO', 'LUN.TO', 'RRL.AX', 'POG.L', 'KGC', 'GFI', 'TGZ.TO', 'CDE', 
           'TXG.TO', 'DRD', 'NG', 'AU', 'GOLD', 'EVN.AX', 'NEM', 'SAR.AX', 'AUY', 'AEM', 'WPM', 'IAG', 'SSRM', 'OGC.TO', 'AGI', 'NGD', 'HMY', 'SAND', 'SBSW', 'FNV', 'RGLD', 'OR', 'SA', 'ANTM.JK'
           ]
#benchmark = 'DX=F' #(Dollar index)
benchmark = 'GC=F' #(Index SP500)
industry_mark = 'GDX'
ATR_PERIOD = 12
start_date = datetime(2020, 1, 1)
#end_date = datetime(2021, 5, 1)
end_date = datetime.today()

interval = 'd'

ticker1 = ['NEM']
ticker2 = ['GOLD']

tickers = ['NEM', 'GOLD', 'AEM', 'AU']

df = web.get_data_yahoo(tickers, start=start_date, end=end_date, interval=interval)['Close']

df['Diff1'] = df[ticker1].pct_change()
df['Diff2'] = df[ticker2].pct_change()
df = df.dropna()
df = df.reset_index()

X = df['Diff1']
Y = df['Diff2']

df['Zscore'] = zscore(X-Y)
df['Zscore'].plot()

#Get the pairs of tickers
comb_list = list(combinations(tickers, 2))

try: 
    for i in range(0, len(comb_list)):
        ticker1 = comb_list[i][0]
        ticker2 = comb_list[i][1]
    
        df = web.get_data_yahoo(tickers, start=start_date, end=end_date, interval=interval)['Close']
    
        df['Diff1'] = df[ticker1].pct_change()
        df['Diff2'] = df[ticker2].pct_change()
        df = df.dropna()
        df = df.reset_index()

        X = df['Diff1']
        Y = df['Diff2']
        
        score, pvalue, _ = coint(X, Y)
        print(f'{ticker1} and {ticker2}: score: {round(score, 4)}  p_value: {round(pvalue, 4)}')
        
        
        
except:
    print(f'Cannot calculate cointegration for pair: {ticker1} and {ticker2}')
    
    