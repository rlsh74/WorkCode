from yahoofinancials import YahooFinancials
from datetime import date, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# get stock ticker symbol from user
stock_symbol = 'AAPL'

# set date range for historical prices
end_time = date.today()
start_time = end_time - timedelta(days=365)

# reformat date range
end = end_time.strftime('%Y-%m-%d')
start = start_time.strftime('%Y-%m-%d')

# get prices
json_prices = YahooFinancials(stock_symbol
    ).get_historical_price_data(start, end, 'daily')

# transform json file to dataframe
prices = pd.DataFrame(json_prices[stock_symbol]
    ['prices'])[['formatted_date', 'close']]

# sort dates in descending order
prices.sort_index(ascending=False, inplace=True)

# calculate daily logarithmic return
prices['returns'] = (np.log(prices.close / 
    prices.close.shift(-1)))
     
# calculate daily standard deviation of returns
daily_std = np.std(prices.returns)
 
# annualize daily standard deviation
std = daily_std * 252 ** 0.5

 
# Plot histograms
fig, ax = plt.subplots(1, 1, figsize=(7, 5))
n, bins, patches = ax.hist(
    prices.returns.values,
    bins=50, alpha=0.65, color='blue',
    label='12-month')

ax.set_xlabel('log return of stock price')
ax.set_ylabel('frequency of log return')
ax.set_title('Historical Volatility for ' +
    stock_symbol)
 
# get x and y coordinate limits
x_corr = ax.get_xlim()
y_corr = ax.get_ylim()
 
# make room for text
header = y_corr[1] / 5
y_corr = (y_corr[0], y_corr[1] + header)
ax.set_ylim(y_corr[0], y_corr[1])


# print historical volatility on plot
x = x_corr[0] + (x_corr[1] - x_corr[0]) / 30
y = y_corr[1] - (y_corr[1] - y_corr[0]) / 15
ax.text(x, y , 'Annualized Volatility: ' + str(np.round(std*100, 1))+'%',
    fontsize=11, fontweight='bold')
x = x_corr[0] + (x_corr[1] - x_corr[0]) / 15
y -= (y_corr[1] - y_corr[0]) / 20

# display plot
fig.tight_layout()
fig.savefig('historical volatility.png')
