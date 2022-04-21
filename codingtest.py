#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:15:23 2022

@author: xietianci
"""
import os
import pandas as pd
import numpy as np

os.chdir('/Users/xietianci/Desktop/coding_test')

# I would like to replicate a monthly momentum strategy based on cumulated 
# return in pervious 11 month with 1 month lagged . Here tickers are labelled 
# from 1 to 10 based on the return the signal. Long highest return group and 
# short lowest return group. However the backtesting seems bad. Then I plot 
# the mean of return of each groups finding that high return had low return 
# have a relative high return in the month of trading, while mediate  groups
# are trend to have low return. The reason I guess is that high return stocks
# have momentum to keep their increasing, while low return stocks have
# potential bounce energy in short term future. So I modified the strategy to 
# longing high (bin 10),shorting mid (bin 6), or longing high+low (bin10+bin1)
# and shorting mid(bin 6). The result seems quite good. Also, I divided each 
# strategy into equal weight portfolio allocation and value weight portfolio.
# furthermore, I found tickers are from tokyo market and I have heard some 
# literature suggested this momentum strategy does not work in JP.

# define a function that can seperate quantiles
def apply_quantiles(x, bins=5):
    x = pd.Series(x)
    quantiles = np.quantile(x,np.linspace(0, 1, bins+1))
    quantiles[0] = x.min() - 1
    quantiles[-1] = x.max() + 1
    return pd.cut(x, quantiles, labels=False) + 1

# define a rolling production function
def rolling_prod(a, n=11) :
    ret = np.cumprod(a.values)
    ret[n:] = ret[n:] / ret[:-n]
    ret[:n-1] = np.nan
    return pd.Series(ret,index=a.index)

def sharpe(x):
    return x.mean() / x.std()

# read data and store market portfolio
df = pd.read_csv('data.csv')
mkt = df.copy()
mkt['price_lag1'] = mkt.groupby(['ticker'])['last'].shift(1)
mkt['ret'] = (mkt['last'])/mkt['price_lag1']
mkt['month']=mkt['date'].apply(lambda x:x[:7])
mon_ret = mkt.groupby(['ticker','month'])['ret'].prod().values
mkt = mkt.drop_duplicates(subset=['ticker','month']).reset_index(drop=True)
mkt['ret'] = mon_ret
rmkt = mkt.groupby('month')['ret'].mean().reset_index()
rmkt = rmkt.rename(columns={'ret':'mkt'})
l = []

# apply a filter to eliminate the 1/4 of the most illiquid tickers
tickers = sorted([x for x in set(df.ticker)])
liquid = df.groupby('ticker')['volume'].mean()
liquid_rank = apply_quantiles(liquid,bins=4)
tickers_filter1 = pd.DataFrame({'ticker':tickers,'rank':liquid_rank})
tickers_set = tickers_filter1[(tickers_filter1['rank']!=1)]['ticker']

for x in range(len(df)):
    if (df.iloc[x,0] in tickers_set.to_list()):
        l.append(1)
    else:
        l.append(0)

df['filter1'] = l
df = df[df['filter1']==1].reset_index(drop=True)


# generate return and momentum
df['price_lag1'] = df.groupby(['ticker'])['last'].shift(1)
df['ret'] = (df['last'])/df['price_lag1']
df['month']=df['date'].apply(lambda x:x[:7])
monthly_ret = df.groupby(['ticker','month'])['ret'].prod()
df = df.drop_duplicates(subset=['ticker','month']).reset_index(drop=True)
df['ret'] = monthly_ret.values

roll_11 = df.groupby(['ticker'])['ret'].apply(rolling_prod)
df['roll_11']=roll_11
df['mom'] = df['roll_11'].shift(1)


# rank data in from 1 to 10 based on momentum
df1 = df[['ticker','month','ret','mom','last','price_lag1']]
df1 = df1[df1['mom'].notna()]
df1['bin'] = (
    df1
    .groupby('month')
    .apply(lambda group: np.ceil(group['mom'].rank() / len(group['mom']) * 10))
    ).reset_index(level=[0],drop=True).sort_index()
    
# plot the bar chart for return mean of each bin
df1['ret-1'] = df1['ret']-1
ax = df1.groupby('bin')['ret-1'].apply(np.mean).plot(kind='bar')

# calculate return for each bin using equal weight and value weight
portfolios1 = (
    df1
    .groupby(['month', 'bin'])
    .apply(
        lambda g: pd.Series({
            'portfolio': g['ret'].mean(),
            'portfolio1': (g['ret'] * g['price_lag1']).sum() / g['price_lag1'].sum()
        })
    )
).reset_index()

# building strategies
strategy1 = pd.merge(
    portfolios1.query('bin==10'),
    portfolios1.query('bin==1'),
    suffixes=['_long', '_short'],
    on='month')
strategy2 = pd.merge(
    portfolios1.query('bin==10'),
    portfolios1.query('bin==6'),
    suffixes=['_long', '_short'],
    on='month')

# calculate strategy return monthly
strategy1['strategy'] = strategy1['portfolio_long'] - strategy1['portfolio_short']
strategy1['strategy1'] = strategy1['portfolio1_long'] - strategy1['portfolio1_short']
strategy2['strategy'] = strategy2['portfolio_long'] - strategy2['portfolio_short']
strategy2['strategy1'] = strategy2['portfolio1_long'] - strategy2['portfolio1_short']
strategy1['strategy2'] = 0.5*strategy1['portfolio_long'] + 0.5*strategy1['portfolio_short']- strategy2['portfolio_short']
strategy1['strategy3'] = 0.5*strategy1['portfolio1_long'] + 0.5*strategy1['portfolio1_short']- strategy2['portfolio1_short']

# calculate cumulate return for each strategy
strategy1['cum1_ew'] = (strategy1['strategy'] + 1).cumprod() - 1
strategy1['cum1_vw'] = (strategy1['strategy1'] + 1).cumprod() - 1
strategy2['cum'] = (strategy2['strategy'] + 1).cumprod() - 1
strategy2['cum1'] = (strategy2['strategy1'] + 1).cumprod() - 1
strategy1['cum3_ew'] = (strategy1['strategy2'] + 1).cumprod() - 1
strategy1['cum3_vw'] = (strategy1['strategy3'] + 1).cumprod() - 1
strategy1['mkt'] = rmkt['mkt'].cumprod()-1
strategy1['cum2_ew'] = strategy2['cum']
strategy1['cum2_vw'] = strategy2['cum1'] 
#strategy1 = strategy1.merge(rmkt,on='month',how='left')

# plot back testing
strategy1\
    .assign(month=pd.to_datetime(strategy2['month']))\
    .assign(cum1_ew=strategy1['cum1_ew']+1)\
    .assign(cum1_vw=strategy1['cum1_vw']+1)\
    .assign(cum2_ew=strategy1['cum2_ew']+1)\
    .assign(cum2_vw=strategy1['cum2_vw']+1)\
    .assign(cum3_ew=strategy1['cum3_ew']+1)\
    .assign(cum3_vw=strategy1['cum3_vw']+1)\
    .assign(mkt=strategy1['mkt']+1)\
    .plot(x='month',\
          y=['cum1_ew','cum1_vw','cum2_ew','cum2_vw','cum3_ew','cum3_vw','mkt']\
              ,figsize = (7.5,6)).grid(axis='y')

sr1e = sharpe(strategy1['strategy'])
sr1v = sharpe(strategy1['strategy1'])
sr2e = sharpe(strategy2['strategy'])
sr2v = sharpe(strategy2['strategy1'])
sr3e = sharpe(strategy1['strategy2'])
sr3v = sharpe(strategy1['strategy3'])
srmkt = sharpe(rmkt['mkt']-1)

print(sr1e)
print(sr1v)
print(sr2e)
print(sr2v)
print(sr3e) 
print(sr3v) 
print(srmkt) 








