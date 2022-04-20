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



def apply_quantiles(x, bins=10):
    x = pd.Series(x)
    quantiles = np.quantile(
        x[x.notnull()],
        np.linspace(0, 1, bins+1)
    )
    quantiles[0] = x.min() - 1
    quantiles[-1] = x.max() + 1
    return pd.cut(x, quantiles, labels=False) + 1

def rolling_prod(a, n=11) :
    ret = np.cumprod(a.values)
    ret[n:] = ret[n:] / ret[:-n]
    ret[:n-1] = np.nan
    return pd.Series(ret,index=a.index)


df = pd.read_csv('data.csv')
l = []

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

df['price_lag1'] = df.groupby(['ticker'])['last'].shift(1)
df['ret'] = (df['last'])/df['price_lag1']
df['month']=df['date'].apply(lambda x:x[:7])
monthly_ret = df.groupby(['ticker','month'])['ret'].prod()
df = df.drop_duplicates(subset=['ticker','month'])
df['ret'] = monthly_ret.values


roll_11 = df.groupby(['ticker'])['ret'].apply(rolling_prod)
df['roll_11']=roll_11
df['mom'] = df['roll_11'].shift(1)

df1 = df[['ticker','date','ret','mom','last']]
df1 = df1[df1['mom'].notna()]
df1['bin'] = (
    df1
    .groupby('date')
    .apply(lambda group: np.ceil(group['mom'].rank() / len(group['mom']) * 10))
    ).reset_index(level=[0], drop=True).sort_index()

portfolios = (
    df1
    .groupby(['date', 'bin'])
    .apply(
        lambda g: pd.Series({
            'portfolio': g['ret'].mean()
        })
    )
).reset_index()
    
portfolios2 = pd.merge(
    portfolios.query('bin==10'),
    portfolios.query('bin==1'),
    suffixes=['_long', '_short'],
    on='date')

portfolios2['strategy'] = portfolios2['portfolio_long'] - portfolios2['portfolio_short']
portfolios2['cum'] = (portfolios2['strategy'] + 1).cumprod() - 1

portfolios2\
    .assign(date=pd.to_datetime(portfolios2['date']))\
    .assign(cum=portfolios2['cum']+1)\
    .plot(x='date', y=['cum']).grid(axis='y')
    



