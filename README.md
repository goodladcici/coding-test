# coding-test
 *I would like to replicate a monthly momentum strategy based on cumulated 
 return in pervious 11 month with 1 month lagged . Here tickers are labelled 
 from 1 to 10 based on the return the signal. Long highest return group and 
 short lowest return group. However the backtesting seems bad. Then I plot 
 the mean of return of each groups finding that high return had low return 
 have a relative high return in the month of trading, while mediate  groups
 are trend to have low return. The reason I guess is that high return stocks
 have momentum to keep their increasing, while low return stocks have
 potential bounce energy in short term future. So I modified the strategy to 
 longing high (bin 10),shorting mid (bin 6), or longing high+low (bin10+bin1)
 and shorting mid(bin 6). The result seems quite good, so volitility plays a great role in this market and various strategies based on 
 volitility are worth to be test. Also, I divided each 
 strategy into equal weight portfolio allocation and value weight portfolio allocation.
 Furthermore, I found tickers are from tokyo market and I have heard that some 
 literatures suggested this momentum strategy does not work in Japan.*
 
![mean return of each bin](https://user-images.githubusercontent.com/99357310/164297863-f5a30420-f17e-4daf-9175-e621d75c5740.png)

Bar chart above is mean return of each bin

![](https://user-images.githubusercontent.com/99357310/164297893-2e240832-9257-4aca-9ba8-23d4608b27da.png)

In this plot, cum1 is for bin10-bin1, cum2 is for bin10-bin6 and cum3 is for 0.5*bin10+0.5*bin1-bin6. mkt is for market portfolio made by all tickers in 
data file
