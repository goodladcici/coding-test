# coding-test
 *I would like to replicate a monthly momentum strategy based on cumulated 
 return in pervious 11 month with 1 month lagged. Also, I divided each 
 strategy into equal weight portfolio allocation and value weight portfolio allocation. Here tickers are labelled 
 from 1 to 10 based on the return the signal. Long highest return group and 
 short lowest return group. However the backtesting seems bad. Then I plot 
 the mean of return of each groups finding that high return had low return 
 have a relative high return in the month of trading, while mediate  groups
 are trend to have low return. The reason I guess is that high return stocks
 have momentum to keep their increasing, while low return stocks have
 potential bounce energy in short term future. So I modified the strategy to 
 longing high (bin 10),shorting mid (bin 6), or longing high+low (bin10+bin1)
 and shorting mid(bin 6). The result seems better with almost same return with market but more smooth. Meanwhile, I calculate the sharpe ratio of each strategy.*
 
 <img width="205" alt="Screenshot 2022-04-21 at 14 49 10" src="https://user-images.githubusercontent.com/99357310/164391492-2ab5e7ab-20ea-45a3-9f9d-17b4d72002af.png">

 *Furthermore, I found tickers are from tokyo market and I have heard that some 
 literatures suggested this momentum strategy does not work well in Japan because there will be reversal effect, which implies low return will bounce back and high return will reverse to be low. Therefore, according to the bin chart below, the strategy with low min high (bin1-bin9 or bin1-bin8) is worth and reasonable to be tested(i do not test here because time limit)*
 
![mean return of each bin](https://user-images.githubusercontent.com/99357310/164297863-f5a30420-f17e-4daf-9175-e621d75c5740.png)

*Bar chart above is mean return of each bin*

![Figure 2022-04-21 144444](https://user-images.githubusercontent.com/99357310/164390813-a4f21f42-6f0c-4652-8f2f-074d5cc2946b.png)


*In this plot, cum1 is for bin10-bin1, cum2 is for bin10-bin6 and cum3 is for 0.5\*bin10+0.5\*bin1-bin6. mkt is for market portfolio made by all tickers in 
data file*
