---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

## Root account from compmech

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('fivethirtyeight')
from numpy.random import default_rng
```

# Project #02 - NYSE random walk predictor

In the [Stats and Monte Carlo](../module_02/04_Stats_and_Montecarlo) module, you created a Brownian motion model to predict the motion of particles in a fluid. The Monte Carlo model took steps in the x- and y-directions with random magnitudes. 

This [random walk](https://en.wikipedia.org/wiki/Random_walk_hypothesis) can be used to predict stock prices. Let's take a look at some data from the New York Stock Exchange [NYSE](https://www.kaggle.com/dgawlik/nyse) from 2010 through 2017. 

> __Important Note__: 
> I am not a financial advisor and these models are _purely_ for academic exercises. If you decide to use anything in these notebooks to make financial decisions, it is _at your own risk_. _I am not an economist/financial advisor/etc., I am just a Professor who likes to learn and exeriment._

Here, I will show an example workflow to analyze and predict the Google
stock price [[GOOGL]](https://en.wikipedia.org/wiki/Alphabet_Inc.) from
2010 - 2014. Then, you can choose your own stock price to evaluate and
create a predictive model.

1. Explore data and select data of interest
2. Find statistical description of data: mean and standard deviation
3. Create random variables
4. Generate random walk for [[GOOGL]](https://en.wikipedia.org/wiki/Alphabet_Inc.) stock opening price

+++

## 1. Explore data

Here, I load the data into a Pandas dataframe to see what headings and values are available. I see two columns that I want to analyze
- 'date'
- 'open'

```{code-cell} ipython3
data = pd.read_csv('../data/nyse-data.csv')
data['date'] = pd.to_datetime(data['date'])
data
```

I only want the `symbol == GOOGL` data, so I use a Pandas call. I also want to remove the big drop in price after Mar, 2014, so I specify the date < 2014-03-01.

```{code-cell} ipython3
google_data = data[data['symbol'] == 'GOOGL']

plt.plot(google_data['date'], google_data['open'], 'r')

# remove data > 2014-03-01

google_data_pre_2014 = google_data[ google_data['date'] < pd.to_datetime('2014-03-01')]
plt.plot(google_data_pre_2014['date'], google_data_pre_2014['open'])
plt.xlabel('date')
plt.ylabel('opening price (\$)');
```

## 2. Data analysis

The GOOGL stock nearly doubled in price from 2010 through 2014. Day-to-day, the price fluctuates randomly. Here, I look at the fluctuations in price using [`np.diff`](https://numpy.org/doc/1.20/reference/generated/numpy.diff.html).

```{code-cell} ipython3
dprice = np.diff(google_data_pre_2014['open'])
plt.plot(google_data_pre_2014['date'][1:], dprice)
plt.xlabel('date')
plt.ylabel('change in opening price (\$/day)');
```

Looking at the price day-to-day, it would appear to be an average change of $0/day. Next, I explore the statistical results of the change in opening price

- mean
- standard deviation
- histogram

```{code-cell} ipython3
mean_dprice = np.mean(dprice)
std_dprice = np.std(dprice)
x = np.linspace(-40, 40)
from scipy import stats
price_pdf = stats.norm.pdf(x, loc = mean_dprice, scale = std_dprice)
```

```{code-cell} ipython3
plt.hist(dprice, 50, density=True)
plt.plot(x, price_pdf)
plt.title('GOOGL changes in price over 4 years\n'+
         'avg: \${:.2f} stdev: \${:.2f}'.format(mean_dprice, std_dprice));
```

From this statistical result, it looks like the price changes followed a normal distribution with an average change of $\$0.57$ and a standard deviation of $\$9.84$.

+++

## 3. Create random variables

Now, I know the distribution shape and characteristics to simulate the random walk price changes for the GOOGL prices each day. Here, I generate random variables with the following array structure:

Date |model 1| model 2| model 3|...| model N|
---|---|---|---|---|---|
day 1|$\Delta \$ model~1$|$\Delta \$ model~2$|$\Delta \$ model~3$|...|$\Delta \$ model~N$|
day 2|$\Delta \$ model~1$|$\Delta \$ model~2$|$\Delta \$ model~3$|...|$\Delta \$ model~N$|
...|...|...|...|...|...|

Each column is one random walk model. Each row is one simulated day. If I want to look at _one_ model predition, I would plot one column. If I want to look at the _average_ result, I take the average of each row. To start, I'll create 100 random walk models. I use the [`normal`](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.normal.html#numpy.random.Generator.normal) distribution to match the statistical distribution I found in part 2.

```{code-cell} ipython3
rng = default_rng(42) 
N_models = 100 
dprice_model = rng.normal(size = (len(google_data_pre_2014), N_models), loc = 0.568, scale = 9.838)


plt.hist(dprice, 50, density=True, label = 'NYSE data')
plt.plot(x, price_pdf)
plt.hist(dprice_model[:, 0], 50, density = True, 
         histtype = 'step', 
         linewidth = 3, label = 'model prediction 1')
plt.title('GOOGL changes in price over 4 years\n'+
         'avg: \${:.2f} stdev: \${:.2f}'.format(mean_dprice, std_dprice))
plt.legend();
```

## 4. Generate random walk predictions

Above, I show tha the simulated data follows the requested normal distribution. Now, I can cumulatively sum these steps to predict the stock prices each day. I use the  [`np.cumsum`](https://numpy.org/doc/1.20/reference/generated/numpy.cumsum.html) argument, `axis = 0` to sum along the columns i.e. each row becomes the sum of previous rows. 

```python
>>> a = np.array([[1,2,3], [4,5,6]])
>>> a
array([[1, 2, 3],
       [4, 5, 6]])
>>>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
array([[1, 2, 3],
       [5, 7, 9]])
```

Then, I plot all of the random walk models to compare to the NYSE data. The models are given transparency using the `alpha = 0.3` command (_`alpha = 0` is invisible, `alpha = 1` is opaque_).

```{code-cell} ipython3
price_model = np.cumsum(dprice_model, axis = 0) + google_data_pre_2014['open'].values[0]

plt.plot(google_data_pre_2014['date'], price_model, alpha = 0.3);

plt.plot(google_data_pre_2014['date'], google_data_pre_2014['open'], c = 'k', label = 'NYSE data')
plt.xlabel('date')
plt.ylabel('opening price (\$)');
```

As you would expect, there are a wide variety of predictions for the price of GOOGL stocks using random numbers. Next, I try to get some insight into the average changes in the random walk model. I use the `np.mean` and `np.std` across the columns of the `price_model` prediction data, using `axis = 1` now.

```{code-cell} ipython3
price_model_avg = np.mean(price_model, axis = 1)
price_model_std = np.std(price_model, axis = 1)

plt.plot(google_data_pre_2014['date'], price_model, alpha = 0.3);

plt.plot(google_data_pre_2014['date'], google_data_pre_2014['open'], c = 'k', label = 'NYSE data')
plt.xlabel('date')
plt.ylabel('opening price (\$)');

skip = 100
plt.errorbar(google_data_pre_2014['date'][::skip], price_model_avg[::skip], 
             yerr = price_model_std[::skip], 
             fmt = 'o',
             c = 'r', 
             label = 'model result', 
            zorder = 3);
plt.legend();
    
```

## Wrapping up

In this analysis, I went through data exploration, analysis, and Monte Carlo model prediction. The average random walk should resemble a straight line. There are further insights you can get by analyzing the random walk data, but for now it looks like we can accurately predict the growth of GOOGL stock over four years. __What are some caveats to this method? If we continue to predict prices into 2015, what would happen compared to the real data?__

## Next Steps

Now, you can try your hand at predicting stock prices on your own stock. Choose your own stock symbol and go through the same 4 steps I detailed above:
1. Explore data and select _your own_ stock of interest
2. Find statistical description of data: mean and standard deviation _use some of the graphing + analysis techniques in [01_Cheers_stats_beers](../module_02/01_Cheers_Stats_Beers) and [02_Seeing_stats](02_Seeing_Stats).
3. Create random variables
4. Generate random walk for _choose your own_ stock opening price

Here are the list of stocks in this dataset:
'A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABT', 'ACN', 'ADBE',
       'ADI', 'ADM', 'ADP', 'ADS', 'ADSK', 'AEE', 'AEP', 'AES', 'AET',
       'AFL', 'AGN', 'AIG', 'AIV', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALK',
       'ALL', 'ALLE', 'ALXN', 'AMAT', 'AME', 'AMG', 'AMGN', 'AMP', 'AMT',
       'AMZN', 'AN', 'ANTM', 'AON', 'APA', 'APC', 'APD', 'APH', 'ARNC',
       'ATVI', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXP', 'AYI', 'AZO', 'BA',
       'BAC', 'BAX', 'BBBY', 'BBT', 'BBY', 'BCR', 'BDX', 'BEN', 'BHI',
       'BIIB', 'BK', 'BLK', 'BLL', 'BMY', 'BSX', 'BWA', 'BXP', 'C', 'CA',
       'CAG', 'CAH', 'CAT', 'CB', 'CBG', 'CBS', 'CCI', 'CCL', 'CELG',
       'CERN', 'CF', 'CFG', 'CHD', 'CHK', 'CHRW', 'CHTR', 'CI', 'CINF',
       'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC',
       'CNP', 'COF', 'COG', 'COH', 'COL', 'COO', 'COP', 'COST', 'COTY',
       'CPB', 'CRM', 'CSCO', 'CSRA', 'CSX', 'CTAS', 'CTL', 'CTSH', 'CTXS',
       'CVS', 'CVX', 'CXO', 'D', 'DAL', 'DD', 'DE', 'DFS', 'DG', 'DGX',
       'DHI', 'DHR', 'DIS', 'DISCA', 'DISCK', 'DLPH', 'DLR', 'DLTR',
       'DNB', 'DOV', 'DOW', 'DPS', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN',
       'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'EMN', 'EMR',
       'ENDP', 'EOG', 'EQIX', 'EQR', 'EQT', 'ES', 'ESRX', 'ESS', 'ETFC',
       'ETN', 'ETR', 'EVHC', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F',
       'FAST', 'FB', 'FBHS', 'FCX', 'FDX', 'FE', 'FFIV', 'FIS', 'FISV',
       'FITB', 'FL', 'FLIR', 'FLR', 'FLS', 'FMC', 'FOX', 'FOXA', 'FRT',
       'FSLR', 'FTI', 'FTR', 'FTV', 'GD', 'GE', 'GGP', 'GILD', 'GIS',
       'GLW', 'GM', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GPS', 'GRMN', 'GS',
       'GT', 'GWW', 'HAL', 'HAR', 'HAS', 'HBAN', 'HBI', 'HCA', 'HCN',
       'HCP', 'HD', 'HES', 'HIG', 'HOG', 'HOLX', 'HON', 'HP', 'HPE',
       'HPQ', 'HRB', 'HRL', 'HRS', 'HSIC', 'HST', 'HSY', 'HUM', 'IBM',
       'ICE', 'IDXX', 'IFF', 'ILMN', 'INTC', 'INTU', 'IP', 'IPG', 'IR',
       'IRM', 'ISRG', 'ITW', 'IVZ', 'JBHT', 'JCI', 'JEC', 'JNJ', 'JNPR',
       'JPM', 'JWN', 'K', 'KEY', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI',
       'KMX', 'KO', 'KORS', 'KR', 'KSS', 'KSU', 'L', 'LB', 'LEG', 'LEN',
       'LH', 'LKQ', 'LLL', 'LLTC', 'LLY', 'LMT', 'LNC', 'LNT', 'LOW',
       'LRCX', 'LUK', 'LUV', 'LVLT', 'LYB', 'M', 'MA', 'MAA', 'MAC',
       'MAR', 'MAS', 'MAT', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT',
       'MET', 'MHK', 'MJN', 'MKC', 'MLM', 'MMC', 'MMM', 'MNK', 'MNST',
       'MO', 'MON', 'MOS', 'MPC', 'MRK', 'MRO', 'MSFT', 'MSI', 'MTB',
       'MTD', 'MU', 'MUR', 'MYL', 'NAVI', 'NBL', 'NDAQ', 'NEE', 'NEM',
       'NFLX', 'NFX', 'NI', 'NKE', 'NLSN', 'NOC', 'NOV', 'NRG', 'NSC',
       'NTAP', 'NTRS', 'NUE', 'NVDA', 'NWL', 'NWS', 'NWSA', 'O', 'OKE',
       'OMC', 'ORCL', 'ORLY', 'OXY', 'PAYX', 'PBCT', 'PBI', 'PCAR', 'PCG',
       'PCLN', 'PDCO', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH',
       'PHM', 'PKI', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PPG', 'PPL',
       'PRGO', 'PRU', 'PSA', 'PSX', 'PVH', 'PWR', 'PX', 'PXD', 'PYPL',
       'QCOM', 'QRVO', 'R', 'RAI', 'RCL', 'REGN', 'RF', 'RHI', 'RHT',
       'RIG', 'RL', 'ROK', 'ROP', 'ROST', 'RRC', 'RSG', 'RTN', 'SBUX',
       'SCG', 'SCHW', 'SE', 'SEE', 'SHW', 'SIG', 'SJM', 'SLB', 'SLG',
       'SNA', 'SNI', 'SO', 'SPG', 'SPGI', 'SPLS', 'SRCL', 'SRE', 'STI',
       'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SWN', 'SYF', 'SYK', 'SYMC',
       'SYY', 'T', 'TAP', 'TDC', 'TDG', 'TEL', 'TGNA', 'TGT', 'TIF',
       'TJX', 'TMK', 'TMO', 'TRIP', 'TROW', 'TRV', 'TSCO', 'TSN', 'TSO',
       'TSS', 'TWX', 'TXN', 'TXT', 'UAA', 'UAL', 'UDR', 'UHS', 'ULTA',
       'UNH', 'UNM', 'UNP', 'UPS', 'URBN', 'URI', 'USB', 'UTX', 'V',
       'VAR', 'VFC', 'VIAB', 'VLO', 'VMC', 'VNO', 'VRSK', 'VRSN', 'VRTX',
       'VTR', 'VZ', 'WAT', 'WBA', 'WDC', 'WEC', 'WFC', 'WFM', 'WHR',
       'WLTW', 'WM', 'WMB', 'WMT', 'WRK', 'WU', 'WY', 'WYN', 'WYNN',
       'XEC', 'XEL', 'XL', 'XLNX', 'XOM', 'XRAY', 'XRX', 'XYL', 'YHOO',
       'YUM', 'ZBH', 'ZION', 'ZTS'

+++

What are some caveats to this method? If we continue to predict prices into 2015, what would happen compared to the real data?

This method works best when there are no jumps in the data. The Google stock data has a big drop in 2014. If that drop is taken into consideration, the data will change a lot. For example, the change in opening price per day data will no longer be centered around $0/day, as the significant decrease in it's price in 2014 will show a large dip. This is also true for the other graphs above. If the data continued into 2015, the mean of the data would be much lower than when all data after 2014 is neglected.

+++

## 1. Explore data - Apple (APPL)

```{code-cell} ipython3
data = pd.read_csv('../data/nyse-data.csv')
data['date'] = pd.to_datetime(data['date'])
data
```

```{code-cell} ipython3
apple_data = data[data['symbol'] == 'AAPL']

plt.plot(apple_data['date'], apple_data['open'])

#remove data > 2014-06-09

apple_data_pre_2014 = apple_data[apple_data['date'] < pd.to_datetime('2014-06-09')]
plt.plot(apple_data_pre_2014['date'], apple_data_pre_2014['open'], 'k')
plt.xlabel('date')
plt.ylabel('opening price (\$)');
```

I chose to look at Apple stock. Similar to Google stock that was shown in the example, in 2014 there was a significant drop in opening price. According to an article by Huffington Post, this drop in price was simply due to Apple spliting their stocks. To get an accurate sense of the trend of the data, the black portion of the chart shown above will be the portion of the data being looked at for analysis. (https://www.washingtonpost.com/news/the-switch/wp/2014/06/09/apples-stock-price-just-dropped-more-than-500-a-share-but-dont-panic/)

```{code-cell} ipython3
apple_dprice = np.diff(apple_data_pre_2014['open']) #This is taking the difference of the array along the axis.
plt.plot(apple_data_pre_2014['date'][1:], apple_dprice) #Creates a plot showing the change in opening price per day.
plt.xlabel('date')
plt.ylabel('change in opening price (\$/day)');
```

Similar to the example with Google above, the data is centered around approximately $0/day.

```{code-cell} ipython3
mean_apple_dprice = np.mean(apple_dprice) # Mean of Apple stock's opening price per day data.
std_apple_dprice = np.std(apple_dprice) # Standard Deviation of Apple stock's opening price per day data.
x = np.linspace(-40, 40)
from scipy import stats
apple_price_pdf = stats.norm.pdf(x, loc = mean_apple_dprice, scale = std_apple_dprice) # loc = mean, scale = stand dev
```

```{code-cell} ipython3
plt.hist(apple_dprice, 50, density=True) #Plots a histogram of the changing of price of Apple stock per day, using 50 boxes.
plt.plot(x, apple_price_pdf) 
plt.title('APPL changes in price over 4 years\n'+
         'avg: \${:.2f} stdev: \${:.2f}'.format(mean_apple_dprice, std_apple_dprice));
```

```{code-cell} ipython3
rng = default_rng(42) #Random number generator with seed to ensure the same random numbers
N_models = 100 #Creates 100 random walks
dprice_model = rng.normal(size = (len(apple_data_pre_2014), N_models), loc = mean_apple_dprice, scale = std_apple_dprice)
#Size indicates how many changes in price we want to display, loc is the mean of the data, scale is the standard dev.

plt.hist(apple_dprice, 50, density=True, label = 'NYSE data')
plt.plot(x, apple_price_pdf)
plt.hist(dprice_model[:, 0], 50, density = True, 
         histtype = 'step', 
         linewidth = 3, label = 'model prediction 1')
plt.title('APPL changes in price over 4 years\n'+
         'avg: \${:.2f} stdev: \${:.2f}'.format(mean_apple_dprice, std_apple_dprice))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right');
```

This data seems to make sense, as the model prediction is very similar to the bell curve and follows the same trend as the NYSE data. This is a good indication that our model is accurate.

```{code-cell} ipython3
apple_price_model = np.cumsum(dprice_model, axis = 0) + apple_data_pre_2014['open'].values[0]

plt.plot(apple_data_pre_2014['date'], apple_price_model, alpha = 0.3);

plt.plot(apple_data_pre_2014['date'], apple_data_pre_2014['open'], c = 'k', label = 'NYSE data')
plt.xlabel('date')
plt.ylabel('opening price (\$)');
```

This plot is comparing all 100 of the random walks that were generated in the previous step to the data from the NYSE data, where the NYSE data is the black line. We are using the cumulative sum, which is kind of like taking an integral of all the values, and adding the initial price values. We do this to minimize the error in the model because if we know the inital conditions of the model we can predict more accurate values from there. It is important to note that the walks all originate at the same point.

```{code-cell} ipython3
apple_price_model_avg = np.mean(apple_price_model, axis = 1) # This finds the average of the data along axis 1
apple_price_model_std = np.std(apple_price_model, axis = 1) # This finds the standard deviation of the data along axis 1

plt.plot(apple_data_pre_2014['date'], apple_price_model, alpha = 0.3); #Plots the random walks 

plt.plot(apple_data_pre_2014['date'], apple_data_pre_2014['open'], c = 'k', label = 'NYSE data') #Plots the NYSE data
plt.xlabel('date')
plt.ylabel('opening price (\$)');

skip = 100
plt.errorbar(apple_data_pre_2014['date'][::skip], apple_price_model_avg[::skip], #Plots the average of all the random walks
             yerr = apple_price_model_std[::skip], #Indicates how long to make the error bar on the y axis
             fmt = 'o',
             c = 'r', 
             label = 'model result', 
            zorder = 3);
plt.legend();
    
```

Finally, the figure above is showing the error bars in the data. The dots in the middle of the error bar indicate the average of the price model defined in the step above using the cumulative sum and the initial conditions. The length of the error bars on the y axis is the standard deviation. This is helpful for showing the accuracy of the predicted data against the actual NYSE data.

```{code-cell} ipython3
apple_data = data[data['symbol'] == 'AAPL']

plt.plot(apple_data['date'], apple_data['open'])

#remove data > 2014-06-09

apple_data_post_2014 = apple_data[apple_data['date'] > pd.to_datetime('2014-06-09')]
plt.plot(apple_data_post_2014['date'], apple_data_post_2014['open'], 'k')
plt.xlabel('date')
plt.ylabel('opening price (\$)');
```

```{code-cell} ipython3
post_apple_dprice = np.diff(apple_data_post_2014['open']) #This is taking the difference of the array along the axis.
plt.plot(apple_data_post_2014['date'][1:], post_apple_dprice) #Creates a plot showing the change in opening price per day.
plt.xlabel('date')
plt.ylabel('change in opening price (\$/day)');
```

```{code-cell} ipython3
post_mean_apple_dprice = np.mean(post_apple_dprice) # Mean of Apple stock's opening price per day data.
post_std_apple_dprice = np.std(post_apple_dprice) # Standard Deviation of Apple stock's opening price per day data.
x = np.linspace(-40, 40)
from scipy import stats
post_apple_price_pdf = stats.norm.pdf(x, loc = post_mean_apple_dprice, scale = post_std_apple_dprice) # loc = mean, scale = stand dev
```

```{code-cell} ipython3
plt.hist(post_apple_dprice, 50, density=True) #Plots a histogram of the changing of price of Apple stock per day, using 50 boxes.
plt.plot(x, post_apple_price_pdf) 
plt.title('APPL changes in price over 4 years\n'+
         'avg: \${:.2f} stdev: \${:.2f}'.format(post_mean_apple_dprice, post_std_apple_dprice));
```

```{code-cell} ipython3
rng = default_rng(42) #Random number generator with seed to ensure the same random numbers
N_models = 100 #Creates 100 random walks
post_dprice_model = rng.normal(size = (len(apple_data_post_2014), N_models), loc = post_mean_apple_dprice, scale = post_std_apple_dprice)
#Size indicates how many changes in price we want to display, loc is the mean of the data, scale is the standard dev.

plt.hist(post_apple_dprice, 50, density=True, label = 'NYSE data')
plt.plot(x, post_apple_price_pdf)
plt.hist(post_dprice_model[:, 0], 50, density = True, 
         histtype = 'step', 
         linewidth = 3, label = 'model prediction 1')
plt.title('APPL changes in price over 4 years\n'+
         'avg: \${:.2f} stdev: \${:.2f}'.format(post_mean_apple_dprice, post_std_apple_dprice))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper right');
```

```{code-cell} ipython3
post_apple_price_model = np.cumsum(post_dprice_model, axis = 0) + apple_data_post_2014['open'].values[0]

plt.plot(apple_data_post_2014['date'], post_apple_price_model, alpha = 0.3);

plt.plot(apple_data_post_2014['date'], apple_data_post_2014['open'], c = 'k', label = 'NYSE data')
plt.xlabel('date')
plt.ylabel('opening price (\$)');
```

```{code-cell} ipython3
post_apple_price_model_avg = np.mean(post_apple_price_model, axis = 1) # This finds the average of the data along axis 1
post_apple_price_model_std = np.std(post_apple_price_model, axis = 1) # This finds the standard deviation of the data along axis 1

plt.plot(apple_data_post_2014['date'], post_apple_price_model, alpha = 0.3); #Plots the random walks 

plt.plot(apple_data_post_2014['date'], apple_data_post_2014['open'], c = 'k', label = 'NYSE data') #Plots the NYSE data
plt.xlabel('date')
plt.ylabel('opening price (\$)');

skip = 100
plt.errorbar(apple_data_post_2014['date'][::skip], post_apple_price_model_avg[::skip], #Plots the average of all the random walks
             yerr = post_apple_price_model_std[::skip], #Indicates how long to make the error bar on the y axis
             fmt = 'o',
             c = 'r', 
             label = 'model result', 
            zorder = 3);
plt.legend();
    
```

```{code-cell} ipython3

```
