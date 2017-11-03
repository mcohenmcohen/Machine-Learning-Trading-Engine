'''
Technical analysis utilities to calculate technical analysis indicators
and related data manipulation
'''
# Author:  Matt Cohen
# Python Version 2.7

import pandas as pd
import numpy as np
import statsmodels.api as sm
import talib
import time
# https://github.com/mrjbq7/ta-lib
# index - https://github.com/mrjbq7/ta-lib/blob/master/docs/index.md
# documentation: https://github.com/mrjbq7/ta-lib/blob/master/docs/func.md


def run_techicals(df):
    '''
    Preform technical anlysis calcuaitons and add to a dataframe.

    Algorithms stem from the TAlib module other helper functions, including distretizing data,
    using alternate series, and time shifting

    TAlib docs: https://github.com/mrjbq7/ta-lib/blob/master/docs/func_groups/momentum_indicators.md
    '''
    days_back = 0
    opn = np.roll(df['exp_smooth_open'], days_back)
    high = np.roll(df['exp_smooth_high'], 0)
    low = np.roll(df['exp_smooth_low'], 0)
    close = np.roll(df['exp_smooth_close'], days_back)
    volume = np.roll(df['exp_smooth_volume'], 0)

    # series = df['close'].values
    series = close

    # df['roc'] = talib.ROCP(series, timeperiod=1)
    df['roc'] = rate_of_change(df['close'], 1)  # > 0.04
    df['roc_d'] = discrete_series_pos_neg(df['roc'])  # > 80.638
    df['rsi'] = talib.RSI(series, timeperiod=14)  # > 80.638
    df['rsi_d'] = continuous_to_discrete_w_bounds(df['rsi'], 30, 70)  # > 80.638
    df['willr'] = talib.WILLR(high, low, series, timeperiod=14)  # > -11
    df['willr_d'] = discrete_trend(df['willr'])  # > -11
    df['cci'] = talib.CCI(high, low, series, timeperiod=14)  # > -11
    df['cci_d'] = continuous_to_discrete_w_bounds(df['cci'], -200, 200)  # > -11
    df['obv'] = talib.OBV(series, volume)
    df['mom'] = talib.MOM(series)
    df['mom_d'] = discrete_series_pos_neg(df['mom'])
    df['sma20'] = discrete_series_compare(close, talib.SMA(series, 20))  # > talib.MA(series, 200)
    df['sma50'] = discrete_series_compare(close, talib.SMA(series, 50))  # > talib.MA(series, 200)
    df['sma200'] = discrete_series_compare(close, talib.SMA(series, 200))  # > talib.MA(series, 200)
    df['wma10'] = discrete_series_compare(close, talib.WMA(series, 10))
    df['macd'], df['macd_sig'], macdhist = talib.MACD(series, fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_d'] = discrete_trend(df['macd'])# > -11
    df['stok'], df['stod'] = talib.STOCH(high, low, series, fastk_period=5, slowk_period=3,
                               slowk_matype=0, slowd_period=3, slowd_matype=0)
    df['stok_d'] = discrete_trend(df['stok'])# > -11
    df['stod_d'] = discrete_trend(df['stod'])# > -11
    #df['sto'] = slowk #> 80

    return df


def rate_of_change(arr, period=1):
    '''
    Calcuate the rate of change from n periods ago.  Seems Talib ROC is buggy.  TODO.
    '''
    prevPrice = arr.shift(period)
    today = arr

    #return (price-prevPrice)/prevPrice
    return (today - prevPrice)/prevPrice


def discrete_series_compare(series_a, series_b):
    '''
    Convert continuous range to discrete value:
    - +1 if series a is above series b.  -1 if below.

    eg, Simple moving average
    '''
    x = (series_a > series_b).astype(int)
    x[x == 0] = -1

    return x


def discrete_series_pos_neg(series):
    '''
    Convert continuous range to discrete value:
    - +1 if series value is above 0, -1 if below.

    eg, momentum indicator
    '''
    x = (series > 0).astype(int)
    x[x == 0] = -1

    return x


def discrete_trend(series):
    '''
    Convert continuous oscillator to discrete value:
    - +1 if oscillator is above, -1 if below.
    '''
    diff = np.diff(series)
    diff[diff > 0] = 1
    diff[diff <= 0] = -1

    return np.append(0, diff)


def continuous_to_discrete_w_bounds(series, lower_bound, upper_bound):
    '''
    Convert RSI to discrete value:
    - +1 if indicator is above upper_bound, -1 if below lower_bound,
      take the +/- diff for in between

    These setting are mean reverting
    '''
    diff = np.diff(series)
    diff[diff <= 0] = -1
    diff[diff > 0] = -2
    d = np.append(0, diff)

    d[(series < lower_bound)] = 1
    d[(series > upper_bound)] = -1

    d[d == -2] = 1

    return d


# def run_exp_smooth(self, df, **kwparams):
def run_exp_smooth(df, alpha):
    '''
    Exponential smoothing via pandas.
        http://pandas.pydata.org/pandas-docs/stable/computation.html
        deprecated: http://pandas-docs.github.io/pandas-docs-travis/computation.html#exponentially-weighted-windows

        One must specify precisely one of span, center of mass, half-life
        and alpha to the EW functions
    '''
    # df['exp_smooth'] = pd.ewma(df['close'].values, kwparams)
    df['exp_smooth_open'] = pd.ewma(df['open'].values, alpha)
    # df['exp_smooth_open'] = df['open'].ewm(span=20).mean()
    df['exp_smooth_high'] = pd.ewma(df['high'].values, alpha)
    df['exp_smooth_low'] = pd.ewma(df['low'].values, alpha)
    df['exp_smooth_close'] = pd.ewma(df['close'].values, alpha)
    df['exp_smooth_volume'] = pd.ewma(df['volume'].values, alpha)

    return df


def run_holt_winters_second_order_ewma(series, period, beta):
    '''
    Holt-Winters exponential smoothing
        http://connor-johnson.com/2014/02/01/smoothing-with-exponentially-weighted-moving-averages/

    input:  the series, the MA period, the beta param
    output: return the smoothed
    '''
    N = series.size
    alpha = 2.0 / (1 + period)
    s = np.zeros((N, ))
    b = np.zeros((N, ))
    s[0] = series[0]
    for i in range(1, N):
        s[i] = alpha * series[i] + (1 - alpha) * (s[i-1] + b[i-1])
        b[i] = beta * (s[i] - s[i-1]) + (1 - beta) * b[i-1]

    return s


def liregslope(series, period):
    '''
    Calculate a rolling window linear regression slope over n periods

    input
        series - pandas Series object
        period - lookback periods for rolling window
    output
        pandas Series with the rolling slope
    '''
    def get_slope(x):
        Y = x
        #Y = Y[-period:]  # subset range of Y
        X = range(1,len(Y)+1)
        X = sm.add_constant(X)
        model = sm.OLS(Y,X)
        results = model.fit()
        return results.params[-1]

    # rolling_slope = series.rolling(period).apply(lambda x: get_slope(x))
    rolling_slope = series.rolling(period).apply(get_slope)

    return rolling_slope


def velocity(in_list, period=20):
    '''
    Forecast the next day's movement as a function of slope divided by slope period

    input:  A list of slopes calculated over given period
    eg: close + (df['slope20'] * close) / 20
    '''
    series = close + (in_list * close) / period
    return series[-1]


def hurst(ts):
    '''
    Returns the Hurst Exponent of the time series vector ts

    eg, Call the function
        hurst(df['close'])

    https://www.quantstart.com/articles/Basics-of-Statistical-Mean-Reversion-Testing
    '''
    from numpy import cumsum, log, polyfit, sqrt, std, subtract
    from numpy.random import randn

    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    # Here it calculates the variances, but why it uses
    # standard deviation and then make a root of it?
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0

    # Example:
    # # Create a Gometric Brownian Motion, Mean-Reverting and Trending Series
    # gbm = log(np.cumsum(randn(100000))+1000)
    # mr = log(np.random.randn(100000)+1000)
    # tr = log(np.cumsum(randn(100000)+1)+1000)
    #
    # # Output the Hurst Exponent for each of the above series
    # # and the price of Google (the Adjusted Close price) for
    # # the ADF test given above in the article
    # print "Hurst(GBM):   %s" % hurst(gbm)
    # print "Hurst(MR):    %s" % hurst(mr)
    # print "Hurst(TR):    %s" % hurst(tr)
    #
    # # Assuming you have run the above code to obtain 'goog'!
    # print "Hurst(GOOG):  %s" % hurst(goog['Adj Close'])


def volatility(series, period):
    '''
    The moving historical standard deviation of the log returns, i.e. the moving
    historical volatility, might be more of interest:
        Also make use of pd.rolling_std(data, window=x) * math.sqrt(window) for
        the moving historical standard deviation of the log returns (aka the
        moving historical volatility).

    Input:
        series - pandas series
        period - num for rolling window
    '''
    # Daily returns
    series_roc = series.pct_change()

    # Replace NA values with 0
    #series_roc.fillna(0, inplace=True)

    # Inspect daily returns
    #print(series_roc)

    # Daily log returns
    series_log = np.log(series.pct_change()+1)

    # Calculate the volatility
    vol = series_log.rolling(period).std() * np.sqrt(period)

    # Plot the volatility
    vol.plot(figsize=(10, 8))


def SCTR(df, series_name):
    '''
    Calculate StockCharts SCTR scores

    Input
        df - a pandas datareader structured dataframe with multi-index of
             symbols, datetime
        series_name - The price series column name to use for the SCTR calc.
             Probably 'Close'.
    Output
        pabdas series - a 'SCTR' column to be added to the dataframe
    '''
    dff = df.copy()

    t1 = time.time()
    dff['SCTR_raw'] = SCTR_raw(dff[series_name])
    t2 = time.time()
    print "Time to process SCTR scores: " + str((t2 - t1)) + "\n"

    SCTRs = dff[['SCTR_raw']].reset_index().pivot('Date', 'Symbol', 'SCTR_raw')
    ranks = SCTR_ranks(SCTRs)

    sctr_series = ranks.stack().swaplevel().sort_index(level=0)
    return sctr_series

def SCTR_raw(series):
    '''
    Calculate the SCTR indiciator for one price series.

    stockcharts.com SCTR indicator
    - http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:sctr

    Input:
        series - pandas Series.  Should be the close.
    Output
        series of raw SCTR values.  Note, this isn't normalized.
                                    Use function SCTR_ranks for that.

    Calculation:

        Long-Term Indicators (weighting)

          * Percent above/below 200-day EMA (30%)
          * 125-Day Rate-of-Change (30%)

        Medium-Term Indicators (weighting)

          * Percent above/below 50-day EMA (15%)
          * 20-day Rate-of-Change (15%)

        Short-Term Indicators (weighting)

          * 3-day slope of PPO-Histogram (5%)
          * 14-day RSI (5%)

        http://www.stratasearch.com/forum/viewtopic.php?f=9&t=1198
    '''
    close = series
    ema200 = talib.EMA(close.values, 200)
    ema50 = talib.EMA(close.values, 50)
    ema26 = talib.EMA(close.values, 26)
    ema12 = talib.EMA(close.values, 12)
    roc125 = (close - close.shift(125)) / (close.shift(125)*1.)
    roc20 = (close - close.shift(20)) / (close.shift(20)*1.)
    ppo = pd.Series(talib.PPO(close.values, 12, 26), index=close.index)
    rsi = pd.Series(talib.RSI(close.values, 14), index=close.index)

    # Weights
    LT_EMA = (close - ema200) / (ema200*1.) * 100 * 0.3
    LT_ROC = roc125 * 100 * 0.3

    MT_EMA = (close - ema50) / (ema50*1.) * 100 * 0.15
    MT_ROC = roc20 * 100 * 0.15

    PPO_HIST = ppo - talib.EMA(ppo.values, 9)
    SLOPE = liregslope(PPO_HIST, 3)
    ST_PPO = (SLOPE >= 1) * 0.05 * 100   # Should be 5 points, not .05
    ST_RSI = rsi * 0.05

    SCTR = LT_EMA + LT_ROC + MT_EMA + MT_ROC + ST_PPO + ST_RSI

    # for debugging
    dct = {'SCTR':SCTR, 'close':close, 'slope':SLOPE,
           'LT_EMA':LT_EMA,'LT_ROC':LT_ROC,'MT_EMA':MT_EMA,
           'MT_ROC':MT_ROC,'ST_PPO':ST_PPO,'ST_RSI':ST_RSI}
    df = pd.DataFrame(dct, index=close.index)

    return SCTR


def SCTR_ranks(sctr_df):
    '''
    Calculate the SCTR ranks for a multiple symbols compared to each other.

    Sort the SCTRs, split into deciles, within each decile rank again
    equally spaced.

    Input:
        sqtr_df - Dataframe of symbols and SCTR ranks of the form:

                    AAPL    MSFT    TSLA
        date
        1/1/2000    12.3    23.4    6.78
        ...
    Output:
        sqtr_rank_df - padas dataframe of the same form, but with ranks rather
        than values.
    '''
    sctr_ranks_df = sctr_df.copy()

    # for each date...
    for i in range(0, sctr_ranks_df.shape[0]):
        # fill NAs with -1000 and rank the column values (now transposed as a series)
        sctr_ranks_df.iloc[i] = sctr_ranks_df.iloc[i].fillna(-1000).rank(pct=True) * 100

    return sctr_ranks_df
