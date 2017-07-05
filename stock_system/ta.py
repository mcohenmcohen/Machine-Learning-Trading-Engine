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


# def get_lin_reg_slope(df, period):
#     import statsmodels.api as sm
#
#     Y = df[-period:].close.values
#     X = range(1, period+1)
#     # dates = df.index.to_julian_date().values[-period:, None]
#     # X = np.concatenate([np.ones_like(dates), dates], axis=1)
#     X = sm.add_constant(X)
#     model = sm.OLS(Y, X)
#     results = model.fit()
#     intercept = results.params[0]
#     slope = results.params[1]
#
#     return (intercept, slope)
def slope_calc(in_list):
    cnstnt = sm.add_constant(range(-len(in_list) + 1, 1))
    return sm.OLS(in_list, cnstnt).fit().params[-1]  # slope

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
