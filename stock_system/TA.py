###################################################################################################
# Technical Analysis
#
# Utilities to calculate technical analysis indicators and related data manipulation
###################################################################################################

import pandas as pd
import numpy as np
import talib
# https://github.com/mrjbq7/ta-lib
# documentation: https://github.com/mrjbq7/ta-lib/blob/master/docs/func.md


################################
# Technical Analysis functions
################################
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
    df['roc'] = rate_of_change(df['close'], 1)# > 0.04
    df['roc_d'] = discrete_series_pos_neg(df['roc'])# > 80.638
    df['rsi'] = talib.RSI(series, timeperiod=14)# > 80.638
    df['rsi_d'] = continuous_to_discrete_w_bounds(df['rsi'], 30,70)# > 80.638
    df['willr'] = talib.WILLR(high, low, series, timeperiod=14)# > -11
    df['willr_d'] = discrete_trend(df['willr'])# > -11
    df['cci'] = talib.CCI(high, low, series, timeperiod=14)# > -11
    df['cci_d'] = continuous_to_discrete_w_bounds(df['cci'], -200, 200)# > -11
    df['obv'] = talib.OBV(series, volume)
    df['mom'] = talib.MOM(series)
    df['mom_d'] = discrete_series_pos_neg(df['mom'])
    df['sma20'] = discrete_series_compare(close, talib.SMA(series, 20)) #> talib.MA(series, 200)
    df['sma50'] = discrete_series_compare(close, talib.SMA(series, 50)) #> talib.MA(series, 200)
    df['sma200'] = discrete_series_compare(close, talib.SMA(series, 200)) #> talib.MA(series, 200)
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
    prevPrice = np.roll(arr, period)
    today = arr

    #return (price-prevPrice)/prevPrice
    return (today-prevPrice)/prevPrice


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
