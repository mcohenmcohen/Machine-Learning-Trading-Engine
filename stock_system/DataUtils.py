import numpy as np
import pandas as pd
import psycopg2 as pg2
from sqlalchemy import create_engine
import quandl

# https://github.com/mrjbq7/ta-lib
# documentation: https://github.com/mrjbq7/ta-lib/blob/master/docs/func.md
import talib

# select * from symbols where date > timestamp '2017-05-01 00:00:00' and EXTRACT(HOUR FROM date) = 16 and EXTRACT(MINUTE FROM date) = 0;

##################################
# Class for accessing data
##################################
class DataUtils(object):

    def __init__(self):
        self.conn = pg2.connect(dbname='stocksdb', user='mcohen', host='localhost')
        self.c = self.conn.cursor()
        self.engine = create_engine('postgresql://localhost:5432/stocksdb')

    ##################################
    # Database access
    ##################################

    def _test(self):
        self.c.execute("""SELECT count(*) from symbols""")
        rows = self.c.fetchall()
        print "\nShow me:\n"
        for row in rows:
            print "   ", row[0]

    def write_symbol_data(self, df):
        df.to_sql('symbols', self.engine, if_exists='append', index=False)

    def read_symbol_data(self, symbol, period='M'):
        '''
        Extract symbol time series data by period
        Valid time periods: minute, 15 min, hour, daily, weekly

        Input:   Symbol to query, time series period
        Output:  Dataframe of the period time series
        '''
        period = str(period).upper()

        prefix = "select * from symbols where symbol = '%s'" % symbol
        # No where clause for minutes.  Default is minutes
        where = ''
        if period == '15':
            where = " and EXTRACT(MINUTE FROM date) = 0 \
                     or EXTRACT(MINUTE FROM date) = 15 \
                     or EXTRACT(MINUTE FROM date) = 30 \
                     or EXTRACT(MINUTE FROM date) = 45"
        if period == 'H':
            where = " and EXTRACT(MINUTE FROM date) = 0"
        if period == 'D':
            where = " and EXTRACT(HOUR FROM date) = 16 and EXTRACT(MINUTE FROM date) = 0"
        if period == 'W':
            where = " and EXTRACT(HOUR FROM date) = 16 and EXTRACT(MINUTE FROM date) = 0 \
                     and EXTRACT(DOW FROM date) = 5"

        # if where not '':
        #     where = where

        query = prefix + where + ' order by date'
        return pd.read_sql(query, self.engine)

    ##################################
    # get data from Quandl
    ##################################
    def get_data_q(self, symbol):
        '''
        Get data from quandl

        input:  symbol
        output: dataframe of daily data
        '''
        sym = 'WIKI/%s', symbol
        return quandl.get(sym)

    ##################################
    # Math / Algos
    ##################################

    # def run_exp_smooth(self, df, **kwparams):
    def run_exp_smooth(self, df, alpha):
        '''
        Exponential smoothing via pandas.
            http://pandas-docs.github.io/pandas-docs-travis/computation.html#exponentially-weighted-windows

            One must specify precisely one of span, center of mass, half-life
            and alpha to the EW functions
        '''
        # df['exp_smooth'] = pd.ewma(df['close'].values, kwparams)
        df['exp_smooth_open'] = pd.ewma(df['open'].values, alpha)
        df['exp_smooth_high'] = pd.ewma(df['high'].values, alpha)
        df['exp_smooth_low'] = pd.ewma(df['low'].values, alpha)
        df['exp_smooth_close'] = pd.ewma(df['close'].values, alpha)
        df['exp_smooth_volume'] = pd.ewma(df['volume'].values, alpha)
        return df

    def run_holt_winters_second_order_ewma(self, series, period, beta):
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

    ##################################
    # Technical Analysis
    ##################################

    def run_techicals(self, df):
        '''
        talib
        - docs: https://github.com/mrjbq7/ta-lib/blob/master/docs/func_groups/momentum_indicators.md
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
        df['roc'] = self.rate_of_change(series, 1)# > 0.04
        df['roc_d'] = self.discrete_series_pos_neg(df['roc'])# > 80.638
        df['rsi'] = talib.RSI(series, timeperiod=14)# > 80.638
        df['rsi_d'] = self.continuous_to_discrete_w_bounds(df['rsi'], 30,70)# > 80.638
        df['willr'] = talib.WILLR(high, low, series, timeperiod=14)# > -11
        df['willr_d'] = self.discrete_trend(df['willr'])# > -11
        df['cci'] = talib.CCI(high, low, series, timeperiod=14)# > -11
        df['cci_d'] = self.continuous_to_discrete_w_bounds(df['cci'], -200, 200)# > -11
        df['obv'] = talib.OBV(series, volume)
        df['mom'] = talib.MOM(series)
        df['mom_d'] = self.discrete_series_pos_neg(df['mom'])
        df['sma20'] = self.discrete_series_compare(close, talib.SMA(series, 20)) #> talib.MA(series, 200)
        df['sma50'] = self.discrete_series_compare(close, talib.SMA(series, 50)) #> talib.MA(series, 200)
        df['sma200'] = self.discrete_series_compare(close, talib.SMA(series, 200)) #> talib.MA(series, 200)
        df['wma10'] = self.discrete_series_compare(close, talib.WMA(series, 10))
        df['macd'], df['macd_sig'], macdhist = talib.MACD(series, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd_d'] = self.discrete_trend(df['macd'])# > -11
        df['stok'], df['stod'] = talib.STOCH(high, low, series, fastk_period=5, slowk_period=3,
                                   slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['stok_d'] = self.discrete_trend(df['stok'])# > -11
        df['stod_d'] = self.discrete_trend(df['stod'])# > -11
        #df['sto'] = slowk #> 80

        return df

    ################################
    # Technical Analysis functions
    ################################

    def rate_of_change(self, arr, period=1):
        '''
        Calcuate the rate of change from n periods ago.  Seems Talib ROC is buggy.  TODO.
        '''
        prevPrice = np.roll(arr, period)
        today = arr

        #return (price-prevPrice)/prevPrice
        return (today-prevPrice)/prevPrice

    def discrete_series_compare(self, series_a, series_b):
        '''
        Convert continuous range to discrete value:
        - +1 if series a is above series b.  -1 if below.

        eg, Simple moving average
        '''
        x = (series_a > series_b).astype(int)
        x[x == 0] = -1
        return x

    def discrete_series_pos_neg(self, series):
        '''
        Convert continuous range to discrete value:
        - +1 if series value is above 0, -1 if below.

        eg, momentum indicator
        '''
        x = (series > 0).astype(int)
        x[x == 0] = -1
        return x


    def discrete_trend(self, series):
        '''
        Convert continuous oscillator to discrete value:
        - +1 if oscillator is above, -1 if below.
        '''
        diff = np.diff(series)
        diff[diff > 0] = 1
        diff[diff <= 0] = -1
        return np.append(0, diff)

    def continuous_to_discrete_w_bounds(self, series, lower_bound, upper_bound):
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
