import numpy as np
import pandas as pd
import psycopg2 as pg2
from sqlalchemy import create_engine
import quandl

# https://github.com/mrjbq7/ta-lib
# documentation: https://github.com/mrjbq7/ta-lib/blob/master/docs/func.md
import talib


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

    def read_symbol_data(self, symbol, query, period='D'):
        '''
        Extract symbol time series data by period
        Valid time periods: minute, 15 min, hour, daily, weekly

        Input:   Symbol to query, time series period
        Output:  Dataframe of the period time series
        '''
        period = str(period).upper()

        prefix = "select * from symbols where symbol = '%s' and " % symbol
        # No where clause for minutes.  Default is minutes
        if period == '15':
            where = "EXTRACT(MINUTE FROM date) = 0 \
                     or EXTRACT(MINUTE FROM date) = 15 \
                     or EXTRACT(MINUTE FROM date) = 30 \
                     or EXTRACT(MINUTE FROM date) = 45"
        if period == 'H':
            where = "EXTRACT(MINUTE FROM date) = 0"
        if period == 'D':
            where = "EXTRACT(HOUR FROM date) = 16 and EXTRACT(MINUTE FROM date) = 0"
        if period == 'W':
            where = "EXTRACT(HOUR FROM date) = 16 and EXTRACT(MINUTE FROM date) = 0 \
                     and EXTRACT(DOW FROM date) = 5"

        query = prefix + where
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
        # open  = df['open']
        high = df['exp_smooth_high'].values
        low = df['exp_smooth_low'].values
        close = df['exp_smooth_close'].values
        volume = df['exp_smooth_volume'].values.astype(float)

        df['roc'] = talib.ROCP(close, timeperiod=10)
        df['rsi'] = talib.RSI(close, timeperiod=14)
        df['willr'] = talib.WILLR(high, low, close, timeperiod=14)
        df['obv'] = talib.OBV(close, volume)

        macd, macdsignal, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd

        slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3,
                                   slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['sto'] = slowk

        return df
