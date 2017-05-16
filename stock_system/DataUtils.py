###################################################################################################
# Class for accessing data
###################################################################################################

import numpy as np
import pandas as pd
import psycopg2 as pg2
from sqlalchemy import create_engine
import quandl

# select * from symbols where date > timestamp '2017-05-01 00:00:00' and EXTRACT(HOUR FROM date) = 16 and EXTRACT(MINUTE FROM date) = 0;

class DataUtils(object):
    '''
    This class provides database services for stock symbol data

    init instantiates the database connection
    get_data_quandl collects data from Quandl
    '''
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
    def get_data_quandl(self, symbol):
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
