'''
DataUtils class
- Provides database access, alternatively another datasource
- shell functions for data from Quandl and elsewhere on the web
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
import pandas as pd
import psycopg2 as pg2
from sqlalchemy import create_engine
import quandl
import datetime
from pandas_datareader import data as pdr
import fix_yahoo_finance


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

        # Query format:
        # select * from symbols where date > timestamp '2017-05-01 00:00:00' and \
        #   EXTRACT(HOUR FROM date) = 16 and EXTRACT(MINUTE FROM date) = 0;
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

        query = prefix + where + ' order by date'
        df = pd.read_sql(query, self.engine)
        df.set_index('date', inplace=True)  # Set date as the index
        return df

    ##################################
    # get data from pandas_datareader
    ##################################
    def get_data_pdr_yahoo(self, symbols, start_date, end_date=''):
        '''
        Get data via pandas_datareader

        input:
            symbol - string
            start_date - datetime.datetime
            end_date - datetime.datetime
        output:
            dataframe of daily data

        eg,
            import datetime
            from pandas_datareader import data as pdr
            import fix_yahoo_finance

            aapl = pdr.get_data_yahoo('AAPL',
                                      start=datetime.datetime(2006, 10, 1),
                                      end=datetime.datetime(2012, 1, 1))
            aapl = pdr.get_data_yahoo('AAPL',
                                      start=datetime.datetime(2006, 10, 1))
            aapl.tail()
        '''
        def data(symbol):
            if end_date:
                sym_df = pdr.get_data_yahoo(symbol, start=start_date, end=end_date)
            else:
                sym_df = pdr.get_data_yahoo(symbol, start=start_date)
            return sym_df

        datas = map (data, symbols)
        return(pd.concat(datas, keys=symbols, names=['Symbol', 'Date']))

    ##################################
    # get data from pandas.io.web
    ##################################
    def get_data_pweb(self, symbol, source='yahoo', start, end):
        '''
        Get data via pandas web io

        input:
            symbol - string
            start_date - datetime.datetime
            end_date - datetime.datetime
        output:
            dataframe of daily data

        eg,
            import datetime
            import pandas.io.data as web

            AAPL = web.DataReader('AAPL', data_source='google', start='1/1/2010', end='1/1/2016')
            # reads data from Google Finance
            AAPL['42d'] = pd.rolling_mean(AAPL['Close'], 42)
            AAPL['252d'] = pd.rolling_mean(AAPL['Close'], 252)
            # 42d and 252d trends
        '''
        import pandas.io.data as web

        sym_data = web.DataReader(symbol, data_source=source, start=start, end=end)
        return sym_data


    ##################################
    # get data from Quandl
    ##################################
    def get_data_quandl(self, symbol):
        '''
        Get data from quandl

        input:  symbol
        output: dataframe of daily data

        eg,
            import quandl
            aapl = quandl.get("WIKI/AAPL", start_date="2006-10-01", end_date="2012-01-01")
            aapl.head()
        '''
        sym = 'WIKI/%s', symbol
        return quandl.get(sym)
