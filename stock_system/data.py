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
import os
from stock_system.iqfeed import IQFeed


class DataUtils(object):
    '''
    This class provides database services for stock symbol data.
    The database is intended to be postgres.

    init instantiates the database connection.
    get_data_quandl collects data from Quandl.
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
        '''
        Write to postgres database via psycopg2 engine

        Input:
            Dataframe with cols:  sym, data, O, H, L, C, V
        Return:
            None
        '''
        print '\nWriting Symbols to database'
        import time; t1 = time.time()

        import io, StringIO
        f = StringIO.StringIO()  # python 2
        #f = io.StringIO()  #  python 3
        df_copy = df.reset_index()
        df_copy.to_csv(f, index=False, header=False)  # removed header
        f.seek(0)  # move position to beginning of file before reading
        cursor = self.conn.cursor()
        try:
            cursor.copy_from(f, 'symbols', columns=tuple(df_copy.columns), sep=',')
        except pg2.IntegrityError as err:
            for arg in err.args:
                print arg
            print 'Rolling back.'
            self.conn.rollback()
        else:
            self.conn.commit()
            t2 = time.time(); print "- Database write time: " + str((t2 - t1)) + "\n"
        cursor.close()

        print 'Done.'

    def read_symbol_data(self, symbol, period='D'):
        '''
        Extract symbol time series data from postgres by period
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
    def get_data_iqfeed(self, symbols, start_date, end_date='', dbwrite=False):
        '''
        Get data via pandas_datareader

        input:
            symbols - list of string symbols
            start_date - format yyyymmdd
            end_date - format yyyymmdd
            dbwrite - write to the database.  Default is False.
        return:
            dataframe of data, indexes by symbol and datetime

        e.g.:
            df = pdr.get_data_iqfeed('AAPL', start=20140101)
            df.head(1)
                                          open    high     low   close  volume
            symbol date
            AAPL   2014-01-02 09:31:00  555.68  556.48  555.07  556.11  453045
        '''
        dir_path = './_data/iqf_symbols'
        syms = symbols
        #syms = ['SPY']

        # Empty df to aggregate all downloaded sysmbols into
        df_total = pd.DataFrame(columns=['symbol','date','open','high','low','close','volume'])
        df_total.set_index(['symbol', 'date'], inplace=True)

        print 'Downloading symbols from DTN IQFeed'
        feed = IQFeed()
        # Define server host, port and symbols to download
        host = "127.0.0.1"  # Localhost
        port = 9100  # Historical data socket port

        import time; t1 = time.time()
        for sym in syms:
            print sym
            # Construct the message needed by IQFeed to retrieve data
            message = "HIT,%s,60,%s 075000,,,093000,160000,1\n" % (sym, start_date)

            data = feed.get_data(message)

            data = "".join(data.split("\r"))
            data = data.replace(",\n", "\n")[:-1]

            # write the data as .csv file(s) to disk
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            f = open("%s/%s.csv" % (dir_path, sym), "w")
            f.write(data)
            f.close()

            # read csv to dataframe
            csv = '%s/%s.csv' % (dir_path, sym)
            cols = ['date','high','low','open','close','volume','oi']
            df = pd.read_csv(csv, header=None, names=cols)#, index_col=0, infer_datetime_format=True)
            df['symbol'] = sym
            del df['oi']
            df = df[['symbol','date','open','high','low','close','volume']]
            df.set_index(['symbol', 'date'], inplace=True)

            df_total = pd.concat([df_total, df])

        t2 = time.time();
        print '- Download time: %s' % str((t2 - t1))
        if dbwrite:
            self.write_symbol_data(df_total)

        return df_total

    ##################################
    # get data from pandas_datareader
    ##################################
    def get_data_pdr_yahoo(self, symbols, start_date, end_date='', dbwrite=False):
        '''
        Get data via pandas_datareader

        input:
            symbols - list of string symbols
            start_date - string 'yyyy-mm-dd'
            end_date - string 'yyyy-mm-dd'
            dbwrite - write to the database.  Default is False.
        return:
            dataframe of daily data, multiindexed by symbol

        e.g.:
            df = dbutils.get_data_pdr_yahoo('AAPL', start_date=20060101)
            df = dbutils.get_data_pdr_yahoo('AAPL', start_date=20060101, end_date=20060101)
            df.head(1)
                                          open    high     low   close  volume
            symbol date
            AAPL   2014-01-02 09:31:00  555.68  556.48  555.07  556.11  453045
        '''
        if type(symbols) != list:
            err = '\'%s\' is a %s.  Symbols must be supplied in a list object.' % (symbols, symbols.__class__.__name__)
            raise ValueError(err)

        symbols = [s.upper() for s in symbols]
        print 'Symbols:\n', ','.join(symbols)

        import fix_yahoo_finance as yf
        yf.pdr_override()

        print 'Downloading symbols from yahoo via pandas datareader '
        sym_df = None
        if end_date:
            sym_df = pdr.get_data_yahoo(symbols, start=start_date, end=end_date)
        else:
            sym_df = pdr.get_data_yahoo(symbols, start=start_date)

        if 0 in sym_df.shape:
            print '\nNo symbols downloaded. ' \
                  'There is a bug in pdr, try downloading more than one symbols. ' \
                  'Or, there may be an issue with yahoo, try again in a few minutes.'
            return
        df = sym_df
        if sym_df.__class__.__name__ == 'Panel':
            df = sym_df.to_frame()
        elif sym_df.__class__.__name__ == 'DataFrame':
            df['symbol'] = symbols[0]
            df.set_index(['symbol'], append=True, inplace=True)

        # add 4pm hour to date for close time
        df.index.names = ['date','symbol']
        df.reset_index(inplace=True)
        #df['date'] = df.index.get_level_values('date') + pd.to_timedelta(16, unit='h')
        df['date'] = df['date'] + pd.to_timedelta(16, unit='h')
        df.set_index(['symbol','date'], inplace=True)

        # Swap date and symbol for major/minor index
        df = df.swaplevel(0, 1, axis=0)
        df = df.sort_index(level='symbol')
        # set columns to lowercase
        df.columns = map(str.lower, df.columns)
        # set close as adj close
        del df['close']
        df.rename(columns={'adj close': 'close'}, inplace=True)

        s2 = set(df.index.get_level_values('symbol').unique().tolist())
        diff = [i for i in symbols if not i in s2]
        if len(diff) > 0:
            print '\nThe following symbols were not able to be downloaded:'
            print ', '.join(diff)

        if dbwrite:
            self.write_symbol_data(df)

        return df

    ##################################
    # get data from pandas.io.web
    ##################################
    # def get_data_pweb(self, symbol, start, end, source='yahoo'):
    #     '''
    #     Get data via pandas web io
    #     ** DEPRECATED.  Use pandas datareader
    #
    #     input:
    #         symbol - string
    #         start_date - datetime.datetime
    #         end_date - datetime.datetime
    #     output:
    #         dataframe of daily data
    #
    #     eg,
    #         import datetime
    #         import pandas.io.data as web
    #
    #         AAPL = web.DataReader('AAPL', data_source='google', start='1/1/2010', end='1/1/2016')
    #         # reads data from Google Finance
    #         AAPL['42d'] = pd.rolling_mean(AAPL['Close'], 42)
    #         AAPL['252d'] = pd.rolling_mean(AAPL['Close'], 252)
    #         # 42d and 252d trends
    #     '''
    #     import pandas.io.data as web
    #
    #     sym_data = web.DataReader(symbol, data_source=source, start=start, end=end)
    #     return sym_data


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
