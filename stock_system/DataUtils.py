###################################################################################################
# Class for accessing data from a database or other datasource, and other processing of data
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
        df = pd.read_sql(query, self.engine)
        df.set_index('date', inplace=True)  # Set date as the index
        return df

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
    # get data from the web via data reader
    ##################################
    def get_symbols_dr():
        '''
        http://stackoverflow.com/questions/22991567/pandas-yahoo-finance-datareader
        '''
        #TODO Experiment with this source and api
