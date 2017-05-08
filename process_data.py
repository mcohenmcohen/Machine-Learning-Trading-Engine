import pandas as pd
import numpy as np
from stock_system import IQFeed, DBUtils
import timeit

# cols = ['DATE','OPEN','LOW','HIGH','CLOSE','VOLUME','OPEN INTEREST']
# df = pd.read_csv('AAPL.csv', names=cols)


if __name__ == "__main__":

    feed = IQFeed.DataFeed()
    db = DBUtils.DBUtils()

    symbols = ['FB','SPY','QQQ']
    start_date = '20170101'

    for sym in symbols:
        print 'Processing %s' % sym
        start_time = timeit.default_timer()
        df = feed.get_data_hist(sym, start_date)

        retrieve_time = timeit.default_timer() - start_time
        print '- time to retrieve: ', retrieve_time

        start_time = timeit.default_timer()
        db.write_symbol_data(df)
        print '- time to db write: ', timeit.default_timer() - start_time

    print 'Done.'
