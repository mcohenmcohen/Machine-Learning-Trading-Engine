import pandas as pd
import numpy as np
from stock_system import IQFeed, DataUtils
import timeit
from StringIO import StringIO  # For python 2.  For python 3 import from io

# cols = ['DATE','OPEN','LOW','HIGH','CLOSE','VOLUME','OPEN INTEREST']
# df = pd.read_csv('AAPL.csv', names=cols)


if __name__ == "__main__":

    feed = IQFeed.DataFeed()
    db = DataUtils.DataUtils()

    symbols = ['SPY']
    start_date = '20030101'

    for sym in symbols:
        print 'Processing %s' % sym
        start_time = timeit.default_timer()
        message = "HIT,%s,60,%s 075000,,,093000,160000,1\n" % (sym, start_date)

        data = feed.get_data(message)

        # Remove all the endlines and line-ending
        # comma delimiter from each record
        data = "".join(data.split("\r"))
        data = data.replace(",\n", "\n")[:-1]

        cols = ['date', 'open', 'low', 'high', 'close', 'volume', 'open_interest']
        df = pd.read_csv(StringIO(data), names=cols)
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
        df.insert(0, 'symbol', sym)

        retrieve_time = timeit.default_timer() - start_time
        print '- time to retrieve: ', retrieve_time
# Date format from IQfeed: 2007-04-24 09:31:00
# Query that works: select * from symbols where symbol='FB' and date <= '2017-04-03'::date order by date DESC limit 10;
        start_time = timeit.default_timer()
        db.write_symbol_data(df)
        print '- time to db write: ', timeit.default_timer() - start_time

    print 'Done.'
