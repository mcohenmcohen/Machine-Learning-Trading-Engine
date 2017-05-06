import pandas as pd
import numpy as np
from stock_system import IQFeed, DBUtils
import timeit

# cols = ['DATE','OPEN','LOW','HIGH','CLOSE','VOLUME','OPEN INTEREST']
# df = pd.read_csv('AAPL.csv', names=cols)


if __name__ == "__main__":

    feed = IQFeed.DataFeed()
    db = DBUtils.DBUtils()

    start_time = timeit.default_timer()
    df = feed.get_data('FB')
    print(timeit.default_timer() - start_time)

    start_time = timeit.default_timer()
    db.write_symbol_data(df)
    print(timeit.default_timer() - start_time)
