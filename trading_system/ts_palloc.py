'''
Trading System - Portfolio Allocation
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
from sklearn.preprocessing import Imputer
from trading_system import TradingSystem
from trading_system import ta
import talib
import tensorflow


class TradingSystem_RNN(TradingSystem):

    def __init__(self):
        self.df = None      # X matrix features
        self.target = None  # y label target
        self.features = []  # X feature column names
        TradingSystem.__init__(self)

    def preprocess(self, data):
        df = data.copy()

        ret = lambda x,y: np.log(y/x) #Log return
        zscore = lambda x:(x -x.mean())/x.std() # zscore
        df['log_perc_return'] = ret(df.close/df.close.shift(1))

        period = 20
        df['slope'] = ta.liregslope(df['close'], period)
        df['velocity'] = df['close'] + (df['slope20'] * df['close']) / period
        df['stdClose'] = df['close'].shift(1).rolling(window=period).std()
        df['zscore'] = (df['close'] - df['close'].shift(1).rolling(window=period).mean()) / df['stdClose']

        # Get all the tickers
        # - by SCTR, for one.  Maybe other criteria
        # Get correlated tickers, remove them?
        # TODO - fill this dataframe
        corr = df_top_SCTR.corr()

        self.features = x column names
        self.target = y, such as the index number of the SCTR, or
                        if the next day a stock goes up
