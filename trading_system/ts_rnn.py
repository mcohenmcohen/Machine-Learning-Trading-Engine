'''
Trading System RNN
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
from sklearn.preprocessing import Imputer
from stock_system.trading_system import TradingSystem
from stock_system import ta
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

        # high = df['high'].ewm(com=.8).mean()
        # low = df['low'].ewm(com=.8).mean()
        # close = df['close'].ewm(com=.8).mean()
        # volume = df['volume'].astype(float)

        # Use log data, make a zscore
        ret = lambda x,y: np.log(y/x) #Log return
        zscore = lambda x:(x -x.mean())/x.std() # zscore
        df['log_perc_return'] = ret(df.close/df.close.shift(1))

        # Period widown for analysis
        period = 20
        df['slope'] = ta.liregslope(df['close'], period)
        df['velocity'] = df['close'] + (df['slope20'] * df['close']) / period
        df['stdClose'] = df['close'].shift(1).rolling(window=period).std()
        df['zscore'] = (df['close'] - df['close'].shift(1).rolling(window=period).mean()) / df['stdClose']

        pass
