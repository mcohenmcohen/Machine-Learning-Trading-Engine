'''
** Placeholder for a mean reverting trading systems
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
from minepy import MINE
from sklearn.feature_selection import RFE
import inspect
import talib
from trading_system import TradingSystem
from core import ModelUtils
import ta

class TradingSystem_MeanReverting(TradingSystem):
    '''
    Parent class for trading systems
    '''
    def __init__(self):
        self.df = None      # X matrix features
        self.target = None  # y label target
        self.features = []  # X feature column names
        TradingSystem.__init__(self)

    def preprocess_data(self, data):
        '''
        Perform any data preprocessing steps such as normalizing, smoothing,
        remove correlated columns, etc
        '''
        df = data.copy()


        #df = TA.run_techicals(df)

        # normal and smoothed Price series
        opn = df['open']
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume'].astype(float)

        df['stdClose20'] = close.rolling(window=20).std()

        # Mean revert as a pairs trade by lin reg stock A with stock B
        # http://pandas.pydata.org/pandas-docs/version/0.10.0/computation.html
        # Also winsorization
        # another good ref - quant start_time
        # https://www.quantstart.com/articles/Backtesting-An-Intraday-Mean-Reversion-Pairs-Strategy-Between-SPY-And-IWM
