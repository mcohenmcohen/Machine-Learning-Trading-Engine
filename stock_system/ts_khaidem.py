'''
Trading System to replicate study
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
from stock_system.trading_system import TradingSystem
from stock_system import ta


class TradingSystem_Khaidem(TradingSystem):
    '''
    Parent class for trading systems
    '''
    def __init__(self):
        self.df = None      # X matrix features
        self.target = None  # y label target
        self.features = []  # X feature column names
        TradingSystem.__init__(self)

    def preprocess_data(self, df):
        '''
        Perform any data preprocessing steps such as normalizing, smoothing,
        remove correlated columns, etc
        '''
        df = ta.run_exp_smooth(df, alpha=.5)
        df = ta.run_techicals(df)

        # Impute - delete rows with Nan and null, will be the first several rows
        for name in df.columns:
            df = df[df[name].notnull()]

        self.df = df

        # Now that we've calculated the features above,
        # save off the names to a list
        self.features = [col for col in df.columns
                         if col not in self.excluded_features]
        self._generate_target()


        return self.df

    def get_features(self):
        self.features = ['roc', 'rsi', 'willr', 'obv', 'stok']
        # Oscilators
        # x_osc = ['rsi', 'cci', 'stod', 'stok', 'willr']
        # x_oscd_cols = ['rsi_d', 'cci_d', 'stod_d', 'stok_d', 'willr_d']
        # # MAs
        # x_ma_cols = ['sma20', 'sma50', 'sma200', 'wma10', 'macd_d']
        # x_all_dscrete_cols = ['roc_d', 'rsi_d', 'cci_d', 'stod_d', 'stok_d', 'willr_d', 'mom_d']
        # #x_cols = ['roc', 'rsi', 'willr', 'obv', 'stok']#'mom', , 'cci',  'stod', 'macd', 'sma', 'sma50', 'wma']
        # #x_cols = ['roc']
        # x_cols = x_all_dscrete_cols + x_ma_cols
        return self.features

    def feature_forensics(self, model):
        return TradingSystem.feature_forensics(self, model)

    def _generate_target(self):
        '''
        Trading system goes here.
        This runs the trading system on the training data to generate the y label.

        Returns a dataframe with the y label column, ready to use in a model for fit and predict.
        '''
        if self.df is None:
            print 'This trading system has no data.  Call preprocess_data first.'
            return

        # Number of days ahead to see if the price moved up or down
        days_ahead = -1
        self.df['gain_loss'] = np.roll(self.df['close'], days_ahead) - self.df['close']
        self.df['y_true'] = (self.df['gain_loss'] >= 0).astype(int)

        # Drop the last row becaue of the shift by 1 - it puts the first to the last
        # Probably needs to change
        self.df = self.df[:-1]

        return self.df
