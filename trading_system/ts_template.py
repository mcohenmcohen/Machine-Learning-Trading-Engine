'''
** Templace for trading systems class
'''
# Author:  Matt Cohen
# Python Version 2.7

from stock_system.TradingSystem import TradingSystem


class TradingSystem_Comp(TradingSystem):
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

        self.df = df

        return self.df

    def get_features(self):
        return self.features

    def feature_forensics(self, model):
        return TradingSystem.feature_forensics(self, model)
