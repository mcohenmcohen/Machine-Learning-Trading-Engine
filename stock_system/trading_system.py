'''
Parent class for trading systems
'''
# Author:  Matt Cohen
# Python Version 2.7

import pandas as pd


class TradingSystem(object):
    '''
    Parent class for trading systems
    '''
    def __init__(self):
        self.name = ''  # An optional name
        self.excluded_features = ['symbol', 'open', 'high', 'low', 'close']

    def preprocess(self):
        '''
        Perform any data preprocessing steps such as normalizing, smoothing,
        remove correlated columns, imputing data, etc
        '''
        pass

    def get_features(self):
        '''
        Return the features used by the trading system and implementing model.
        Example features are a set of technical indicators
        '''
        pass

    def set_features(self, features):
        '''
        Set the features for the tradng system to be used by the model

        input:  list of features, column names
        '''
        self.features = features

    def set_features_from_file(self, model_name, num_features_to_use=10):
        '''

        Set the features to be used by the tradng system, pulled from a file
        that stores an ordered list of features, which was generated previously
        by a feature engineering function
        '''
        features_df = pd.read_csv('_data/_FeatureEngineering.csv', index_col=0)
        self.features = features_df[model_name][0:num_features_to_use].tolist()

    def generate_target(self):
        '''
        Trading system goes here.
        This runs the trading system on the training data to generate the
        y label.

        Returns a dataframe with the y label column, ready to use in a model
        for fit and predict.
        '''
        pass
