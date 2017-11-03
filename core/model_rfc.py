'''
Instantiate a modeler to hold a random foreset classifier
with data set from trading system features
'''
# Author:  Matt Cohen
# Python Version 2.7

from stock_system.model import ModelUtils


class Model_RFC(ModelUtils):

    def __init__(self):
        ModelUtils.__init__(self)
        self.model_name = 'rfc'
        self.hyperparams = {}
        self.set_model('rfc')
        self.features = []
        self.X_train = None
        self.X_test = None
        self.y_test = None
        self.y_train = None

    def split(self, in_df):
        df = in_df.copy()

        y = df.pop('y_true').values
        self.features = df.columns.tolist()
        X = df.values

        X_train, X_test, y_train, y_test = self.simple_data_split(X, y,
                                           test_set_size=int(df.shape[0]*.2))
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train
