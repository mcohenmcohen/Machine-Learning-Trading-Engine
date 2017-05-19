###################################################################################################
# Parent class for trading systems
###################################################################################################
import numpy as np
from minepy import MINE
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import inspect
from stock_system import TA, ModelUtils


class TradingSystem(object):
    '''
    Parent class for a trading systems
    '''
    def __init__(self):
        self.name = ''  # An optional name
        self.features = []

    def preprocess_data(self):
        '''
        Perform any data preprocessing steps such as normalizing, smoothing,
        remove correlated columns, imputing data, etc
        '''
        pass

    def get_features(self):
        '''
        Return the features used by the trading system and implementing model.
        '''
        pass

    def generate_target(self):
        '''
        Trading system goes here.
        This runs the trading system on the training data to generate the y label.

        Returns a dataframe with the y label column, ready to use in a model for fit and predict.
        '''
        pass

    def feature_forensics(self, model, rfe_num_feat):
        print '====== Identify to Remove highly correlated variables ======'
        self.check_corr(self.get_features())
        print '====== Feature selection via Maximal Information Coefficient (MIC) ======'
        self.check_mic()
        print '====== Recursive Feature Extraction ======'
        self.check_rfe(model, rfe_num_feat)

    def check_corr(self, feature_set):
        '''
        Get/print a correlation matrix to assist in identifying correlated columns
        '''
        # df = self.df.select_dtypes(['number'])  # Use only numeric columns
        df = self.df.copy()
        #import pdb; pdb.set_trace()
        df = df[feature_set]  # Use only sub set features
        print("Correlation Matrix")
        print(df.corr())
        print()

        def get_redundant_pairs(df):
            '''
            Get/print diagonal and lower triangular pairs of correlation matrix
            '''
            pairs_to_drop = set()
            cols = df.columns
            for i in range(0, df.shape[1]):
                for j in range(0, i+1):
                    pairs_to_drop.add((cols[i], cols[j]))
            return pairs_to_drop

        def get_top_abs_correlations(df, n=5):
            au_corr = df.corr().abs().unstack()
            labels_to_drop = get_redundant_pairs(df)
            au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
            return au_corr[0:n]

        print("Top Absolute Correlations")
        print(get_top_abs_correlations(df, 100))

        return (df.corr(), get_top_abs_correlations(df, 100))


    def check_mic(self):
        '''
        Get/print maximal information coefficient
        - Indicates features that have a high MIC with respect to the target variable.
          Stock data probably wont have much given the noisyness
          (ref - http://minepy.sourceforge.net/docs/1.0.0/python.html)

        x - one feature
        y - target

        TODO - float to int bug
        '''
        #import pdb; pdb.set_trace()
        df = self.df.copy()

        def print_stats(mine, feature):
            print "%s MIC: %s" % (feature, mine.mic())
            # print "MAS", mine.mas()
            # print "MEV", mine.mev()
            # print "MCN (eps=0)", mine.mcn(0)
            # print "MCN (eps=1-MIC)", mine.mcn_general()

        features = df[self.get_features()]
        #features = df['deltabWidth310']
        try:
            y = df['y_true']
        except KeyError:
            print "%s.%s: Data has no 'target' column.  Exiting." % (__name__, inspect.currentframe().f_code.co_name)
            return
        #import pdb; pdb.set_trace()
        for feature in features:
            #print '--- MIC for feature "%s" vs "y_true" ---' % feature
            x = df[feature]
            mine = MINE(alpha=0.6, c=15)
            mine.compute_score(x, y)

            #print "Without noise:"
            #print_stats(mine, feature)

            np.random.seed(0)
            y += np.random.uniform(-1, 1, x.shape[0]) # add some noise
            mine.compute_score(x, y)

            #print "With noise:"
            print_stats(mine, feature)

    def check_rfe(self, model, num_top_features):
        print '- model: ', model.__class__
        df = self.df.copy()
        #import pdb; pdb.set_trace()
        X = df.ix[:,6:].copy()
        y = X.pop('y_true')

        estimator = model
        selector = RFE(estimator, num_top_features, step=1)
        selector = selector.fit(X, y)

        selector.support_

        selected_features = []
        for i in np.argsort(selector.ranking_):
            selected_features.append(X.columns[i])
            print X.columns[i]

        return selected_features
