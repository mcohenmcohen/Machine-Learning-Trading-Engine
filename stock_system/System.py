###################################################################################################
# Parent class for trading systems
###################################################################################################
import pandas as pd
from minepy import MINE


class System(object):
    '''
    Parent class for trading systems
    '''
    def __init__(self):
        self.data = None  # The source data set - the X features
        self.target = None  # The target to be y label, generated from a system algorithm
        self.name = ''  # An optional name
        self.x_cols = []  # list to hold the X columns used by the trading system

    def preprocess_data(self):
        '''
        Perform any data preprocessing steps such as normalizing, smoothing,
        remove correlated columns, etc
        '''
        pass

    def generate_target(self):
        '''
        Trading system goes here, result is a y label
        '''
        pass

    def check_corr(self):
        df = self.data
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
        print(get_top_abs_correlations(df, 3))

    def check_mic(self):
        '''
        Get/print maximal information coefficient
        - Indicates features that have a high MIC with respect to the target variable.
          Stock data probably wont have much given the noisyness
          (ref - http://minepy.sourceforge.net/docs/1.0.0/python.html)
        '''
        df = self.data

        def print_stats(mine):
            print "MIC", mine.mic()
            print "MAS", mine.mas()
            print "MEV", mine.mev()
            print "MCN (eps=0)", mine.mcn(0)
            print "MCN (eps=1-MIC)", mine.mcn_general()

        x = df[self.x_cols]
        y = df['target']
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x, y)

        print "Without noise:"
        print_stats(mine)
        print

        np.random.seed(0)
        y +=np.random.uniform(-1, 1, x.shape[0]) # add some noise
        mine.compute_score(x, y)

        print "With noise:"
        print_stats(mine)