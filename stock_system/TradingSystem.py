###################################################################################################
# Parent class for trading systems
###################################################################################################
from minepy import MINE
from sklearn.feature_selection import RFE
import inspect
from stock_system import TA, ModelUtils


class TradingSystem(object):
    '''
    Parent class for a trading systems
    '''
    def __init__(self):
        self.name = ''  # An optional name

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

    def check_corr(self):
        '''
        Get/print a correlation matrix to assist in identifying correlated columns
        '''
        df = self.data.select_dtypes(['number'])  # Use only numeric columns
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
        df = self.data

        def print_stats(mine):
            print "MIC", mine.mic()
            print "MAS", mine.mas()
            print "MEV", mine.mev()
            print "MCN (eps=0)", mine.mcn(0)
            print "MCN (eps=1-MIC)", mine.mcn_general()

        x = df[self.x_cols]
        try:
            y = df['target']
        except KeyError:
            print "%s.%s: Data has no 'target' column.  Exiting." % (__name__, inspect.currentframe().f_code.co_name)
            return
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
