'''
Utility functions for analyses on a set of features
Instantiate a trading system class and
run various features analysis for a given data and column set, and model(s)
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
from minepy import MINE
from sklearn.feature_selection import RFE
from stock_system.data import DataUtils
from stock_system.model import ModelUtils
from stock_system.ts_composite import TradingSystem_Comp
import sys
import pandas as pd


def get_feature_engineering_file_path():
    return '_data/_FeatureEngineering.csv'


def run(df, model, feature_set, rfe_num_feat):
    print '====== Identify to Remove highly correlated variables ======'
    check_corr(df, feature_set)

    print '====== Feature selection via MIC ======'
    #check_mic(df, feature_set)

    # RFE should be run offline via run_rfe to generate csv file
    #print '====== Recursive Feature Extraction ======'
    #check_rfe(df, model, feature_set, rfe_num_feat)


def check_corr(df, feature_set, print_matrix=False):
    '''
    Get/print a correlation matrix to assist in identifying correlated columns
    '''
    # df = self.df.select_dtypes(['number'])  # Use only numeric columns
    # df = self.df.copy()
    # import pdb; pdb.set_trace()
    df = df[feature_set]  # Use only sub set features
    if print_matrix:
        print "Correlation Matrix"
        print df.corr()

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

    #return (df.corr(), get_top_abs_correlations(df, 100))


def check_mic(df, feature_set):
    '''
    Get/print Maximal Information Coefficient
    - Indicates features that have a high MIC with respect to the target variable.
      Stock data probably wont have much given the noisyness
      (ref - http://minepy.sourceforge.net/docs/1.0.0/python.html)

    x - one feature
    y - target

    TODO - float to int bug
    '''
    # df = self.df.copy()
    def print_stats(mine, feature):
        print "%s MIC: %s" % (feature, mine.mic())
        # print "MAS", mine.mas()
        # print "MEV", mine.mev()
        # print "MCN (eps=0)", mine.mcn(0)
        # print "MCN (eps=1-MIC)", mine.mcn_general()

    features = df[feature_set]
    # features = df['deltabWidth310']
    try:
        y = df['y_true'].copy()
    except KeyError:
        print "%s.%s: Data has no 'target' column.  Exiting." % (__name__, inspect.currentframe().f_code.co_name)
        return
    # import pdb; pdb.set_trace()
    for feature in features:
        # print '--- MIC for feature "%s" vs "y_true" ---' % feature
        x = df[feature]
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x, y)

        # print "Without noise:"
        # print_stats(mine, feature)

        np.random.seed(0)
        y += np.random.uniform(-1, 1, x.shape[0])  # add some noise
        mine.compute_score(x, y)

        #print "With noise:"
        print_stats(mine, feature)


def check_rfe(in_df, model, feature_set, num_top_features):
    #https://medium.com/@aneesha/recursive-feature-elimination-with-scikit-learn-3a2cbdf23fb7
    # df = self.df.copy()
    # import pdb; pdb.set_trace()
    # cols = [col for col in df.columns if col not in self.excluded_features]
    df = in_df.copy()
    X = df[feature_set]
    y = df.pop('y_true')
    #import pdb; pdb.set_trace()
    estimator = model
    selector = RFE(estimator, num_top_features, step=1)

    print 'Fitting RFE (this can take a while with many features.  This trading system  has %s features)' % X.shape[1]
    import time; t1 = time.time()
    selector = selector.fit(X, y)
    t2 = time.time(); print "Done. RFE processing time: " + str((t2 - t1))

    # print summaries for the selection of attributes
    #print selector.support_
    # print selector.ranking_
    selected_features = []
    print 'top features: '
    for i in np.argsort(selector.ranking_):
        selected_features.append(X.columns[i])
        print '-', X.columns[i]

    return selected_features


#####################################

def run_rfe(in_df, num_top_features, ts, symbol):
    print '====== Feature Engineering RFE for symbol: %s ======' % symbol
    m = ModelUtils()
    model_names = m.get_model_list()
    series_list = []
    #model_names = 'abr,linr,logr,lasso,ridge'.split(',')

    df = in_df.copy()
    for model_name in model_names:
        # no coef_ or feature_importance_ not supported for the estimators:
        if model_name == 'knn':
            continue
        if model_name == 'svc':  # svc has no coef_ or feature_importance_ param
            continue
        if model_name == 'svr':  # svr has no coef_ or feature_importance_ param
            continue
        if model_name == 'abr':
            continue
        model = m.set_model(model_name)
        # ts.check_corr()
        # ts.check_mic()
        print '=== Process top features for model: %s ===' % model.__class__.__name__
        selected_features = check_rfe(df, model, ts.get_features(), num_top_features)
        feature_series = pd.Series(selected_features, name=model_name)
        series_list.append(feature_series)

    # model_features_df = pd.concat(series_list, axis=1).reset_index()
    model_features_df = pd.concat(series_list, join='inner',axis=1)
    # import pdb; pdb.set_trace()
    # del model_features_df['Unnamed: 0']
    f_path = get_feature_engineering_file_path()
    print 'Write features to csv to %s' % f_path
    model_features_df.to_csv(f_path)

    #pd.read_csv('_FeatureEngineering.csv',index_col=0)

def run_cov_matrix(ts, df):
    print '====== Feature Forensics Covariance Matrix for symbol ======'
    corr_df, corr_top_df = ts.check_corr(df.ix[:,6:-1].columns)
    corr_df.to_csv('_CorrelationMatrix.csv')
    corr_top_df.to_csv('_CorrelationTopMatrix.csv')


def VWAP(in_df):
    '''
    Volume Weighted Average Price (VWAP)

    The volume weighted average price (VWAP) is a trading benchmark used
    especially in pension plans. VWAP is calculated by adding up the
    dollars traded for every transaction (price multiplied by number of shares
    traded) and then dividing by the total shares traded for the day.

    The theory is that if the price of a buy trade is lower than the VWAP, it
    is a good trade. The opposite is true if the price is higher than the VWAP.
    '''
    pass

def TWAP(in_df):
    '''
    Time Weighted Average Price (TWAP)
    '''
    pass

if __name__ == '__main__':
    db = DataUtils()
    m = ModelUtils()
    ts = TradingSystem_Comp()

    symbol = sys.argv[1:][0] if len(sys.argv[1:]) > 0 else 'SPY'

    # get stock data from db as features dataframe from the trading system
    df = db.read_symbol_data(symbol, 'd')
    df = ts.preprocess(df)

    # run_cov_matrix(ts, df)
    run_rfe(df, 5, ts, symbol)
