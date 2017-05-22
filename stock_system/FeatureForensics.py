# Instantiate a trading system class and
# run various features analysis for a given data and column set, and model(s)
from stock_system import DataUtils, ModelUtils
from stock_system import TradingSystem_Comp
import sys
import pandas as pd


def run_rfe(num_top_features, ts):
    print '====== Feature Forensics RFE for symbol: ======'
    m = ModelUtils.ModelUtils()
    model_names = m.get_model_list()
    series_list = []
    #model_names = 'abr,linr,logr,lasso,ridge'.split(',')
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
        model = m.get_model(model_name)
        # ts.check_corr()
        # ts.check_mic()
        selected_features = ts.check_rfe(model, num_top_features)
        feature_series = pd.Series(selected_features, name=model_name)
        series_list.append(feature_series)

    # model_features_df = pd.concat(series_list, axis=1).reset_index()
    model_features_df = pd.concat(series_list, join='inner',axis=1)
    # import pdb; pdb.set_trace()
    # del model_features_df['Unnamed: 0']
    model_features_df.to_csv('_FeatureEngineering.csv')

    #pd.read_csv('_FeatureEngineering.csv',index_col=0)

def run_cov_matrix(ts, df):
    print '====== Feature Forensics Covariance Matrix for symbol ======'
    corr_df, corr_top_df = ts.check_corr(df.ix[:,6:-1].columns)
    corr_df.to_csv('_CorrelationMatrix.csv')
    corr_top_df.to_csv('_CorrelationTopMatrix.csv')

if __name__ == '__main__':
    db = DataUtils.DataUtils()
    m = ModelUtils.ModelUtils()
    ts = TradingSystem_Comp.TradingSystem_Comp()

    symbol = sys.argv[1:][0] if len(sys.argv[1:]) > 0 else 'SPY'
    print 'Feature engineering for symbol: ', symbol

    # get stock data from db as dataframe
    df = db.read_symbol_data(symbol, 'd')
    df = ts.preprocess_data(df)
    df = ts.generate_target()

    # run_cov_matrix(ts, df)
    run_rfe(5, ts)
