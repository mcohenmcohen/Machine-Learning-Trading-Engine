# Instantiate a trading system class and
# run various features analysis for a given data and column set, and model(s)
from stock_system import DataUtils, ModelUtils
from stock_system import TradingSystem_Comp
import sys
import pandas as pd


db = DataUtils.DataUtils()
m = ModelUtils.ModelUtils()
ts = TradingSystem_Comp.TradingSystem_Comp()

symbol = sys.argv[1:][0] if len(sys.argv[1:]) > 0 else 'SPY'
print 'Feature engineering for symbol: ', symbol

# get stock data from db as dataframe
df = db.read_symbol_data(symbol, 'd')
df = ts.preprocess_data(df)
df = ts.generate_target()
model_names = m.get_model_list()
series_list = []
# model_names = 'linr,logr,lasso,ridge'.split(',')
for model_name in model_names:
    if model_name == 'knn':  # knn has no coef_ or feature_importance_ param
        continue
    if model_name == 'svc':  # svc has no coef_ or feature_importance_ param
        continue
    if model_name == 'svr':  # svr has no coef_ or feature_importance_ param
        continue
    model = m.get_model(model_name)
    # ts.check_corr()
    # ts.check_mic()
    selected_features = ts.check_rfe(model)
    feature_series = pd.Series(selected_features, name=model_name)
    series_list.append(feature_series)

model_features_df = pd.concat(series_list, axis=1).reset_index()

model_features_df.to_csv('FeatureEngineering.csv')
