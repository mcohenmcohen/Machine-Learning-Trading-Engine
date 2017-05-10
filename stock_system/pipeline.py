import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from stock_system import DataUtils, ModelUtils


# get stock data from db as dataframe
db = DataUtils.DataUtils()
# df = db.read_symbol_data("Select * from symbols where symbol='FB' limit 100") # Dummy data
df = db.read_symbol_data("Select * from symbols where symbol='SPY'")
# run expinential smoothing
df = db.run_exp_smooth(df, alpha=.5)
# run technical analysis, add columns
df = db.run_techicals(df)
# Imput - fillna with 0 for now...
df = df.fillna(0)
# get y data as a function of last n bars (can vary)
# - could generate this as a param:
# - - time period:  daily, 5 minute, estimators_w_grid_dict
df['target'] = (df.close.pct_change(1) >= 0).astype(int)
# fit, predict
cols = ['roc', 'sto', 'macd', 'willr', 'rsi']
X = df[cols].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestClassifier()

# Run grid search for hyper parameters
m = ModelUtils.ModelUtils()
# m.grid_search(rf, X_train, y_train)

# feature importance, scores

rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)
# rf.score(yhat, y_test)

m.print_scores(y_test, y_hat)
rf.feature_importances_


# crovalidate?
# confusion matrix, plot
