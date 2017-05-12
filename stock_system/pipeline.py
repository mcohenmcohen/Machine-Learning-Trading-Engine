import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from stock_system import DataUtils, ModelUtils, GridUtils

db = DataUtils.DataUtils()
m = ModelUtils.ModelUtils()

# get stock data from db as dataframe
# df = db.read_symbol_data("Select * from symbols where symbol='FB' limit 100") # Dummy data
df = db.read_symbol_data('SPY', 'd')
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
# Drop the last row?
df = df[:-1]

# Use only relevant columns for the model in X
x_cols = ['roc', 'sto', 'macd', 'willr', 'rsi']
X = df[x_cols].values
# Keep the y label and date to keep track of which row in y
y_cols = ['target', 'date']
y = df[y_cols].values

# Split
# X_train, X_test, y_train, y_test = train_test_split(X, y)  # Can't do with time series
X_train, X_test, y_train_d, y_test_d = m.simple_data_split(X, y, test_set_size=100)
y_train_dates = y_train_d[:,1]
y_train = y_train_d[:,0].astype(int)
y_test_dates = y_test_d[:,1]
y_test = y_test_d[:,0].astype(int)

# Instantiate model(s)
# Results from grid search
# {'max_features': 'log2', 'n_estimators': 1000, 'min_samples_leaf': 10}
rf = RandomForestClassifier(
       n_estimators=500,
       max_depth=None,
       min_samples_leaf=10,
       max_features='log2',
       oob_score=True
   )

# Fit, train, and predit the model
all_scores = m.predict_tscv(rf, X_train, y_train)
print '====== Cross Val Mean Scores ======'
for key in all_scores[0].keys():
    mean_val = np.mean([d[key] for d in all_scores])
    print '- %s: %s' % (key, mean_val)


y_hat = rf.predict(X_test)
# rf.score(yhat, y_test)

# Scores and confusion matrix
m.print_scores(y_test, y_hat)
m.print_standard_confusion_matrix(y_test, y_hat)

# Do the accounting
# Compute the gain and loss of each tp, fp, tn, fn


# Run grid search for hyper parameters?
gs = GridUtils.GridSearcher()
gs.grid_search_reporter(X_train, y_train)
# Feature importances
rf.feature_importances_


def get_positives_data(X, y):
    return X[y == 1], y[y == 1]
