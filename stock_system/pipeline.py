import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from stock_system import DataUtils, ModelUtils, GridUtils, Accounting

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
# Add a gain/loss column in dollars between close price data points
diff = np.diff(df['close'])
df['gain_loss'] = np.append(0, diff)
# get y data as a function of last n bars (can vary)
# - could generate this as a param:
# - - time period:  daily, 5 minute, estimators_w_grid_dict
# df['y_true'] = (df.close.pct_change(1) >= 0).astype(int)
df['y_true'] = (df.gain_loss >= 0).astype(int)
# Drop the last row?
df = df[:-1]

############## Orig ##############
# # Use only relevant columns for the model in X
# x_cols = ['roc', 'sto', 'macd', 'willr', 'rsi']
# X = df[x_cols].values
# # Keep the y label and date to keep track of which row in y
# y_cols = ['y_true', 'date']
# y = df[y_cols].values
#
# # Split
# # X_train, X_test, y_train, y_test = train_test_split(X, y)  # Can't do with time series
# X_train, X_test, y_train_d, y_test_d = m.simple_data_split(X, y, test_set_size=100)
# y_train_dates = y_train_d[:,1]
# y_train = y_train_d[:,0].astype(int)
# y_test_dates = y_test_d[:,1]
# y_test = y_test_d[:,0].astype(int)
##############
# Use only relevant columns for the model in X
y = df.pop('y_true').values
X = df.values
x_cols = ['roc', 'sto', 'macd', 'willr', 'rsi']
X_subset = df[x_cols].values

# Split
# X_train, X_test, y_train, y_test = train_test_split(X, y)  # Can't do with time series
X_train, X_test, y_train, y_test = m.simple_data_split(X_subset, y, test_set_size=100)
# y_train_dates = y_train_d[:,1]
# y_train = y_train_d[:,0].astype(int)
# y_test_dates = y_test_d[:,1]
# y_test = y_test_d[:,0].astype(int)

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


y_pred = rf.predict(X_test)
# rf.score(yhat, y_test)

# Scores and confusion matrix
m.print_scores(y_test, y_pred)
m.print_standard_confusion_matrix(y_test, y_pred)

# Run y labels through Logistic Regression

# Recreate the original dataframe of test data including the predicted and true y labels
train_len = X_train.shape[0]
test_len = y_train.shape[0]
df_train = df[0:train_len].copy()
df_test = df[train_len:df.shape[0]].copy()
# Add back in the y_true and y_pred label columns
df_test['y_true'] = y_test
df_test['y_pred'] = y_pred

df_test['tp'] = (df_test['y_true'] == df_test['y_pred']) & (df_test['y_true'] == 1)
df_test['tn'] = (df_test['y_true'] == df_test['y_pred']) & (df_test['y_true'] == 0)
df_test['fp'] = (df_test['y_true'] != df_test['y_pred']) & (df_test['y_pred'] == 1)
df_test['fn'] = (df_test['y_true'] != df_test['y_pred']) & (df_test['y_pred'] == 0)

# df_test[['close', 'gain_loss', 'y_true', 'y_pred','tp','tn','fp','fn']]
df_test[['close', 'gain_loss', 'y_true', 'y_pred', 'roc', 'sto', 'macd', 'willr', 'rsi']]

tp_gl_mean = df_test['gain_loss'][df_test['tp']].mean()
tn_gl_mean = df_test['gain_loss'][df_test['tn']].mean()
fp_gl_mean = df_test['gain_loss'][df_test['fp']].mean()
fn_gl_mean = df_test['gain_loss'][df_test['fn']].mean()
#
# # Do the accounting
# # Compute the gain and loss of each tp, fp, tn, fn
# profit_curve_main('data/churn.csv', cost_benefit_matrix)
#
#
# # Run grid search for hyper parameters?
# gs = GridUtils.GridSearcher()
# gs.grid_search_reporter(X_train, y_train)
# # Feature importances
# rf.feature_importances_
#
#
# def get_positives_data(X, y):
#     return X[y == 1], y[y == 1]
