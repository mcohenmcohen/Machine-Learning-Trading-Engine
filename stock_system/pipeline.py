import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import sys
from stock_system import DataUtils, ModelUtils, GridUtils, Accounting


def get_features():
    x_cols = ['roc', 'rsi', 'willr', 'obv', 'stok']
    # Oscilators
    # x_osc = ['rsi', 'cci', 'stod', 'stok', 'willr']
    # x_oscd_cols = ['rsi_d', 'cci_d', 'stod_d', 'stok_d', 'willr_d']
    # # MAs
    # x_ma_cols = ['sma20', 'sma50', 'sma200', 'wma10', 'macd_d']
    # x_all_dscrete_cols = ['roc_d', 'rsi_d', 'cci_d', 'stod_d', 'stok_d', 'willr_d', 'mom_d']
    # #x_cols = ['roc', 'rsi', 'willr', 'obv', 'stok']#'mom', , 'cci',  'stod', 'macd', 'sma', 'sma50', 'wma']
    # #x_cols = ['roc']
    # x_cols = x_all_dscrete_cols + x_ma_cols
    return  x_cols


def get_model():
    model = RandomForestClassifier(
       n_estimators=500,
       max_depth=None,
       min_samples_leaf=10,
       max_features='log2',
       bootstrap=False
       #oob_score=True
       )
    # model = RandomForestRegressor(
    #     n_estimators=500,
    #     n_jobs=-1,
    #     max_depth=None,
    #     max_features='auto',
    #     oob_score=True
    # )
    # model = AdaBoostRegressor(
    #     n_estimators=500,
    #     random_state=0,
    #     learning_rate=0.1
    # )

    return model


# Fit, train, and predit the model
def run_once():
    all_scores = m.predict_tscv(model, X_train, y_train)
    print '====== Cross Val Mean Scores ======'
    for key in all_scores[0].keys():
        try:
            mean_val = np.mean([d[key] for d in all_scores])
            print '- %s: %s' % (key, mean_val)
        except:
            pass

    print '====== Top feature imporance ======'
    m.print_feature_importance(model, df[get_features()])

    print '====== Predict Scores ======'
    y_pred = model.predict(X_test)
    # model.score(yhat, y_test)

    # Scores and confusion matrix
    m.print_scores(y_test, y_pred)
    m.print_standard_confusion_matrix(y_test, y_pred)

    return y_pred

def run_for_n_days_ahead(num_days):
    model = get_model()
    scores_list = []
    for n in range(1,num_days+1):
        print '====== num days ahead: %s ======' % n
        days_ahead = -n
        df['gain_loss'] = np.roll(df['close'], days_ahead) - df['close']
        df['y_true'] = (df['gain_loss'] >= 0).astype(int)
        y = df.pop('y_true').values
        X = df.values
        X_train, X_test, y_train, y_test = m.simple_data_split(df[get_features()].values, y, test_set_size=int(df.shape[0]*.2))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_scores = m.print_scores(y_test, y_pred)
        # all_scores['predict_proba'] = model.predict_proba(X_test)
        m.print_standard_confusion_matrix(y_test, y_pred)
        m.print_feature_importance(model, df[get_features()])
        scores_list.append(all_scores)

    return scores_list


db = DataUtils.DataUtils()
m = ModelUtils.ModelUtils()

symbol = sys.argv[1:][0] if len(sys.argv[1:]) > 0 else 'SPY'
print 'Fitting for symbol: ', symbol

# get stock data from db as dataframe
# df = db.read_symbol_data("Select * from symbols where symbol='FB' limit 100") # Dummy data
df = db.read_symbol_data(symbol, 'd')
# run expinential smoothing
df = db.run_exp_smooth(df, alpha=.5)
# run technical analysis, add columns
df = db.run_techicals(df)
# subset out rows that have nan's
# df = df[np.isfinite(df['macd_d'])]
for name in df.columns:
    df = df[df[name].notnull()]
# Imput - fillna with 0 for now...
#df = df.fillna(0)
# Number of days ahead to see if the price moved up or down
days_ahead = -30
df['gain_loss'] = np.roll(df['close'], days_ahead) - df['close']
df['y_true'] = (df['gain_loss'] >= 0).astype(int)
# Drop the last row?
df = df[:-1]

##############
# Use only relevant columns for the model in X
y = df.pop('y_true').values
X = df.values

# Split
X_train, X_test, y_train, y_test = m.simple_data_split(df[get_features()].values, y, test_set_size=100)

# Instantiate model(s)
# Results from grid search
# {'max_features': 'log2', 'n_estimators': 1000, 'min_samples_leaf': 10}
model = get_model()

y_pred = run_once()

# # Recreate the original dataframe of test data including the predicted and true y labels
# train_len = X_train.shape[0]
# test_len = y_train.shape[0]
# df_train = df[0:train_len].copy()
# df_test = df[train_len:df.shape[0]].copy()
# # Add back in the y_true and y_pred label columns
# df_test['y_true'] = y_test
# df_test['y_pred'] = y_pred
#
# df_test['tp'] = (df_test['y_true'] == df_test['y_pred']) & (df_test['y_true'] == 1)
# df_test['tn'] = (df_test['y_true'] == df_test['y_pred']) & (df_test['y_true'] == 0)
# df_test['fp'] = (df_test['y_true'] != df_test['y_pred']) & (df_test['y_pred'] == 1)
# df_test['fn'] = (df_test['y_true'] != df_test['y_pred']) & (df_test['y_pred'] == 0)
#
# # df_test[['close', 'gain_loss', 'y_true', 'y_pred','tp','tn','fp','fn']]
# df_test[['close', 'gain_loss', 'y_true', 'y_pred', 'roc', 'stok', 'macd', 'willr', 'rsi']]
#
# tp_gl_mean = df_test['gain_loss'][df_test['tp']].mean()
# tn_gl_mean = df_test['gain_loss'][df_test['tn']].mean()
# fp_gl_mean = df_test['gain_loss'][df_test['fp']].mean()
# fn_gl_mean = df_test['gain_loss'][df_test['fn']].mean()
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
# model.feature_importances_
#
#
# def get_positives_data(X, y):
#     return X[y == 1], y[y == 1]
