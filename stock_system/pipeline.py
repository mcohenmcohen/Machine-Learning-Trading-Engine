import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import sys
from stock_system import DataUtils
from stock_system import ModelUtils, GridUtils, Accounting, TA
from stock_system import TradingSystem_Comp, TradingSystem_Khaidem


# Fit, train, and predit the model
def run_once():
    all_scores = m.predict_tscv(model, X_train, y_train, print_on=True)
    print '====== Cross Val Mean Scores ======'
    for key in all_scores[0].keys():
        try:
            mean_val = np.mean([d[key] for d in all_scores])
            print '- %s: %s' % (key, mean_val)
        except:
            pass

    print '====== Top feature imporance ======'
    m.print_feature_importance(model, df[ts.get_features()])

    print '====== Predict Scores ======'
    y_pred = model.predict(X_test)
    # model.score(yhat, y_test)

    # Scores and confusion matrix
    m.get_scores(y_test, y_pred, print_on=True)
    m.print_standard_confusion_matrix(y_test, y_pred)

    return y_pred


def run_for_n_days_ahead(num_days):
    scores_list = []
    for n in range(1, num_days+1):
        print '====== num days ahead: %s ======' % n
        days_ahead = -n
        df['gain_loss'] = np.roll(df['close'], days_ahead) - df['close']
        df['y_true'] = (df['gain_loss'] >= 0).astype(int)
        y = df.pop('y_true').values
        X = df[features].values
        X_train, X_test, y_train, y_test = m.simple_data_split(X, y,
                                                               test_set_size=int(df.shape[0]*.2))

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        all_scores = m.get_scores(y_test, y_pred)
        # all_scores['predict_proba'] = model.predict_proba(X_test)
        m.print_standard_confusion_matrix(y_test, y_pred)
        m.print_feature_importance(model, df[ts.get_features()])
        scores_list.append(all_scores)

    return scores_list


db = DataUtils.DataUtils()
m = ModelUtils.ModelUtils()
ts = TradingSystem_Comp.TradingSystem_Comp()
# ts = TradingSystem_Khaidem.TradingSystem_Khaidem()

symbol = sys.argv[1:][0] if len(sys.argv[1:]) > 0 else 'SPY'
print 'Fitting for symbol: ', symbol

# get stock data from db as dataframe
df = db.read_symbol_data(symbol, 'd')

# Using the tradng system, preprocess the data for it
df = ts.preprocess_data(df)
# Using the tradng system, generate the y column
# df = pd.read_csv('/Users/mcohen/Dev/Trading/Robot Wealth/ML_Scripts_and_Data/eu_daily_midnight.csv')
df = ts.generate_target()
ts.feature_forensics()

# Do feature estimation
# Get the featuers for the trading system
features = ts.get_features()

##############
# Use only relevant columns for the model in X
y = df.pop('y_true').values
X = df.values  # All columns of the original dataframe
df_model_X = df[features]  # Save off the model's X features as a dataframe

# Split
X_train, X_test, y_train, y_test = m.simple_data_split(df_model_X.values, y,
                                                       test_set_size=int(df.shape[0]*.2))
# Instantiate model(s)
model = m.get_model('rf')
# Predict
y_pred = run_once()

# ##### For Accounting ######
# # Recreate the original dataframe of test data including the predicted and true y labels
# df_train = df[0:X_train.shape[0]].copy()
df_test = df[X_train.shape[0]:].copy()
# Add back in the y_true and y_pred label columns
df_test['y_true'] = y_test
df_test['y_pred'] = y_pred
df_test['gain_loss'] = np.roll(df_test['close'], -1) - df_test['close']

# Convenient subset of accounting
df['gain_loss'] = np.roll(df_test['close'], 1) - df_test['close']
# cols_acc = ['date', 'close', 'gain_loss', 'y_true', 'y_pred']
cols_acc = ['close', 'gain_loss', 'y_true', 'y_pred']
df_test = df_test[cols_acc]

acct = Accounting.get_abs_return(df_test)
print acct

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
