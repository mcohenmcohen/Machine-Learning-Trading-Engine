import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import sys
from stock_system import DataUtils, ModelUtils, GridUtils, Accounting, TA, FeatureForensics
from stock_system import TradingSystem_Comp, TradingSystem_Khaidem


# Fit, train, and predit the model
def run_once(X_train, X_test, y_train, y_test, features, thresh=.5):
    #all_scores = m.fit_predict_score_tscv(model, X_train, y_train, num_folds=10, print_on=True)
    all_scores = [m.fit_predict_score(model, X_train, y_train, X_test, y_test, print_on=True)]
    print '====== Cross Val Mean Scores ======'
    for key in all_scores[0].keys():
        try:
            mean_val = np.mean([d[key] for d in all_scores])
            print '- %s: %s' % (key, mean_val)
        except:
            pass

    print '====== Top feature imporance ======'
    m.print_feature_importance(model, df[features])

    print '====== Predict Scores ======'
    y_pred = model.predict(X_test)
    # y_pred = model.predict_proba(X_test)
    # y_pred = (y_pred[:,1] > thresh).astype(int)

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
print 'Retrieving symbold data for: ', symbol

df_orig = db.read_symbol_data(symbol, 'd')  # get stock data from db as dataframe
#df.set_index('date', inplace=True)
df_orig = ts.preprocess_data(df_orig)  # Using the tradng system, preprocess the data for it
df_orig = ts.generate_target()  # generate the y label column
# df = pd.read_csv('/Users/mcohen/Dev/Trading/Robot Wealth/ML_Scripts_and_Data/eu_daily_midnight.csv')

# Run feature engineering/forensics.
# ts.feature_forensics(model)
# ts.check_corr()
# ts.check_mic()
# ts.check_rfe(model)
# FF = FeatureForensics.run_rfe(5, ts)

# Get the featuers for the trading system
#features = ts.get_features()
model_names = m.model_list
model_names = ['rfc']
model_results = []
for model_name in model_names:
    print '================================'
    print '====== Fit, predict: %s =======' % model_name
    print '================================'
    df = df_orig.copy()
    model = m.get_model(model_name)
    df_model_features = pd.read_csv('/Users/mcohen/Dev/Trading/trading_ml/_FeatureEngineering.csv',index_col=0)
    num_features_to_use = 10
    try:
        features = df_model_features[model_name][0:num_features_to_use].tolist()
    except:
        print '- - - model %s does not use feautres. - - -' % model_name

    # Use only relevant columns for the model in X
    y = df.pop('y_true').values
    X = df.values  # All columns of the original dataframe
    df_model_X = df[features]  # Save off the model's X features as a dataframe

    # Split
    X_train, X_test, y_train, y_test = m.simple_data_split(df_model_X.values, y,
                                                           test_set_size=int(df.shape[0]*.2))

    # Predict
    y_pred = run_once(X_train, X_test, y_train, y_test, features, .7)

    results_dict = m.get_scores(y_test, y_pred)
    try:
        mat = m.standard_confusion_matrix(y_test, y_pred)
    except:
        mat = np.zeros((2,2))
    results_dict['tp'] = mat[0][0]
    results_dict['fp'] = mat[0][1]
    results_dict['tn'] = mat[1][1]
    results_dict['fn'] = mat[1][0]

    df_model = pd.DataFrame(results_dict.values(), index=results_dict.keys(),columns=[model_name])
    model_results.append(df_model)

df_all_model_results = pd.concat(model_results,join='inner',axis=1)
df_all_model_results = df_all_model_results.reindex('f1_score,precision_score,recall_score,accuracy_score,roc_auc_score,r2_score,mean_squared_error,tp,fp,tn,fn'.split(','))
df_all_model_results.to_csv('_ModelOutput.csv')

# ##### For Accounting ######
# # Recreate the original dataframe of test data including the predicted and true y labels
# df_train = df[0:X_train.shape[0]].copy()
df = df[X_train.shape[0]:].copy()
# Add back in the y_true and y_pred label columns
df['y_true'] = y_test
df['y_pred'] = y_pred
#df_test['daily_ret'] = np.roll(df_test['close'], -1) - df_test['close']

# Convenient subset of accounting
#df['daily_ret'] = np.roll(df_test['close'], 1) - df_test['close']
# cols_acc = ['date', 'close', 'daily_ret', 'y_true', 'y_pred']
cols_acc = ['close', 'y_true', 'y_pred', 'daily_ret', 'daily_ret_pred', 'running_ret_pred']
#df_test = df_test[cols_acc]

profit_cm = Accounting.get_profit_confusion_matrix_df(df)
print profit_cm

# profit_curve_main('data/churn.csv', cost_benefit_matrix)
#
# # Run grid search for hyper parameters?
# gs = GridUtils.GridSearcher()
# gs.grid_search_reporter(X_train, y_train)
