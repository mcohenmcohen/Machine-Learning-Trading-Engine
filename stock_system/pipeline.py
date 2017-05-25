import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import sys
from stock_system import DataUtils, GridUtils, Accounting, TA, FeatureForensics
from stock_system import ModelUtils, Model_RFC
from stock_system import TradingSystem_Comp, TradingSystem_Khaidem


# Fit, train, and predit the model
def run_once(in_df, model, thresh=.5, print_on=True):
    df = in_df.copy()
    # Use only relevant columns for the model in X
    y = df.pop('y_true').values
    X = df.values  # All columns of the original dataframe

    scores = [model.fit_predict_score(print_on=print_on)]
    y_pred = model.model.predict(model.X_test)

    if (print_on):
        print '====== Cross Val Mean Scores ======'
        for key in scores[0].keys():
            try:
                mean_val = np.mean([d[key] for d in scores])
                print '- %s: %s' % (key, mean_val)
            except:
                pass

        print '====== Top feature imporance ======'
        model.print_feature_importance(model, df[model.features])

        print '====== Predict Scores ======'
        model.get_scores(model.y_test, y_pred, print_on=print_on)
        # y_pred = model.predict_proba(X_test)
        # y_pred = (y_pred[:,1] > thresh).astype(int)

        model.print_standard_confusion_matrix(model.y_test, y_pred,)

    return y_pred, scores


def run_n_day_forecast(in_df, model, n_day_forecast, print_on=True):
    df = in_df.copy()
    scores_list = []

    df['gain_loss'] = df['close'].shift(-n_day_forecast) - df['close']
    df['y_true'] = (df['gain_loss'] >= 0).astype(int)

    features_and_y = list(model.features)
    features_and_y = features_and_y + ['y_true']
    m_rfc.split(df[features_and_y])  # split on the feature and target columns        model.model.fit(model.X_train, model.y_train)
    model.fit()

    #import pdb; pdb.set_trace()
    y_pred = model.model.predict(model.X_test[0:-n_day_forecast])
    y_pred = np.append(y_pred, np.zeros(n_day_forecast))  # zero backfill

    if (print_on):
        model.print_feature_importance(model.model, df[model.features])
        model.print_standard_confusion_matrix(model.y_test, y_pred)

    scores = model.get_scores(model.y_test, y_pred)
    # scores['predict_proba'] = model.predict_proba(X_test)

    return y_pred, scores


db = DataUtils.DataUtils()
m = ModelUtils.ModelUtils()
ts = TradingSystem_Comp.TradingSystem_Comp()
tsk = TradingSystem_Khaidem.TradingSystem_Khaidem()

symbol = sys.argv[1:][0] if len(sys.argv[1:]) > 0 else 'SPY'
print 'Retrieving symbold data for: ', symbol

df_orig = db.read_symbol_data(symbol, 'd')  # get stock data from db as dataframe
df_orig = ts.preprocess_data(df_orig)  # Using the tradng system, preprocess the data for it
df_orig = ts.generate_target()  # generate the y label column

# Run feature engineering/forensics.
# ts.feature_forensics(model)
# ts.check_corr()
# ts.check_mic()
# ts.check_rfe(model)
# FF = FeatureForensics.run_rfe(5, ts)

# Get features for the model from file
features_df = pd.read_csv('_FeatureEngineering.csv', index_col=0)
m_rfc = Model_RFC.Model_RFC()  # get the model
features = features_df[m_rfc.name][0:10].tolist()  # Use top 10 features.  Experiment with values
features_and_y = features + ['y_true']
m_rfc.split(df_orig[features_and_y])  # split on the feature and target columns

# y _pred, scores = run_once(df_orig, m_rfc, .7)
y_pred, scores = run_n_day_forecast(df_orig, m_rfc, 90, .7)

results_dict = m_rfc.get_scores(m_rfc.y_test, y_pred)
try:
    mat = m_rfc.standard_confusion_matrix(m_rfc.y_test, y_pred)
except:
    mat = np.zeros((2, 2))
results_dict['tp'] = mat[0][0]
results_dict['fp'] = mat[0][1]
results_dict['tn'] = mat[1][1]
results_dict['fn'] = mat[1][0]

model_results = []
df_model = pd.DataFrame(results_dict.values(),
                        index=results_dict.keys(), columns=[m_rfc.name])
model_results.append(df_model)

df_all_model_results = pd.concat(model_results, join='inner', axis=1)
df_all_model_results = df_all_model_results.reindex('f1_score,precision_score,recall_score,accuracy_score,roc_auc_score,r2_score,mean_squared_error,tp,fp,tn,fn'.split(','))
df_all_model_results.to_csv('_ModelOutput.csv')

# ##### For Accounting ######
# # Recreate the original dataframe of test data including the predicted and true y labels
_df = df_orig[-m_rfc.y_test.shape[0]:].copy()
_df['y_true'] = m_rfc.y_test
_df['y_pred'] = y_pred
_df, profit_cm = Accounting.get_profit_confusion_matrix_df(_df)
print profit_cm

# profit_curve_main('data/churn.csv', cost_benefit_matrix)
#
# # Run grid search for hyper parameters?
# gs = GridUtils.GridSearcher()
# gs.grid_search_reporter(X_train, y_train)
