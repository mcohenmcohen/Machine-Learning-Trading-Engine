import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, mean_squared_error, r2_score, f1_score

from stock_system import IQFeed, DataUtils


# get stock data from db as dataframe
db = DataUtils.DataUtils()
df = db.read_symbol_data("Select * from symbols where symbol='FB' limit 100")
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
cols = ['roc','sto','macd','willr','rsi']
X = df[cols].values
y = df['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)
rf.score(yhat, y_test)

relevant_metrics = [precision_score, recall_score, accuracy_score, roc_auc_score, mean_squared_error, r2_score, f1_score]
for metric in relevant_metrics:
    m = metric(y_test, y_hat)
    #scores[metric.__name__] = m
    print metric.__name__, ' = ', m

random_forest_grid = {
    'n_estimators': [50, 100, 1000],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 10, 50],
}
gscv = GridSearchCV(
    rf,
    random_forest_grid,
    n_jobs=-1,
    verbose=True,
    scoring='f1'
)
gscv.fit(X_train, y_train)
print "  f1 score: {}".format(gscv.best_score_)
print "  params: {}".format(gscv.best_params_)



# crovalidate?
# feature importance, scores
# confusion matrix, plot
