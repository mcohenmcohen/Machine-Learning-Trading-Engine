'''
Provide backtesting for a given model and data frame, over a single date or
a date range.
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np


def run_once(in_df, model, thresh=.5, print_on=True):
    '''
    Run the backtest on the fitted model and data set.
    We execute once - to predict the next period.

    Input:
        in_df : dataframe (TODO: switch to np arrays)
            The cost matrix of the bipartite graph
        model : ModelUtils object
            Object stores the model and scoring functions
        print_on : boolean
            Flag to print outputs

    Output:
        y_pred : array
            Array of prediction labels
        scores : list
            List of various model scores
    '''

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
    '''
    Backtest the trading system against an ROC exit signal n days in the future.

    This function was origianlly implemented to support the Khaidem research
    paper replication, which targeted a price change 90 days out.

    *Should be just the X and y values, this funtion shouldnt have to figure out
    the y label

    Input:
        in_df : pandas dataframe
            Expected to be comprised of trading features and date range subset
        n_day_forecast : integer
            Number of days to forecast out the price change
        print_on : boolean
            Flag to output backtest results
    Output:
        y_pred : array
            Array of prediction labels
        scores : list
            List of various model scores
    '''
    df = in_df.copy()
    scores_list = []

    # Create a y_true feature column n days out
    df.loc[:,('gain_loss')] = df['close'].shift(-n_day_forecast) - df['close']
    df['y_true'] = (df['gain_loss'] >= 0).astype(int)

    features_and_y = list(model.features)
    features_and_y = features_and_y + ['y_true']
    m_rfc.split(df[features_and_y])  # split on the feature and target columns       model.model.fit(model.X_train, model.y_train)
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
