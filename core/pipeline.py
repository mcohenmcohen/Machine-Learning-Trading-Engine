'''
Import data, implement a trading system and model and backtest on date range

The pipeline is backtesting engine for a given set of symbols, date range,
trading system and selected machine learning model.
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
import pandas as pd
import sys
from data.dataio import DataUtils
from core import accounting, backtester, feature_forensics
from core.model_rfc import Model_RFC
from trading_system.ts_composite import TradingSystem_Comp
from trading_system.ts_khaidem import TradingSystem_Khaidem


if __name__ == '__main__':
    '''
    The process flow is as follows:
        Input symbol(s) to backtest via command line args
        Get symbol data from (local/remote) database
        Preprocess/generate X features and y target for the given trading system
        Select a model and fit it on the trading system features and symbol data
        Calculate the profit and loss for the trade made in the backtest

    The output to the commandline includes model scores and basic profit/loss
    accounting of the profit/loss for the trades made in the backtest run

    '''
    # Read symbols to run on from the stdin
    symbol = sys.argv[1:][0] if len(sys.argv[1:]) > 0 else 'SPY'

    # Get symbol data from the db
    print 'Retrieving symbol data for: ', symbol
    db = DataUtils()
    df_sym = db.read_symbol_data(symbol, 'd')

    # Instantiate the trading system class
    ts = TradingSystem_Comp()
    # tsk = TradingSystem_Khaidem()
    print '= Trading System: ', ts.__class__.__name__

    # Generate the trading system X features and y label
    # via the trading system preprocess
    df_ts = ts.preprocess(df_sym)

    # Get the model, a random forest classifier in this case...
    # m = ModelUtils()
    modeler = Model_RFC()
    print '= Modeler: ', modeler.__class__.__name__
    print '= Model: ', modeler.model.__class__.__name__

    # Run feature engineering/forensics, which identifies important features
    # from the trading system dataframe's features.
    # - check_all runs the feature engineering functions for output only
    # - build_fe_file generates a csv of top features for a given model and
    #   only needs to be re-run if the trading system or feature set changes
    #rfe_num_feat = 10
    #feature_forensics.check_all(df_ts, modeler.get_model(), ts.get_features(), rfe_num_feat)
    #feature_forensics.build_fe_file(df_ts, 4, ts, symbol)

    # Get the top features for the trading system dataframe to use for the chosen model.
    # The top features are ordered in a csv file via feature_forensics
    # - If this file isn't generated or the top n features aren't present in the
    #   trading system, this is bypasses and the existing trading systems
    #   features are used.
    num_top_features = 10  # Use top n features.  Experiment with values.
    csv_path = feature_forensics.get_feature_engineering_file_path()
    features = pd.read_csv(csv_path, index_col=0)
    top_features = features[modeler.model_name][0:num_top_features].tolist()
    # if the top features arent' in the trading system dataframe (iow, a subset)
    # then use existing.
    if set(top_features) < set(ts.get_features()):
        print '= Top features ', ', '.join(top_features).encode("utf-8")
        Xy = top_features + ['y_true']
    else:
        print '= Top features in csv file are not present in trading system dataframe. ' \
              'Using existing:\n  > ', ', '.join(ts.get_features()).encode("utf-8")
        Xy = ts.get_features() + ['y_true']
    df_Xy = df_ts[Xy]

    # Split the model object's data into a train and test set
    modeler.split(df_Xy)

    # Run the backtester in normal mode
    print '= Running backtest...'
    y_pred, scores = backtester.run_once(df_ts, modeler, .7)
    # Run the backtester in multi mode
    # y_pred, scores = backtester.run_n_day_forecast(df_ts, model, 90, .7)

    # Get the scores
    # TODO: Separate scores for classifiers vs regressors
    print
    results_dict = modeler.get_scores(modeler.y_test, y_pred)
    try:
        mat = modeler.standard_confusion_matrix(modeler.y_test, y_pred)
    except:
        mat = np.zeros((2, 2))
    results_dict['tp'] = mat[0][0]
    results_dict['fp'] = mat[0][1]
    results_dict['tn'] = mat[1][1]
    results_dict['fn'] = mat[1][0]

    model_results = []
    df_model = pd.DataFrame(results_dict.values(),
                            index=results_dict.keys(), columns=[modeler.model_name])
    model_results.append(df_model)

    df_all_model_results = pd.concat(model_results, join='inner', axis=1)
    df_all_model_results = df_all_model_results.reindex('f1_score,precision_score,recall_score,accuracy_score,roc_auc_score,r2_score,mean_squared_error,tp,fp,tn,fn'.split(','))
    df_all_model_results.to_csv('_ModelOutput.csv')

    # ** For Accounting **
    # Recreate the original dataframe of test data including the predicted
    # and true y labels
    # TODO: Refactor this
    _df = df_ts[-modeler.y_test.shape[0]:].copy()
    _df['y_true'] = modeler.y_test
    _df['y_pred'] = y_pred
    _df, profit_cm = accounting.get_profit_confusion_matrix_df(_df)
    print profit_cm

    # ** For graphs, best if in a notebook
    # TODO: Polish this up
    # profit_curve_main('data/churn.csv', cost_benefit_matrix)
    #
    # # Run grid search for hyper parameters?
    # gs = GridUtils.GridSearcher()
    # gs.grid_search_reporter(X_train, y_train)
