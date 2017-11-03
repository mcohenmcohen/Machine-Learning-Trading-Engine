'''
Class provides functions to fit, cross validate, predict, score and
print model stats
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV
import timeit


class ModelUtils(object):
    '''
    Class provides functions to fit, cross validate, predict, score and print model stats
    '''
    def __init__(self):
        self.model = None
        self.model_list = 'rfc,rfr,abc,abr,gbc,gbr,knn,svc,svr,linr,logr,lasso,ridge'.split(',')

    def get_model_list(self):
        '''
        Return a list of the available model types to implement
        '''
        return self.model_list

    def set_model(self, model_name='rfc'):
        '''
        Set the sklearn model.  Defaults to RandomForestClassifier

        Input:
            model_name - short hand refernce to an sklearn model.
                Must be a valid element in model_list
        Return:
            sklearn model of the given name.  Parameter values are preset.
        TODO: configurable parameters
        '''
        model_name = model_name.lower()
        if model_name == 'rfc':
            model = RandomForestClassifier(
               n_estimators=500,
               max_depth=None,
               min_samples_leaf=10,
               max_features='log2',
               bootstrap=False
               #oob_score=True
               )
        elif model_name == 'rfr':
            model = RandomForestRegressor(
                n_estimators=500,
                n_jobs=-1,
                max_depth=None,
                max_features='auto',
                oob_score=True
                )
        elif model_name == 'abc':
            model = AdaBoostClassifier()
        elif model_name == 'abr':
            model = AdaBoostRegressor(
                n_estimators=500,
                random_state=0,
                learning_rate=0.1
                )
        elif model_name == 'gbc':
            model = GradientBoostingClassifier()
        elif model_name == 'gbr':
            model = GradientBoostingRegressor(
                n_estimators=500,
                random_state=0,
                learning_rate=0.1
            )
        elif model_name == 'knn':
            model = KNeighborsClassifier()
        elif model_name == 'svc':
            model = SVC()
        elif model_name == 'svr':
            model = SVR()
        elif model_name == 'linr':
            model = LinearRegression()
        elif model_name == 'logr':
            model = LogisticRegression()
        elif model_name == 'lasso':
            model = Lasso()
        elif model_name == 'ridge':
            model = Ridge()

        else:
            err = 'Invalid model specified: "%s".  Must be one of: %s' % (model_name,self.model_list)
            raise ValueError(err)

        self.model = model
        return model

    def get_model(self):
        '''
        Return this class' sklearn model
        '''
        return self.model

    def simple_data_split(self, X, y, test_set_size=100):
        '''
        Simple train/test set split where the tail end of the total set
        becomes the test set.

        Input:
            test_set_size (representing the last n bars of (price) data)
        Output:
            X_train, X_test, y_train, y_test

        - TODO: Could perform a query to order by date to ensure the df/array
                is date ordered.  This is currently presumed.
        '''
        X_train = X[0:-test_set_size]
        X_test = X[-test_set_size:]
        y_train = y[0:-test_set_size]
        y_test = y[-test_set_size:]

        return X_train, X_test, y_train, y_test

    def get_scores(self, y_true, y_pred, print_on=False):
        '''
        Print all score, including the sklearn classification report

        Input:
            y pred and true labels
        Output:
            Print all score values, and
            return a dict of the scores: key = metric name, value = score

        '''
        # relevant_metrics = [precision_score, recall_score, accuracy_score, roc_auc_score,
        #                     mean_squared_error, r2_score, f1_score]
        relevant_metrics = [precision_score, recall_score, accuracy_score, roc_auc_score, f1_score]
        # met = []
        met = {}
        for metric in relevant_metrics:
            try:
                m = metric(y_true, y_pred)
            except:
                m = 'pass'
            # met.append(m)
            met[metric.__name__] = m
            # scores[metric.__name__] = m
            if print_on:
                print metric.__name__, ' = ', m

        # Print out of the box classification_report
        if print_on:
            try:
                print classification_report(y_true, y_pred)
            except:
                print "Can't print classification report for regressor"

        # return the list of metric scores
        return met

    def standard_confusion_matrix(self, y_true, y_pred):
        '''
        Convert confusion matrix to standard format format.

        Input:
            y_true : ndarray - 1D
            y_pred : ndarray - 1D
        Output:
            ndarray - 2D
        '''
        try:
            [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
            return np.array([[tp, fp], [fn, tn]])
        except:
            return np.zeros((2,2))

    def print_standard_confusion_matrix(self, y_true, y_pred):
        '''
        Pretty printing wrapper for standard_confusion_matrix, in the form of:
                      -----------
                      | TP | FP |
                      -----------
                      | FN | TN |
                      -----------
        '''
        try:
            mat = self.standard_confusion_matrix(y_true, y_pred)
            print '              Actual'
            print '        ------------------'
            print '        | %s | %s |' % (mat[0][0], mat[0][1])
            print 'Predict ------------------'
            print '        | %s | %s |' % (mat[1][0], mat[1][1])
            print '        ------------------'
        except:
            print "Can't print confusion matrix for regressor"


    def print_feature_importance(self, model, features, num_feat=10):
        '''
        Print the top n features

        Input:
            model : the sklearn model (eg RandomForestClassifier),
            features : list of dataframe features to sort by importance
            num features : amount fo features for feature importance
        Return:
            None
        '''
        # Feature importances are only valid for classifiers
        try:
            importances = model.feature_importances_[:num_feat]
            sort_feat_impts = np.argsort(model.feature_importances_[:num_feat])
            features = [features[idx] for idx in sort_feat_impts]
            pairs = sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), features), reverse=True)
            for pair in pairs:
                print '%s: %s' % (pair[1], pair[0])
        except:
            print 'Model %s has no feature importance data' % model.__class__.__name__
            features = []


    def rmse(self, y, y_pred):
    	'''
        Compute Root-mean-squared-error
        '''
    	return np.sqrt(np.mean((y - y_pred) ** 2))

    def fit_predict_score_tscv(self, model, X_train, y_train, num_folds=5, print_on=False):
        '''
        Run a time series cross validation on the model prediction
        '''
        tscv = TimeSeriesSplit(n_splits=num_folds)
        error = np.empty(num_folds)

        index = 0
        all_scores = []
        for train_index, test_index in tscv.split(X_train):
            train = X_train[train_index]
            train_y = y_train[train_index]
            test = X_train[test_index]
            test_y = y_train[test_index]

            if print_on:
                print '===== Fitting model split %s =====' % (index+1)
            model.fit(train, train_y)
            pred = model.predict(test)
            error[index] = self.rmse(test_y, pred)
            if print_on:
                print 'train, test size: ', str(train_index.shape[0]) + ',', str(test_index.shape[0])
                print '- rmse: ', error[index]
            all_scores.append(self.get_scores(test_y, pred, print_on))

            if print_on:
                self.print_standard_confusion_matrix(test_y, pred)

            index += 1

        # score = cross_val_score(model, X_train, y_train, cv=tscv.split(X_train))
        # predict = cross_val_predict(model, X_train, y_train, cv=tscv.split(X_train))
        # print 'cross_val_score: ', score

        # return np.mean(error)
        return all_scores

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def fit_predict_score(self, print_on=False):
        '''
        fit the model and return scores
        '''
        #import pdb; pdb.set_trace()
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        error = self.rmse(y_test, y_pred)
        if print_on:
            print 'train, test size: ', str(X_train.shape[0]) + ',', str(X_test.shape[0])
            print '- rmse: ', error
        score = self.get_scores(y_test, y_pred, print_on)

        if print_on:
            self.print_standard_confusion_matrix(y_test, y_pred)

        return score

    def walk_forward_train(self, model, X, y, start_index=100, window_range=100):
        '''
        Perform walk forward testing on the input data.

        input:
        start_index:  the index to initiate the X data set with - 0 to index
        window_range: the step to walk forward

        **  I don't think there is any benefit for this since the model doen's
            incorporate the new data in an iterative way.
        '''
        # X = series.values
        n_records = len(X)
        #import pdb; pdb.set_trace()
        print '====== Starting walk forward: %s records ======' % n_records
        for i in range(start_index, n_records, window_range):
            print '--- index: %s' % i
            X_train, X_test = X[0:i], X[i:i+window_range]
            y_train, y_test = y[0:i], y[i:i+window_range]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            score = self.print_scores(y_test, y_pred)
            # print('train=%d, test=%d' % (len(X_train), len(X_test)))
            #print 'Cnt, Pred, True, Accuracy, Total: %s' % (i)#, y_pred[:-2:-1], y_test[:-2:-1])#, score, y_pred_proba)
            print '- score: %s, prob: %s' % (score, y_pred_proba)


################################################################################
# Class to support grid searching
################################################################################


class GridSearcher(object):
    '''
    This class provides services to support grid search on classifier models

    init configures the parameters for each model
    grid_search_reporter output the results of each models search
    grid_search runs the search for one model
    '''
    def __init__(self):
        '''
        Init with a list of (model, dict) tuples.
        The dictionary for each model has parameters specific to that model.
        '''
        gd_boost = {
            'learning_rate': [1, 0.05, 0.02, 0.01],
            'max_depth': [2, 4, 6],
            'max_features': ['sqrt', 'log2'],
            'n_estimators': [50, 100, 1000]
        }

        ada_boost = {
            'learning_rate': [1, 0.05, 0.02, 0.01],
            'base_estimator__max_depth': [2, 4, 6],
            'base_estimator__max_features': ['sqrt', 'log2'],
            'n_estimators': [50, 100, 1000]
        }

        decision_tree = {
            'max_depth': [2, 4, 6, 10],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [3, 5, 9, 17]
        }

        random_forest_grid = {
            'n_estimators': [50, 100, 1000],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 10, 50],
        }

        knn_grid = {
            'n_neighbors': [5, 10, 15],
            'weights': ['uniform', 'distance'],
        }

        svc_grid = {
            'C': [0.1, 1.0, 5],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3],
            'shrinking': [True, False],
            'gamma': [.02, .08, 1.5, 5]
        }

        lin_reg = {
            'fit_intercept':[True,False],
            'normalize':[True,False],
            'copy_X':[True, False]
        }

        self.models = [
            (GradientBoostingClassifier(), gd_boost),
            (AdaBoostClassifier(DecisionTreeClassifier()), ada_boost),
            (DecisionTreeClassifier(), decision_tree),
            (RandomForestClassifier(), random_forest_grid),
            (KNeighborsClassifier(), knn_grid),
            (SVC(), svc_grid)
        ]

    def grid_search_reporter(self, X_train, y_train):
        searches = []

        for model, feature_dict in self.models:
            print "Running grid search for {}".format(model.__class__.__name__)
            start_time = timeit.default_timer()
            gs = self.grid_search(model, feature_dict, X_train, y_train)

            print "====={}=====".format(gs.best_estimator_.__class__.__name__)
            print "  processing time: {} ".format(timeit.default_timer() - start_time)
            print "  f1 score: {}".format(gs.best_score_)
            print "  best params: {}".format(gs.best_params_)

            searches.append(gs)

        return searches

    def grid_search(self, model, feature_dict, X_train, y_train):
        gscv = GridSearchCV(
            model,
            feature_dict,
            n_jobs=-1,
            verbose=True,
            scoring='f1',
            iid=False,
            cv=TimeSeriesSplit(n_splits=5).split(X_train)
        )

        gscv.fit(X_train, y_train)
        return gscv
