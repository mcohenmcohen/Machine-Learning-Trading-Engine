###################################################################################################
# Class provides functions to fit, cross validate, predict, score and print model stats
###################################################################################################

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict


class ModelUtils(object):
    '''
    Class provides functions to fit, cross validate, predict, score and print model stats
    '''
    def __init__(self):
        self.model_list = 'rfc,rfr,abr,gbr,knn,svc,svr,linr,logr,lasso,ridge'.split(',')

    def get_model_list(self):
        return self.model_list

    def get_model(self, model_name='rf'):
        '''
        Return a model of the given name.  Parameter values are preset.
        TODO: configurable parameters
        '''
        model_name = model_name.lower()
        if model_name == 'rfc' or model_name == 'rfc':
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
        elif model_name == 'abr':
            model = AdaBoostRegressor(
                n_estimators=500,
                random_state=0,
                learning_rate=0.1
                )
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
            err = 'Invalid model specified.  Must be one of: %s' % self.model_list
            raise ValueError(err)

        return model

    def simple_data_split(self, X, y, test_set_size=100):
        '''
        Simple train/test set split where the tail end of the total set becomes the test set.

        Input:  test_set_size (representing the last n bars of (price) data)
        Output: X_train, X_test, y_train, y_test

        - TODO: Could perform a query to order by date to ensure the df/array is date ordered.
                This is currently presumed.
        '''
        # from sklearn.cross_validation import PredefinedSplit
        #
        # X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        # y = np.array([0, 0, 1, 1])
        # test_fold = [0, 0, 1]
        # ps = PredefinedSplit(test_fold=test_fold, )
        #
        # for train_index, test_index in ps:
        #     print("TRAIN:", train_index, "TEST:", test_index)
        X_train = X[0:-test_set_size]
        X_test = X[-test_set_size:]
        y_train = y[0:-test_set_size]
        y_test = y[-test_set_size:]

        return X_train, X_test, y_train, y_test

    def get_scores(self, y_true, y_pred, print_on=False):
        '''
        Print all score, including the sklearn classification report

        Input:  y pred and true labels
        Output: Print all score values
                and return a dict of the scores: key = metric name, value = score

        '''
        relevant_metrics = [precision_score, recall_score, accuracy_score, roc_auc_score,
                            mean_squared_error, r2_score, f1_score]
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
        Convert confusion matrix to standard format format:
        Input:
            y_true : ndarray - 1D
            y_pred : ndarray - 1D
        Output:
            ndarray - 2D
        '''
        [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
        return np.array([[tp, fp], [fn, tn]])

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


    def print_feature_importance(self, rf, df, n=10):
        '''
        Print the top n features

        Input:  the random foreset model, dataframe of test data and columns, num features
        '''
        importances = rf.feature_importances_[:n]
        features = list(df.columns[np.argsort(rf.feature_importances_[:n])])

        print features
        print np.sort(rf.feature_importances_)[::-1]

    def rmse(self, y, y_pred):
    	'''
        Compute Root-mean-squared-error
        '''
    	return np.sqrt(np.mean((y - y_pred) ** 2))

    def predict_tscv(self, model, X_train, y_train, num_folds=5, print_on=False):
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
