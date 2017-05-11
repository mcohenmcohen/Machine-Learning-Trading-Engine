# ModelUtils
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Ridge, Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_val_predict


class ModelUtils(object):

    def __init__(self):
        self.models = []

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

    def print_scores(self, y_true, y_pred):
        '''
        Output all score types for a prediction vs actual y values
        '''
        relevant_metrics = [precision_score, recall_score, accuracy_score, roc_auc_score,
                            mean_squared_error, r2_score, f1_score]
        for metric in relevant_metrics:
            m = metric(y_true, y_pred)
            # scores[metric.__name__] = m
            print metric.__name__, ' = ', m

        # Print out of the box classification_report
        print classification_report(y_true, y_pred)

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
        mat = self.standard_confusion_matrix(y_true, y_pred)
        print '              Actual'
        print '        ------------------'
        print '        | %s | %s |' % (mat[0][0], mat[0][1])
        print 'Predict ------------------'
        print '        | %s | %s |' % (mat[1][0], mat[1][1])
        print '        ------------------'

    def rmse(self, y, y_pred):
    	'''
        Compute Root-mean-squared-error
        '''
    	return np.sqrt(np.mean((y - y_pred) ** 2))

    def predict_tscv(self, model, X_train, y_train, num_folds=5):
        '''
        Run a time series cross validation on the model prediction
        '''
        tscv = TimeSeriesSplit(n_splits=num_folds)
        error = np.empty(num_folds)

        index = 0
        for train_index, test_index in tscv.split(X_train):
            train = X_train[train_index]
            train_y = y_train[train_index]
            test = X_train[test_index]
            test_y = y_train[test_index]

            print '===== Fitting model split %s =====' % (index+1)
            model.fit(train, train_y)
            pred = model.predict(test)
            error[index] = self.rmse(test_y, pred)

            print 'train, test size: ', str(train_index.shape[0]) + ',', str(test_index.shape[0])
            print '- rmse: ', error[index]
            self.print_scores(test_y, pred)
            self.print_standard_confusion_matrix(test_y, pred)

            index += 1

        # score = cross_val_score(model, X_train, y_train, cv=tscv.split(X_train))
        # predict = cross_val_predict(model, X_train, y_train, cv=tscv.split(X_train))
        # print 'cross_val_score: ', score

        return np.mean(error)

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
