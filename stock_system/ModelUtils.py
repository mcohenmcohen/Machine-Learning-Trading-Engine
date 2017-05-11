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
