# Plot utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestClassifier
from collections import OrderedDict


def plot_ts_chart(price, date):
    '''
    A Simple stock chart plot_ts_chart
    '''
    plt.figure(figsize=(12,6))
    plt.scatter(date, price)

# TODO
# ROC curve
# Profit curve? - sure, if I put the $ amount of the next day trade
# Calibration plot - http://scikit-learn.org/stable/modules/calibration.html
# - http://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html
# Lambda curve - for regularization

def plot_learning_curve(estimator, X, y, label=None):
    '''
    Plot learning curve with varying training sizes which shows
    the effect of larger training sizes

    * cross_vaidation deprecated
    * 3/3/indiv
    '''
    scores = list()
    train_sizes = np.linspace(10,100,10).astype(int)
    for train_size in train_sizes:
        cv_shuffle = cross_validation.ShuffleSplit(train_size=train_size,
                                                   test_size=200, n=len(y), random_state=0)
        test_error = cross_validation.cross_val_score(estimator, X, y, cv=cv_shuffle)
        scores.append(test_error)

    plt.plot(train_sizes, np.mean(scores, axis=1), label=label or estimator.__class__.__name__)
    plt.ylim(0,1)
    plt.title('Learning Curve')
    plt.ylabel('Explained variance on test set (R^2)')
    plt.xlabel('Training test size')
    plt.legend(loc='best')
    plt.show()


def plot_errors(X, y, X_train, y_train, X_test, y_test):
    '''
    Plot errors from test and training sets, from 1 to X size

    * 3/3/indiv
    '''
    m = X.shape[1]
    err_test, err_train = [], []
    linear = LinearRegression()
    for ind in xrange(m):
        linear.fit(X_train[:,:(ind+1)], y_train)

        train_pred = linear.predict(X_train[:,:(ind + 1)])
        test_pred = linear.predict(X_test[:,:(ind + 1)])

        err_test.append(np.sqrt(mse(test_pred, y_test)))
        err_train.append(np.sqrt(mse(train_pred, y_train)))

    x = range(1, m+1)
    plt.figure()
    plt.plot(x, np.log(err_test), label='log(Test error)')
    plt.plot(x, err_train, label='Training error')
    plt.title('Errors')
    plt.ylabel('RMSE')
    plt.xlabel('Features')
    plt.legend()
    plt.show()


def plot_alpha_rmse(model, X_train, y_train):
    '''
    Plot alphas/rmse to find the optimal alpha for regularization, either Ridge or Lasso

    We are looking to find the error between our fit model and the data both training and test.
    We will run a different RMSE calculation for each alpha, also for  both training and test.

    *3/3/pair
    '''
    alphas = np.logspace(-2, 2)
    rmse_train = []
    rmse_test = []
    for i, a in enumerate(alphas):
        m = model(alpha=a, normalize=True).fit(X_train, y_train)
        train_prediction = m.predict(X_train)
        test_prediction = m.predict(X_test)
        rmse_train.append(np.sqrt(mse(train_prediction, y_train)))
        rmse_test.append(np.sqrt(mse(test_prediction, y_test)))
    plt.plot(alphas, rmse_train, marker='.', linestyle='None', color='g', label='Train')
    plt.plot(alphas, rmse_test, marker='.', linestyle='None', color='r', label='Test')
    plt.legend()
    plt.xscale('log')
    plt.show()

    plt.plot(alphas, rmse_train, marker='.', linestyle='None', color='g', label='Train')
    plt.plot(alphas, rmse_test, marker='.', linestyle='None', color='r', label='Test')
    plt.legend()
    plt.xlim(0,1)
    plt.show()


def plot_oob_error(X, y, min_estimators=15, max_estimators=100):
    '''
    Plot the OOB error for given X, y data sets

    Input:  X and y data sets
    * Note: a large numer of estimators makes this function slow.
    '''
    RANDOM_STATE = 0
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(warm_start=True, oob_score=True,
                                   max_features="sqrt",
                                   random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(warm_start=True, max_features='log2',
                                   oob_score=True,
                                   random_state=RANDOM_STATE)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(warm_start=True, max_features=None,
                                   oob_score=True,
                                   random_state=RANDOM_STATE))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of n estimators values to explore.
    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, y)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()


def roc_curve(probabilities, labels):
    '''
    Input:  Take a numpy array of the predicted probabilities and a numpy array of the
    true labels.
    Output: Return the True Positive Rates, False Positive Rates and Thresholds for the
    ROC curve.
    '''

    thresholds = np.sort(probabilities)

    tprs = []
    fprs = []

    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases

    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)

        fprs.append(fpr)
        tprs.append(tpr)

    return tprs, fprs, thresholds.tolist()


def plot_roc_curve(probabilities, y_test):
    '''
    Plot the an ROC curve
    - Wrapper for roc_curve function
    '''
    tpr, fpr, thresholds = roc_curve(probabilities, y_test)

    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Sensitivity, Recall)")
    plt.title("ROC plot of stock data")
    plt.show()
