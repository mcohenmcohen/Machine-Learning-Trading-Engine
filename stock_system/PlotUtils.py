###################################################################################################
# Utilities to assist in data plotting
###################################################################################################

import numpy as np
import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
from collections import OrderedDict


def plot_ts_chart(in_df, price_series='close'):
    '''
    A Simple stock chart plot_ts_chart
    '''
    df = in_df.copy()
    plt.figure(figsize=(12,6))
    plt.title(df['symbol'][0])
    plt.scatter(df.index, df[price_series])
    plt.show()

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


def plot_metric_scores(all_scores):
    '''
    Plot metric scores derived from model predictions.
    X Axis is the number of days the y target is from the current price
    '''
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    axs = axs.ravel()

    names = all_scores[0].keys()
    i = 0
    for score_name in names:
        scores = [d[score_name] for d in all_scores]
        axs[i].plot(scores, label=score_name)
        axs[i].legend(loc='best', fontsize=10)
        i += 1
    #plt.title(symbol)
    plt.xlabel('Num Days Ahead For Prediction')

    plt.show()


def plot_calibration_curve(est, name, fig_index, X_train, X_test, y_train, y_test, cv='prefit'):
    '''
    Plot calibration curve for est w/o and with calibration.

    Inputs:
    - est - the model
    - name - the model name
    - fig_index - which figure to plot it in
    - cv - the cross-validation strategy
           The stock models will be fitted already and are applicable to 'prefit'
           Integer values are the number of folds

    e.g.,
        # Plot calibration curve for Gaussian Naive Bayes
        plot_calibration_curve(GaussianNB(), "Naive Bayes", 1)

        # Plot calibration curve for Linear SVC
        plot_calibration_curve(LinearSVC(), "SVC", 2)

    '''
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=cv, method='isotonic')

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(est, cv=cv, method='sigmoid')

    # Logistic regression with no calibration as baseline
    lr = LogisticRegression(C=1., solver='lbfgs')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(lr, 'Logistic'),
                      (est, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        # if name == 'Logistic':
        #     clf.fit(X_train, y_train)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        #import pdb; pdb.set_trace()
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=1)
        #clf_score = brier_score_loss(y_test, prob_pos)
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()

    plt.show()


def plot_equity_curve_compare(in_df):
    df = in_df.copy()
    # Plot two charts to assess trades and equity curve
    # https://www.quantstart.com/articles/Backtesting-a-Moving-Average-Crossover-in-Python-with-pandas
    font = {'family': 'normal', 'weight': 'bold', 'size': 18}
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    matplotlib.rc('font', **font)
    plt.rc('axes', grid=True)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)

    fig = plt.figure(figsize=(20,10))
    fig.patch.set_facecolor('white')     # Set the outer colour to white
    symbol = df['symbol'][0]
    ax1_ylabel_text = '%s Price in $' % symbol
    ax1 = fig.add_subplot(211,  ylabel=ax1_ylabel_text)

    # Plot the base ticker closing price overlaid with the moving averages
    label = '%s price' % symbol
    df['close'].plot(ax=ax1, color='r', lw=2., label=label)

    # Plot the "buy" trades against AAPL
    ax1.plot(df[df['y_pred'] == 1.0].index,
             df['sma5'][df['y_pred'] == 1.0],
             '^', markersize=7, color='m', label='Buy signals')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines[:2], labels[:2], loc='best', fontsize=14)  # legend for first two lines only

    # # Plot the "sell" trades against symbol
    # ax1.plot(signals.ix[signals.positions == -1.0].index,
    #          signals.short_mavg[signals.positions == -1.0],
    #          'v', markersize=10, color='k')

    # Plot the equity curve vs symbol
    ax2 = fig.add_subplot(212, ylabel='Predict vs True portfolio value in $')
    df['running_ret_true'].plot(ax=ax2, color='r', lw=2., label=label)
    df['running_ret_pred'].plot(ax=ax2, lw=2., label='Predicted results')
    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines[:2], labels[:2], loc='best', fontsize=14)  # legend for first two lines only
    ax2.set_xlabel('')

    # Plot the "buy" and "sell" trades against the equity curve
    # ax2.plot(df[df['y_pred'] == 1.0])  # , '^', markersize=10, color='m'
    # ax2.plot(df[df['y_pred'] == -1.0].index,
    #          df[df['y_pred'] == -1.0],
    #          'v', markersize=10, color='k')

    # Might need to put this bit in the python notebook
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 18}
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    matplotlib.rc('font', **font)

    # Plot the figure
    fig.show()


def plot_returns_hist(returns, symbol=''):
    '''
    Print the raw returns as a histogram overlaid with a norma distribution

    input:  pd.Series or np.array
    '''
    #plt.rc('axes', grid=False)
    #plt.plot(density())

    # N = 100
    mn = returns.min()
    mx = returns.max()
    # x = np.random.randint(mn, mx, N)
    # bins = np.arange(10)

    mean = np.mean(returns.tolist())
    variance = np.var(returns.tolist())
    sigma = np.sqrt(variance)
    x = np.linspace(mn, mx,100)
    plt.plot(x,mlab.normpdf(x,mean,sigma), label='Norm', linewidth=2, color='r')
    label = 'Daily price change for %s' % symbol
    plt.hist(returns, bins=100, normed=True, label=label)

    # kde = stats.gaussian_kde(x)
    # xx = np.linspace(mn,mx,1000)
    # plt.plot(xx,kde(xx))
    plt.title('Predicted Return Values')
    plt.legend(loc='best', fontsize=8)
    plt.show()
