# Plot utilities
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.metrics import mean_squared_error as mse


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
