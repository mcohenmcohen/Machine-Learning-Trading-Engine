###################################################################################################
# Accounting
#
# Utilities for calculating the gain or loss from making or missing tades
# based on model signal outputs
###################################################################################################

from itertools import izip
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings


def get_profit_confusion_matrix_df(df):
    '''
    Calculations using close price:
        For a dialy return, the prediction operates on the data available that day, and
        sets the y_pred in that same row, which indicates up/or down the following day.
        Thus, for accounting we assume that we follow the prediction to make a purchase
        that day prior to the close, and we sell the following day prior to the close.

    Input:
        dataframe.  Must include the columns 'close', 'y_true', 'y_pred'
    Output:
        dataframe of gain and loss for each of tp, tn, fp, fn
    '''
    # True daily return: Today close - yesterday close
    df['daily_ret'] = df['close'] - df['close'].shift(1)
    # Predicted daily return: tomorrow close - today close
    df['daily_ret_pred'] = df['close'].shift(-1) - df['close']
    # df['daily_ret_pred'] = np.roll(df['close'], -1) - df['close']

    df['tp'] = ((df['y_true'] == 1) & (df['y_pred'] == 1)).astype(int)
    df['fp'] = ((df['y_true'] == 0) & (df['y_pred'] == 1)).astype(int)
    df['tn'] = ((df['y_true'] == 0) & (df['y_pred'] == 0)).astype(int)
    df['fn'] = ((df['y_true'] == 1) & (df['y_pred'] == 0)).astype(int)

    confusion_matrix = df[['tp', 'fp', 'tn', 'fn']].sum()

    tp_total = df.groupby('tp').daily_ret_pred.sum()[1]
    fp_total = df.groupby('fp').daily_ret_pred.sum()[1]
    tn_total = df.groupby('tn').daily_ret_pred.sum()[1]
    fn_total = df.groupby('fn').daily_ret_pred.sum()[1]

    acct_mat = confusion_matrix.to_frame('Count')
    acct_mat['Amount'] = np.array([tp_total, fp_total, tn_total, fn_total])

    return acct_mat


def cacl_running_returns(df):
    '''
    Calculate a cumulative sum of returns for each buy signal

    This formula is valid for categorical y labels of 1 and 0
    '''
    df['running_ret_pred'] = np.cumsum(df['y_pred'] * df['daily_ret_pred'])
    df['running_ret_true'] = np.cumsum(df['daily_ret'])

def plot_equity_curve_compare(in_df):
    # Plot two charts to assess trades and equity curve
    # https://www.quantstart.com/articles/Backtesting-a-Moving-Average-Crossover-in-Python-with-pandas
    df = in_df.copy()
    fig = plt.figure(figsize=(20,10))
    fig.patch.set_facecolor('white')     # Set the outer colour to white
    symbol = df['symbol'][0]
    ax1_ylabel_text = '%s Price in $' % symbol
    ax1 = fig.add_subplot(211,  ylabel=ax1_ylabel_text)

    # Plot the base ticker closing price overlaid with the moving averages
    df['close'].plot(ax=ax1, color='r', lw=2., label=symbol)
    # df[['sma5', 'sma20']].plot(ax=ax1, lw=2.)

    # Plot the "buy" trades against AAPL
    # ax1.plot(df[df['y_pred'] == 1.0])
    ax1.plot(df[df['y_pred'] == 1.0].index,
             df['sma5'][df['y_pred'] == 1.0],
             '^', markersize=7, color='m', label='Buy signals')
    lines, labels = ax1.get_legend_handles_labels()
    ax1.legend(lines[:2], labels[:2], loc='best', fontsize=14)  # legend for first two lines only


    # # Plot the "sell" trades against AAPL
    # ax1.plot(signals.ix[signals.positions == -1.0].index,
    #          signals.short_mavg[signals.positions == -1.0],
    #          'v', markersize=10, color='k')

    # Plot the equity curve in dollars
    ax2 = fig.add_subplot(212, ylabel='Predict vs True portfolio value in $')
    #df[['running_ret_true','running_ret_pred']].plot(ax=ax2, color='r', lw=2.)
    df['running_ret_true'].plot(ax=ax2, color='r', lw=2., label='True')
    df['running_ret_pred'].plot(ax=ax2, lw=2., label='Predicted')
    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines[:2], labels[:2], loc='best', fontsize=14)  # legend for first two lines only

    # Plot the "buy" and "sell" trades against the equity curve
    # ax2.plot(df[df['y_pred'] == 1.0])  # , '^', markersize=10, color='m'
    # ax2.plot(df[df['y_pred'] == -1.0].index,
    #          df[df['y_pred'] == -1.0],
    #          'v', markersize=10, color='k')

    # Might need to put this bit in the python notebook
    import matplotlib
    font = {'family' : 'normal',
            'weight' : 'bold',
            'size'   : 18}
    matplotlib.rc('xtick', labelsize=10)
    matplotlib.rc('ytick', labelsize=10)
    matplotlib.rc('font', **font)

    # Plot the figure
    fig.show()


def annualised_sharpe(returns, N=252):
    '''
    Calculate the annualised Sharpe ratio of a returns stream
    based on a number of trading periods, N. N defaults to 252,
    which then assumes a stream of daily returns.

    The function assumes that the returns are the excess of
    those compared to a benchmark.

    Eg,
        # Use the percentage change method to easily calculate daily returns
        df['daily_ret'] = df['adj_close'].pct_change()

        # Assume an average annual risk-free rate over the period of 5%
        df['excess_daily_ret'] = df['daily_ret'] - 0.05/252

        # Return the annualised Sharpe ratio based on the excess daily returns
        return annualised_sharpe(df['excess_daily_ret'])
    '''
    return np.sqrt(N) * returns.mean() / returns.std()


def get_cost_benefit_matrix(self, y_true, y_pred, gain_loss):
    '''
    Generate a cost_benefit matrix from mean total values associated with each of profit from.

    Input:  dataframe with columns 'y_true', 'y_pred', 'gain_loss'

    Calculations are based on the gain or loss difference between the price before and after
    a trade signal.
    '''
    # Convert confusion matrix to standard format format:
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    std_matrix = np.array([[tp, fp], [fn, tn]])


def profit_curve(cost_benefit, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels.

    Parameters
    ----------
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    predicted_probs : ndarray - 1D, predicted probability for each datapoint
                                    in labels, in range [0, 1]
    labels          : ndarray - 1D, true label of datapoints, 0 or 1

    Returns
    -------
    profits    : ndarray - 1D
    thresholds : ndarray - 1D
    """
    n_obs = float(len(labels))
    # Make sure that 1 is going to be one of our thresholds
    maybe_one = [] if 1 in predicted_probs else [1]
    thresholds = maybe_one + sorted(predicted_probs, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)


def get_model_profits(model, cost_benefit, X_train, X_test, y_train, y_test):
    """Fits passed model on training data and calculates profit from cost-benefit
    matrix at each probability threshold.

    Parameters
    ----------
    model           : sklearn model - need to implement fit and predict
    cost_benefit    : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    X_train         : ndarray - 2D
    X_test          : ndarray - 2D
    y_train         : ndarray - 1D
    y_test          : ndarray - 1D

    Returns
    -------
    model_profits : model, profits, thresholds
    """
    model.fit(X_train, y_train)
    predicted_probs = model.predict_proba(X_test)[:, 1]
    profits, thresholds = profit_curve(cost_benefit, predicted_probs, y_test)

    return profits, thresholds


def plot_model_profits(model_profits, save_path=None):
    """Plotting function to compare profit curves of different models.

    Parameters
    ----------
    model_profits : list((model, profits, thresholds))
    save_path     : str, file path to save the plot to. If provided plot will be
                         saved and not shown.
    """
    for model, profits, _ in model_profits:
        percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(percentages, profits, label=model.__class__.__name__)

    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def find_best_threshold(model_profits):
    """Find model-threshold combo that yields highest profit.

    Parameters
    ----------
    model_profits : list((model, profits, thresholds))

    Returns
    -------
    max_model     : str
    max_threshold : float
    max_profit    : float
    """
    max_model = None
    max_threshold = None
    max_profit = None
    for model, profits, thresholds in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
    return max_model, max_threshold, max_profit


def profit_curve_main(filepath, cost_benefit):
    """Main function to test profit curve code.

    Parameters
    ----------
    filepath     : str - path to find churn.csv
    cost_benefit  : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    """
    X_train, X_test, y_train, y_test = get_train_test(filepath)
    models = [RF(), LR(), GBC(), SVC(probability=True)]
    model_profits = []
    for model in models:
        profits, thresholds = get_model_profits(model, cost_benefit,
                                                X_train, X_test,
                                                y_train, y_test)
        model_profits.append((model, profits, thresholds))
    plot_model_profits(model_profits)
    max_model, max_thresh, max_profit = find_best_threshold(model_profits)
    max_labeled_positives = max_model.predict_proba(X_test) >= max_thresh
    proportion_positives = max_labeled_positives.mean()
    reporting_string = ('Best model:\t\t{}\n'
                        'Best threshold:\t\t{:.2f}\n'
                        'Resulting profit:\t{}\n'
                        'Proportion positives:\t{:.2f}')
    print reporting_string.format(max_model.__class__.__name__, max_thresh,
                                  max_profit, proportion_positives)
