'''
Trading System - Composite
A composite of a variety of technical indicators.  A work in progress
to contiue experimentation with.
'''
# Author:  Matt Cohen
# Python Version 2.7

import numpy as np
from sklearn.preprocessing import Imputer
#from trading_system.trading_system import TradingSystem
import ta
from trading_system import TradingSystem
#from trading_system import ta
import talib


class TradingSystem_Comp(TradingSystem):

    def __init__(self):
        self.df = None      # X matrix features
        self.target = None  # y label target
        self.features = []  # X feature column names
        TradingSystem.__init__(self)

        super(TradingSystem_Comp, self).__init__(
            name='Trading System Composite')

    def preprocess(self, in_df):
        '''
        Build feature columns added to the input data frame.
        Includes creating features from technical indiator, and any data
        preprocessing steps such as normalizing, smoothing, remove
        correlated columns, etc

        Input:
            data : dataframe
                The source dataframe of symbol(s) data

        Output
            dataframe
                The modified dataframe with feature columns added
        '''
        df = in_df.copy()

        # Smoothed price series to be used in generating other
        # technical indicators
        high = df['high'].ewm(com=.8).mean()
        low = df['low'].ewm(com=.8).mean()
        close = df['close'].ewm(com=.8).mean()
        volume = df['volume'].astype(float)
        mean_log_close_5 = np.log(close.rolling(window=5).mean())
        '''
        # ATR indicators
        atr3 = talib.ATR(high.values, low.values, close.values, timeperiod=3)
        atr10 = talib.ATR(high.values, low.values, close.values, timeperiod=10)
        atr21 = talib.ATR(high.values, low.values, close.values, timeperiod=21)
        atr50 = talib.ATR(high.values, low.values, close.values, timeperiod=50)
        atr100_log = talib.ATR(high.values, low.values,
                               np.log(close).values, timeperiod=100)
        # df['atr14'] = talib.ATR(high.values, low.values, close.values,
        #                          timeperiod=14)
        df['ATRrat3'] = atr3 / atr21
        # df['ATRrat1050'] =  atr10 / atr50
        df['deltaATRrat33'] = df['ATRrat3'] - df['ATRrat3'].shift(3)
        # df['deltaATRrat310'] = df['ATRrat3'] - df['ATRrat3'].shift(10)
        df['apc5'] = mean_log_close_5 / atr100_log

        # Daily return ROC
        daily_return = close - close.shift(1)  # today - yesterday close
        daily_return[np.isnan(daily_return)] = 0
        df['log_daily_return'] = np.log(daily_return + 1 - min(daily_return))

        # Bollinger band indicators
        upperband, middleband, lowerband = talib.BBANDS(
                    close.values, timeperiod=3, nbdevup=2, nbdevdn=2, matype=0)
        df['bWidth3'] = upperband - lowerband
        upperband, middleband, lowerband = talib.BBANDS(
                    close.values, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bWidth20'] = upperband - lowerband
        df['deltabWidth33'] = df['bWidth3'] - np.roll(df['bWidth3'], 3)
        df['deltabWidth310'] = df['bWidth3'] - np.roll(df['bWidth3'], 10)

        closeMA10var = df.close.rolling(window=10).var()
        closeMA30var = df.close.rolling(window=30).var()
        df['price_var_ratio'] = closeMA10var / closeMA30var
        df['deltaPVR5'] = df['price_var_ratio'] - df['price_var_ratio'].shift(5)
        df['willr'] = talib.WILLR(high.values, low.values, close.values, timeperiod=14)
        df['obv'] = talib.OBV(close.values, volume.values)
        stok, df['stod'] = talib.STOCH(high.values, low.values, close.values, fastk_period=5, slowk_period=3,
                                             slowk_matype=0, slowd_period=3, slowd_matype=0)
        df['deltabWidth310'] = df['bWidth3'] - np.roll(df['bWidth3'], 10)
        #df['atr7'] = talib.ATR(high.values, low.values, close.values, timeperiod=7)
        df['ATRrat1020'] = talib.ATR(high.values, low.values, close.values, timeperiod=10) / talib.ATR(high.values, low.values, close.values, timeperiod=20)
        #df['ATRrat10100'] = talib.ATR(high.values, low.values, close.values, timeperiod=10) / talib.ATR(high.values, low.values, close.values, timeperiod=100)

        # Normalize all columns above but these defaults:
        cols = [col for col in df.columns if col not in self.excluded_features]
        def max_min_normalize(ndarr):
            x = (ndarr - ndarr.min())/( ndarr.max()-ndarr.min())
            return x[-1]
        for col in cols:
            df[col] = df[col].rolling(window=50).apply(max_min_normalize)

        # Non-normalized columns
        # stats
        #df['roc1'] = ta.rate_of_change(close, 1)  # highly correlated with daily returns
        #df['roc2'] = ta.rate_of_change(close, 2)
        df['roc5'] = ta.rate_of_change(close, 5)
        df['slope20'] = ta.liregslope(df['close'], 20)
        df['velocity'] = df['close'] + (df['slope20'] * df['close']) / 20
        df['stdClose20'] = df['close'].shift(1).rolling(window=20).std()
        df['zscore'] = (df['close'] - df['close'].shift(1).rolling(window=20).mean()) / df['stdClose20']
        # Oscilattors are by design already normalized
        #df['mom3'] = talib.MOM(close.values, timeperiod=3)
        #df['mom10'] = talib.MOM(close.values, timeperiod=10)  # 2 week momentum
        df['mom20'] = talib.MOM(close.values, timeperiod=20)  # 4 week momentum
        #df['mom10accel'] = df['mom10'] - df['mom10'].shift(4)  # 4 days diff
        df['mom20accel'] = df['mom20'] - df['mom20'].shift(4)  # 4 days diff
        # Price extremes
        df['HH20'] = (df['high'] > df['high'].shift(1).rolling(window=20).max()).astype(int)  # highest high in 4 weeks
        #df['HH5'] = (df['high'] > df['high'].shift(1).rolling(window=5).max()).astype(int)  # highest high in 1 week
        df['LL20'] = (df['low'] < df['low'].shift(1).rolling(window=20).min()).astype(int)  # highest high in 4 weeks
        df['LL5'] = (df['low'] < df['low'].shift(1).rolling(window=5).min()).astype(int)  # highest high in 1 week
        # Candle and volume Size
        vol30 = df['volume'].shift(1).rolling(window=30).mean()
        relVolSize = df['volume'] / vol30
        relVolSize[np.isnan(relVolSize)] = 0  # impute inf to 0
        '''
        #df['relVolSize'] = relVolSize
        cSize = (df['close'] - df['open']).abs()
        cSize30 = cSize.shift(1).rolling(window=30).mean()
        relCanSize = cSize / cSize30
        relCanSize[np.isnan(relCanSize)] = 0  # impute inf to 0
        df['relCanSize'] = relCanSize
        # MAs
        #df['sma5'] = talib.SMA(df['close'].values, 5)
        #df['sma20'] = talib.SMA(df['close'].values, 20)
        #df['sma50'] = talib.SMA(df['close'].values, 50)
        # Last n days
        df['twoDownDays'] = ((df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2))).astype(int)
        df['threeDownDays'] = ((df['close'] < df['close'].shift(1)) & (df['close'].shift(1) < df['close'].shift(2)) & (df['close'].shift(2) < df['close'].shift(3))).astype(int)

        # Impute - delete rows with Nan and null.  Will be the first several rows
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        #imp = imp.fit(X_train)
        for name in df.columns:
            df = df[df[name].notnull()]

        # ** Unused features **
        # df['log_close'] = np.log(close)
        # df = ta.run_exp_smooth(df, alpha=.5)
        # opn_sm = df['exp_smooth_open']
        # high_sm = df['exp_smooth_high']
        # low_sm = df['exp_smooth_low']
        # close_sm = df['exp_smooth_close']
        # volume_sm = df['exp_smooth_volume']
        # MA discrete series
        # df['MACurOver3'] = (df['close'] > df['close'].shift(1).rolling(window=3).mean()).astype(int)
        # df['MA3Over5'] = (df['close'].rolling(window=3).mean() > df['close'].shift(1).rolling(window=5).mean()).astype(int)
        # df['MA5Over10'] = (df['close'].rolling(window=5).mean() > df['close'].shift(1).rolling(window=10).mean()).astype(int)
        # df['MA5ver20'] = (df['close'].rolling(window=5).mean() > df['close'].shift(1).rolling(window=20).mean()).astype(int)
        # df['hurst'] = ta.hurst(close)
        # daily_p = history(bar_count=100, frequency='1d', field='price')
        # daily_ret = daily_p.pct_change()
        # daily_log = np.log1p(daily_ret)
        # daily_log_mean = pd.rolling_mean(daily_log, 5)
        # print daily_log_mean.tail(5)

        self.df = df

        # Now that we've calculated the features above,
        # save off the names to a list
        self.features = [col for col in df.columns
                         if col not in self.excluded_features]
        self._generate_target()

        return self.df

    def get_features(self):
        # ## For SPY ##
        # Oscilators
        # x_osc = ['rsi', 'cci', 'stod', 'stok', 'willr']
        # x_oscd_cols = ['rsi_d', 'cci_d', 'stod_d', 'stok_d', 'willr_d']
        # # MAs
        # x_ma_cols = ['sma20', 'sma50', 'sma200', 'wma10', 'macd_d']
        # x_all_dscrete_cols = ['roc_d', 'rsi_d', 'cci_d', 'stod_d',
        #                       'stok_d', 'willr_d', 'mom_d']
        # #x_cols = ['roc', 'rsi', 'willr', 'obv', 'stok']#'mom', ,
        #            'cci', 'stod', 'macd', 'sma', 'sma50', 'wma']
        # #x_cols = ['roc']
        # x_cols = x_all_dscrete_cols + x_ma_cols
        return self.features

    def set_features(self, features):
        '''
        Set the features for the tradng system to be used by the model

        input:  list of features, column names
        '''
        self.features = features

    # def feature_forensics(self, model):
    #     return TradingSystem.feature_forensics(self, model)

    def _generate_target(self):
        '''
        The true label, usually a price change of +1 day - current_price,
        but can be whatever the trading system calls for.
        The model prediction will be compared to this.

        **** CAVEAT ****
        If target is price change over n days, you need to shift the y target
        by n days (at least one day) to ensure no future leak.

        Returns a dataframe with the y label column, ready to use in a
        model for fit and predict.
        '''
        if self.df is None:
            print 'This trading system has no data. Call preprocess_data first.'
            return

        # Target as a one-day price change
        days_ahead = -1
        gain_loss = np.roll(self.df['close'], days_ahead) - self.df['close']
        self.df['y_true'] = (gain_loss >= 0).astype(int)

        # Drop the last row becaue of the shift by 1, as there is no future
        # price change to compare against (or similarly drop the last n rows)
        # TODO: probably needs to change to a better shift
        self.df = self.df[:-1]

        return self.df
