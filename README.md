# Machine Learning Trading Engine
This application provides a python based backtesting framework to execute trading systems using various machine learning models.

The code base is essentially comprised of these core components:
* A set of data access tools to retrieve stock data          
  * data.py - to retrieve real time and historical data from IQFeed, you provide your own connection outside of python (via wine on a Mac)
  * iqfeed.py - for PostgreSQL.  A schema file is provided to build the tables.  Edit as your environment and taste dictates.
* A modular framework for trading systems and models
  * Trading systems classes define the entry and exit criteria.  A parent class provides some helpful methods.  You provide the subclass.  ts_comp is a trading system that implements a wide variety of technical analysis indicators as an example for the process flow.
  * Model classes implement models and associated hyperparameter settings.  model.py provides a variety of utilities (note: this is soon to be refactored.
* A feature engineering module to identify important and collinear features 
  * helper functions are provided to run correlation, covariance/collinearity analysis, MIC and RFE.
  * Feature importance and pretty print functions for regression and tree models
* A number of plotting functions, I find more easily used at this point with a notebook.  This may take some tweaking.
* An accounting module to calculate profit/loss scenarios (very, very thin right now and will most certainly be replaced.

A good place to start is by looking at pipeline.py, which runs the flow from indicating symbols to backtest on and selecting a trading system and model to run.

Then perform the following to set up your local environment
- database/stockdb.sql: Create the symbol table.
- database/loader.py: Retrieve data from an online source.  pandas_datareader daily data is the current option unless you have DTN IQ feed.  In which case, run your IQ Feed plugin before executing the loader with that data source.
- stock_system/feature_forensics: The main function runs the automated feature engineering processes that generates a csv file with the top features for the given trading system and symbol to backtest.  The csv is used in the pipeline at runtime.
- stock_system/pipeline:  Now you're setup to run the pipeline.  Just select your trading system and model and you should be good to go!

Known issues:
- The system works for classifiers. I have yet to flush out regressors, so those models may fail.
- The neural network trading system is a work in progress.

_** PS: This is a work-in-progress and as such is prone to potential bugs and sometimes large changes to the architecture of the code base._


