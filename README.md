# Machine Learning Trading Engine
This application provides a python based backtesting framework to execute trading systems using various machine learning models.

The code base is comprised of these three modules:
* data
  * loader.py - to retrieve real time and historical data.  There are two resources avaiable: pandas_datareader for yahoo data, and DTN IQFeed, which requires you provide your own connection.  On a Mac, I run this outside of python via wine windows emulator.
  * data.py - The contains the database I/O to PostgreSQL.  
  * A sql schema file is provided to build the tables.  Edit as your environment and taste dictates.
* trading_system
  * This contains various trading systems.  Trading systems classes define the entry and exit criteria.   ts_composite is a good sandbox for trying a variety of technical analysis indicators and is an example for the process flow.  A parent class provides some helpful methods, you provide the subclass.  
* core
  * Model classes implement models and associated hyperparameter settings.  model.py provides a variety of utilities (note: this is soon to be refactored).
  * backtester.py provides the engine to run your trading system on the selected stock over an n-day forecast period.  More features are to come.
  * A feature engineering module to identify important and collinear features, including helper functions are to run correlation, covariance/collinearity analysis, MIC and RFE.
  * Feature importance and print functions for regression and tree models
  * A number of plotting functions, I find more easily used at this point with a notebook.  This may take some tweaking.
  * An accounting module to calculate profit/loss scenarios (very, very thin right now and will most certainly be replaced.

A good place to start is by looking at pipeline.py, which runs the flow from indicating symbols, selecting a trading system and model to run, and backtesting through accounting.

Perform the following to set up your local environment:
- data/stockdb.sql: Create the symbol table.
- data/loader.py: Retrieve data from an online source.  pandas_datareader daily data is the current option unless you have DTN IQ feed.  In which case, run your IQ Feed plugin before executing the loader with that data source.
- core/feature_forensics: The main function runs the automated feature engineering processes that generates a csv file with the top features for the given trading system and symbol to backtest.  The csv is used in the pipeline at runtime.
- core/pipeline:  Now you're setup to run the pipeline.  Just select your trading system and model and you should be good to go.

Known issues:
- The system works with classifiers. I have yet to flush out regressors, so those models may fail.
- The neural network trading system is a work in progress.

Caveat: This is a work-in-progress and as such is prone to potential bugs and sometimes large changes to the architecture of the code base.  

Have fun, comments and feedback are welcome.


