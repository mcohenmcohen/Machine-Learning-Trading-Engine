# Machine Learning Trading Engine
This application provides a python based backtesting framework to execute trading systems using various machine learning models.

The code base is essentially comprised of these core components:
* A set of data access tools to retrieve stock data          
  * data.py - to retrieve real time and historical data from IQFeed, you provide your own connection outside of python (via wine on a Mac)
  * iqfeed.py - for PostgreSQL.  A schema file is provided to build the tables.  Edit as your environment and taste dictates.
* A modular framework for trading systems and models
  * Trading systems classes define the entry and exit criteria.  A parent class provides some helpful methods.  You provide the subclass.  ts_comp is a trading system that implements a wide variety of technical analysis indicators as an example for the process flow.
  * Model classes implement models and associated hyperparameter settings.  model.py provides a variety of utilities (note: this is soon to be refactored) 
* A feature engineering module to identify important and collinear features 
  * helper functions are provided to run correlation, covariance/collinearity analysis, MIC and RFE.
  * Feature importance and pretty print functions for regression and tree models
* A number of plotting functions, I find more easily used at this point with a notebook.  This may take some tweaking.
* An accounting module to calculate profit/loss scenarios (very, very thin right now and will most certainly be replaced.

Start by looking at pipeline.py to understand the backtesting process flow, iqfeed.py for the feed, and data.py for database and schema info.

_** PS: This is a work-in-progress and as such is prone to potential bugs and sometimes large changes to the architecture of the code base._


