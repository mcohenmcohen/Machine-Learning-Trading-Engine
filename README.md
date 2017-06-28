# Machine Learning Trading Engine
This appliation provides a python based framework to execute trading systems built using machine learning models.

The code base has these core components:
* A set of data access tools to retrieve stocks data          
  * Code is provided to retrieve real time and historical data from IQFeed.  
  * Database code is for PostgreSQL and a schema file is provided to build tables.  Edit as your environment dictates.
* A modular framework for trading systems and models
  * Trading systems classes define the entry and exit criteria
  * Model classes implement models and associated hyperparameter settings
* A feature engineering module to identify important and colinear features 
  * helper functions are provided to run correlation and covariance matricies.
  * Feature importance pretty print functions for regression and tree models
* A number of plotting functions
* An accounting module to calculate profit/loss scenarios (very thin right now)

_**This is a work-in-progress and as such is prone to potential bugs and sometimes large changes to the architecture of the code base._
