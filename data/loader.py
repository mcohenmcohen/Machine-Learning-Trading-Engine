import socket
import pandas as pd
import numpy as np
# from stock_system.data import DataUtils
# from stock_system.iqfeed import IQFeed
from data.dataio import DataUtils
from data.iqfeed import IQFeed
import sys, os, datetime


def get_data():
    '''
    Get symbol data from an online data source and return a dataframe
    '''
    usage = 'Usage: \n' \
        '  -d  : Datasource as \'iqf\' or \'pdr\'\n' \
        '  -s  : (Optional) Comma separated string of symbols\n' \
        '  -sd : Start date, format yyyymmdd\n' \
        '  -ed : Optional end date, format yyyymmdd.  If not end date ' \
        'then data from start date to current are retrieved.\n' \
        'Example in python shell: \n' \
        '  run loader.py -d pdr -s \'AAPL,MSFT\' -sd \'20050101\''
    if len(sys.argv) == 1:
        print usage
        return
    elif len(sys.argv[1:]) % 2 != 0:
        print usage

        return

    arg_dict = {x[0] : x[1] for x in zip(sys.argv[1::2], sys.argv[2::2])}
    arg_source = ''
    arg_start_date = ''
    arg_end_date = ''
    syms = []
    for arg in arg_dict:
        if '-d' not in arg_dict:
                print 'Must provide a valid data source to download symbols.'
                print usage
                return
        elif arg == '-d':
            valid_sources = ['iqf', 'pdr']
            arg_source = arg_dict['-d']
            if arg_source not in valid_sources:
                print 'Data source %s is not valid.  Must be one of: %s' % (arg_source, ', '.join(valid_sources))
                print usage
                return
        if '-sd' not in arg_dict:
                print 'Must provide at least a valid start date to download symbols.'
                print usage
                return
        elif arg == '-sd':
            arg_start_date = arg_dict['-sd']
            if len(arg_start_date) != 8 or arg_start_date.isdigit() == False:
                print 'Start date must be \'yyyymmdd\''
                print usage
                return
        if arg == '-ed':
            arg_end_date = arg_dict['-ed']
            if len(arg_end_date) != 8 & arg_end_date.isdigit() == False:
                print 'End date must be \'yyyymmdd\''
                print usage
                return
        if arg == '-s':
            arg_syms = arg_dict['-s']
            if len(arg_syms) == 0:
                print 'Symbols must be comma separated string'
                print usage
                return
            else:
                syms = arg_syms.split(',')


    ## Here's where you'd assign symbols to download if you override the command line
    #syms = pd.read_csv('_data/Top_vol_and_weeklies.csv')['Symbol'].tolist()
    # syms = ['AAPL', 'MSFT']

    utils = DataUtils()

    if arg_source == 'pdr':
        arg_start_date = arg_start_date[0:4] + '-' + arg_start_date[4:6] + '-' + arg_start_date[6:]
        if len(arg_end_date) > 0:
            df = utils.get_data_pdr_yahoo(syms, arg_start_date, end_date=arg_end_date, dbwrite=True)
        else:
            df = utils.get_data_pdr_yahoo(syms, arg_start_date, dbwrite=True)
        return df
    elif arg_source == 'iqf':
        df = utils.get_data_iqfeed(syms, '20140101', dbwrite=True)
        return df
    else:
        return 'No action'

if __name__ == '__main__':
    df = get_data()
