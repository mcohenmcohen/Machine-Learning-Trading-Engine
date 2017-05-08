# https://www.quantstart.com/articles/Downloading-Historical-Intraday-US-Equities-From-DTN-IQFeed-with-Python
# iqfeed.py
import sys
import socket
import pandas as pd
import numpy as np
import timeit
from StringIO import StringIO  # For python 2.  For python 3 import from io


class DataFeed(object):

    def __init__(self):
        self.sock = None
        self.symbols = []

    def _get_data(self, sym, start_date='20170101', host="127.0.0.1", port=9100):

        # message = "HIT,%s,60,20030101 075000,,,093000,160000,1\n" % sym
        message = "HIT,%s,60,%s 075000,,,093000,160000,1\n" % (sym, start_date)

        # Open a streaming socket to the IQFeed server locally
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # Send the historical data request
        # message and buffer the data
        sock.sendall(message)
        self.sock = sock
        # print min(timeit.Timer('a=s[:]; timsort(a)', setup=setup)
        data = self._read_historical_data_socket()
        # timeit.Timer(self._read_historical_data_socket()).timeit()
        sock.close

        return data

    def get_data_hist(self, sym, start_date='20170101', host="127.0.0.1", port=9100):
        '''
        Wrapper for get_data
        Input:  Symbol to lookup, start date
        Output: A dataframe of the symbol data
        '''
        data = self._get_data(sym, start_date)

        # Remove all the endlines and line-ending
        # comma delimiter from each record
        data = "".join(data.split("\r"))
        data = data.replace(",\n", "\n")[:-1]

        cols = ['date', 'open', 'low', 'high', 'close', 'volume', 'open_interest']
        df = pd.read_csv(StringIO(data), names=cols)
        df.insert(0, 'symbol', sym)
        # df = pd.read_csv('AAPL.csv', names=cols)

        return df

    def get_data_stream(self, sym, host="127.0.0.1", port=5009, recv_buffer=4096):
        # message = "HIT,%s,60,20030101 075000,,,093000,160000,1\n" % sym
        message = "w%s\n" % sym

        # Open a streaming socket to the IQFeed server locally
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # Send the historical data request
        # message and buffer the data
        header = 'S,SELECT UPDATE FIELDS,Last,Percent Change,Change,Symbol\n'
        sock.sendall(header)
        sock.sendall(message)
        self.sock = sock
        # print min(timeit.Timer('a=s[:]; timsort(a)', setup=setup)
        while True:
            data = self.sock.recv(recv_buffer)
            print 'data -----: ', data
        # timeit.Timer(self._read_historical_data_socket()).timeit()
        sock.close

        return data

    def get_data_news():
        pass

    def _read_historical_data_socket(self, recv_buffer=4096):
        """
        Read the information from the socket, in a buffered
        fashion, receiving only 4096 bytes at a time.

        Parameters:
        sock - The socket object
        recv_buffer - Amount in bytes to receive per read
        """
        buffer = ""
        data = ""
        while True:
            # print 1
            data = self.sock.recv(recv_buffer)
            # print data
            buffer += data

            # Check if the end message string arrives
            if "!ENDMSG!" in buffer:
                break

        # Remove the end message string
        buffer = buffer[:-12]
        return buffer

    # def write_data(dataframe)

    def send_msg(self, message, host="127.0.0.1", port=9100, recv_buffer=4096):

        # message = "HIT,%s,60,20030101 075000,,,093000,160000,1\n" % sym
        message = message + '\n'
        #message = 'S,SELECT UPDATE FIELDS,Last,Percent Change,Change,Symbol\n' + message + '\n'

        # Open a streaming socket to the IQFeed server locally
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # Send the historical data request
        # message and buffer the data
        sock.sendall(message)
        self.sock = sock
        # print min(timeit.Timer('a=s[:]; timsort(a)', setup=setup)
        data = self.sock.recv(recv_buffer)
        # timeit.Timer(self._read_historical_data_socket()).timeit()
        sock.close

        return data


if __name__ == "__main__":
    # Define server host, port and symbols to download
    host = "127.0.0.1"  # Localhost
    port = 9100  # Historical data socket port
    syms = ["SPY", "AAPL", "GOOG", "AMZN"]

    # Download each symbol to disk
    for sym in syms:
        # import pdb; pdb.set_trace()
        print "Downloading symbol: %s..." % sym

        # Construct the message needed by IQFeed to retrieve data
        message = "HIT,%s,60,20140101 075000,,,093000,160000,1\n" % sym

        # Open a streaming socket to the IQFeed server locally
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # Send the historical data request
        # message and buffer the data
        sock.sendall(message)
        data = read_historical_data_socket(sock)
        sock.close

        # Remove all the endlines and line-ending
        # comma delimiter from each record
        data = "".join(data.split("\r"))
        data = data.replace(",\n", "\n")[:-1]

        # Write the data stream to disk
        f = open("%s.csv" % sym, "w")
        f.write(data)
        f.close()
