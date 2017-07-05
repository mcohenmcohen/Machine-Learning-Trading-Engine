'''
Class DataFeed handles access to IQFeed
Refer to https://www.quantstart.com/articles/Downloading-Historical-Intraday-US-Equities-From-DTN-IQFeed-with-Python
'''
# Author:  Matt Cohen
# Python Version 2.7

import sys
import socket
import pandas as pd
import numpy as np


class DataFeed(object):
    '''
    This class provides IQFeed data access services to retrieve stock symbol data
    '''
    def __init__(self):
        self.sock = None
        self.symbols = []

    def get_data(self, message, host="127.0.0.1", port=9100, recv_buffer=4096):
        '''
        Get data from the IQFeed server.

        Input:  The message to send, plus optional connextion Parameters
        Output: The server's message response
        '''
        # message = "HIT,%s,60,20030101 075000,,,093000,160000,1\n" % sym
        message = message + '\n'
        # message = 'S,SELECT UPDATE FIELDS,Last,Percent Change,Change,Symbol\n' + message + '\n'

        # Open a streaming socket to the IQFeed server locally
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # Send the message and buffer the data.
        # If its a system msg ('S' is the first field) don't buffer it
        sock.sendall(message)
        self.sock = sock

        data = ''
        if message[0:2].upper() == 'S,':
            data = self.sock.recv(recv_buffer)
        else:
            buffer = ''
            while True:
                # print 1
                data = self.sock.recv(recv_buffer)
                # print data
                buffer += data

                # Check if the end message string arrives
                if "!ENDMSG!" in buffer:
                    break

            data = buffer

        sock.close

        return data

    def open(host="127.0.0.1", port=5009, recv_buffer=4096):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        return sock

    def send(sock, message, recv_buffer=4096):
        sock.sendall(message)
        print sock.recv(recv_buffer)

    def get_data_stream(self, sym, host="127.0.0.1", port=5009, recv_buffer=4096):
        '''
        Wrapper for get_data

        Input:  Symbol to stream.  Optional connetion params.
        Output: ? TBD
        '''
        # Open a streaming socket to the IQFeed server locally
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))

        # Send headers
        # Select requested fields
        sock.sendall('S,SELECT UPDATE FIELDS,Last,Percent Change,Change,Symbol\n')
        # Turn off the timestamp feed
        sock.sendall('S,TIMESTAMPSOFF\n')

        # Send the symbol watch message
        message = "w%s\n" % sym
        sock.sendall(message)
        self.sock = sock
        # print min(timeit.Timer('a=s[:]; timsort(a)', setup=setup)
        #while True:
        i = 0
        while i < 10:
            data = self.sock.recv(recv_buffer)
            print 'data -----: ', data
            i += 1
        # timeit.Timer(self._read_historical_data_socket()).timeit()
        sock.close

        return data

    def get_data_news():
        '''
        Wrapper for get_data

        Input:  Symbol to get news for.  Optional connetion params.
        Output: ? TBD
        '''
        pass


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

        data = feed.get_data(message)
        data = "".join(data.split("\r"))
        data = data.replace(",\n", "\n")[:-1]

        # Write the data as .csv to disk
        f = open("%s.csv" % sym, "w")
        f.write(data)
        f.close()
