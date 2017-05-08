import psycopg2 as pg2
from sqlalchemy import create_engine


class DBUtils(object):

    def __init__(self):
        self.conn = pg2.connect(dbname='stocksdb', user='mcohen', host='localhost')
        self.engine = create_engine('postgresql://localhost:5432/stocksdb')


    def write_symbol_data(self, dataframe):
        dataframe.to_sql('symbols', self.engine, if_exists='append', index=False)

    def read_symbold_data():
        pass
