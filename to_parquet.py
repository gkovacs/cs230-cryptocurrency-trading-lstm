#!/usr/bin/env python3

import paratext
import pandas
import pandas as pd
import lz4.frame
import gzip
import io
import pyarrow.parquet as pq
import pyarrow as pa

'''
filepath = 'cboe/lz4_test/BTCUSD_order_book_20170627.csv.lz4'
#filepath = 'cboe/lz4_test/BTCUSD_order_book_20170627.csv.gz'
df = pandas.read_csv(io.TextIOWrapper(lz4.frame.open(filepath)))
#df = pandas.read_csv(filepath)
#df = paratext.load_csv_to_pandas(gzip.open(filepath).read())
print((df))
'''

from glob import glob
from plumbum.cmd import rm
from numpy import dtype

dtypes = {'Event ID': dtype('int64'), 'Event Date': dtype('O'), 'Event Time': dtype('O'), 'Event Millis': dtype('int64'), 'Order ID': dtype('int64'), 'Execution Options': dtype('O'), 'Event Type': dtype('O'), 'Symbol': dtype('O'), 'Order Type': dtype('O'), 'Side': dtype('O'), 'Limit Price (USD)': dtype('float64'), 'Original Quantity (BTC)': dtype('float64'), 'Gross Notional Value (USD)': dtype('float64'), 'Fill Price (USD)': dtype('float64'), 'Fill Quantity (BTC)': dtype('float64'), 'Total Exec Quantity (BTC)': dtype('float64'), 'Remaining Quantity (BTC)': dtype('float64'), 'Avg Price (USD)': dtype('float64')}

for x in sorted(glob('cboe/parquet_BTCUSD/BTCUSD*.csv.lz4')):
  print(x)
  df = pandas.read_csv(io.TextIOWrapper(lz4.frame.open(x)), dtype=dtypes) # low_memory=False, 
  #df = pd.read_csv('BTCUSD_order_book_20171021.csv', low_memory=False)
  df = df.astype(dtype=dtypes)
  table = pa.Table.from_pandas(df)
  outfile = x.replace('.csv.lz4', '.parquet')
  pq.write_table(table, outfile, compression='snappy')
  rm(x)
