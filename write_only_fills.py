#!/usr/bin/env python3

import paratext
import pandas
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

for x in glob('cboe/parquet/*.parquet'):
  print(x)
  outfile = x.replace('cboe/parquet/', 'cboe/parquet_fills_only/')
  df = pq.read_table(x).to_pandas()
  df = df[df['Event Type'] == 'Fill']
  pq.write_table(pa.Table.from_pandas(df), outfile, compression='snappy')
